#!/usr/bin/env python3
# Usage:
#   python3 scripts/evaluate.py
#   python3 scripts/evaluate.py --result_folder dataset/agent_results/baseline/Qwen_Qwen3.5-9B

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_RESULT_FOLDER = "dataset/agent_results/baseline/Qwen_Qwen3.5-9B"
TRUTH_RELATIVE_PATH = Path("dataset/swebench_feature_with_truth.jsonl")
GRANULARITIES = ("files", "classes", "functions")
CHANGE_TYPES = ("added", "edited", "total")


@dataclass(frozen=True)
class InstanceEvaluation:
    instance_id: str
    predictions: dict[str, set[str]]
    ground_truth: dict[str, dict[str, set[str]]]
    is_failure: bool
    failure_reason: str | None = None


@dataclass
class MetricCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class MacroMetricTotals:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    instance_count: int = 0


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate agent result folders against dataset/swebench_feature_with_truth.jsonl."
    )
    parser.add_argument("--result_folder", default=DEFAULT_RESULT_FOLDER)
    return parser.parse_args()


def validate_repo_relative_path(path_text: str) -> str | None:
    if not path_text:
        return "path is empty"
    if path_text.startswith("/"):
        return "path must be repo-relative, not absolute"
    if path_text.startswith("repo/"):
        return "path must be repo-relative and must not start with 'repo/'"
    if "\\" in path_text:
        return "path must use forward slashes"

    path_parts = PurePosixPath(path_text).parts
    if any(part in (".", "..") for part in path_parts):
        return "path must not contain '.' or '..' segments"
    return None


def parse_results_file(results_path: Path) -> tuple[dict[str, set[str]] | None, str | None]:
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "results.txt is missing"
    except json.JSONDecodeError as exc:
        return None, f"results.txt is not valid JSON: {exc.msg}"

    if not isinstance(payload, dict):
        return None, "top-level JSON value must be an object"
    if set(payload) != {"results"}:
        return None, "top-level JSON object must contain exactly one key: 'results'"

    results = payload["results"]
    if not isinstance(results, dict):
        return None, "'results' must be a JSON object"
    if set(results) != set(GRANULARITIES):
        return None, "'results' must contain exactly these keys: files, classes, functions"

    parsed: dict[str, set[str]] = {}
    for field_name in GRANULARITIES:
        entries = results[field_name]
        if not isinstance(entries, list):
            return None, f"'{field_name}' must be a list"
        if any(not isinstance(entry, str) for entry in entries):
            return None, f"'{field_name}' must contain only strings"
        if len(entries) != len(set(entries)):
            return None, f"'{field_name}' contains duplicate entries"
        if entries != sorted(entries):
            return None, f"'{field_name}' must be sorted lexicographically"
        parsed[field_name] = set(entries)

    for file_path in parsed["files"]:
        path_error = validate_repo_relative_path(file_path)
        if path_error:
            return None, f"invalid file entry '{file_path}': {path_error}"

    for field_name in ("classes", "functions"):
        for entry in parsed[field_name]:
            if entry.count("::") != 1:
                return None, f"invalid {field_name[:-1]} entry '{entry}': expected '<file_path>::<qualified_name>'"
            file_path, qualified_name = entry.split("::", 1)
            path_error = validate_repo_relative_path(file_path)
            if path_error:
                return None, f"invalid {field_name[:-1]} entry '{entry}': {path_error}"
            if not qualified_name:
                return None, f"invalid {field_name[:-1]} entry '{entry}': qualified name is empty"

    return parsed, None


def normalize_ground_truth(raw_ground_truth: dict) -> dict[str, dict[str, set[str]]]:
    normalized: dict[str, dict[str, set[str]]] = {
        "added": {granularity: set() for granularity in GRANULARITIES},
        "edited": {granularity: set() for granularity in GRANULARITIES},
        "total": {granularity: set() for granularity in GRANULARITIES},
    }

    for change_type in ("added", "edited"):
        change_payload = raw_ground_truth.get(change_type, {})
        if not isinstance(change_payload, dict):
            continue
        for granularity in GRANULARITIES:
            values = change_payload.get(granularity, [])
            if isinstance(values, list):
                normalized[change_type][granularity] = set(str(value) for value in values)

    for granularity in GRANULARITIES:
        normalized["total"][granularity] = (
            normalized["added"][granularity] | normalized["edited"][granularity]
        )

    return normalized


def load_ground_truth_map(truth_path: Path) -> dict[str, dict[str, dict[str, set[str]]]]:
    ground_truth_map: dict[str, dict[str, dict[str, set[str]]]] = {}
    with truth_path.open("r", encoding="utf-8") as truth_file:
        for line_number, raw_line in enumerate(truth_file, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if row.get("ground_truth_certainty") != "sure":
                continue
            instance_id = row.get("instance_id")
            if not instance_id:
                raise ValueError(f"Missing instance_id in {truth_path} at line {line_number}.")
            ground_truth = row.get("ground_truth")
            if not isinstance(ground_truth, dict):
                raise ValueError(
                    f"Missing or invalid ground_truth for instance {instance_id} in {truth_path} at line {line_number}."
                )
            ground_truth_map[str(instance_id)] = normalize_ground_truth(ground_truth)
    return ground_truth_map


def iter_instance_dirs(result_folder: Path) -> list[Path]:
    return sorted(path for path in result_folder.iterdir() if path.is_dir())


def load_instance_evaluations(
    result_folder: Path,
    ground_truth_map: dict[str, dict[str, dict[str, set[str]]]],
) -> tuple[list[InstanceEvaluation], list[str]]:
    evaluations: list[InstanceEvaluation] = []
    result_instance_dirs = iter_instance_dirs(result_folder)
    skipped_instance_ids = [
        instance_dir.name
        for instance_dir in result_instance_dirs
        if instance_dir.name not in ground_truth_map
    ]

    for instance_id, ground_truth in sorted(ground_truth_map.items()):
        instance_dir = result_folder / instance_id
        if not instance_dir.is_dir():
            evaluations.append(
                InstanceEvaluation(
                    instance_id=instance_id,
                    predictions={granularity: set() for granularity in GRANULARITIES},
                    ground_truth=ground_truth,
                    is_failure=True,
                    failure_reason="instance result directory is missing",
                )
            )
            continue

        predictions, failure_reason = parse_results_file(instance_dir / "results.txt")
        if predictions is None:
            predictions = {granularity: set() for granularity in GRANULARITIES}
            evaluations.append(
                InstanceEvaluation(
                    instance_id=instance_id,
                    predictions=predictions,
                    ground_truth=ground_truth,
                    is_failure=True,
                    failure_reason=failure_reason,
                )
            )
            continue

        evaluations.append(
            InstanceEvaluation(
                instance_id=instance_id,
                predictions=predictions,
                ground_truth=ground_truth,
                is_failure=False,
            )
        )

    return evaluations, skipped_instance_ids


def build_counts(
    evaluations: list[InstanceEvaluation],
    *,
    ignore_failures: bool,
) -> dict[str, dict[str, MetricCounts]]:
    counts = {
        change_type: {granularity: MetricCounts() for granularity in GRANULARITIES}
        for change_type in CHANGE_TYPES
    }

    for evaluation in evaluations:
        if ignore_failures and evaluation.is_failure:
            continue

        for change_type in CHANGE_TYPES:
            for granularity in GRANULARITIES:
                predicted = evaluation.predictions[granularity]
                truth = evaluation.ground_truth[change_type][granularity]
                count = counts[change_type][granularity]
                count.tp += len(predicted & truth)
                count.fp += len(predicted - truth)
                count.fn += len(truth - predicted)

    return counts


def precision_recall_f1(counts: MetricCounts) -> tuple[float, float, float]:
    precision = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def instance_precision_recall_f1(
    predicted: set[str],
    truth: set[str],
) -> tuple[float, float, float]:
    if not predicted and not truth:
        return 1.0, 1.0, 1.0

    counts = MetricCounts(
        tp=len(predicted & truth),
        fp=len(predicted - truth),
        fn=len(truth - predicted),
    )
    return precision_recall_f1(counts)


def build_macro_metrics(
    evaluations: list[InstanceEvaluation],
    *,
    ignore_failures: bool,
) -> dict[str, dict[str, MacroMetricTotals]]:
    totals = {
        change_type: {granularity: MacroMetricTotals() for granularity in GRANULARITIES}
        for change_type in CHANGE_TYPES
    }

    for evaluation in evaluations:
        if ignore_failures and evaluation.is_failure:
            continue

        for change_type in CHANGE_TYPES:
            for granularity in GRANULARITIES:
                precision, recall, f1 = instance_precision_recall_f1(
                    evaluation.predictions[granularity],
                    evaluation.ground_truth[change_type][granularity],
                )
                metric = totals[change_type][granularity]
                metric.precision += precision
                metric.recall += recall
                metric.f1 += f1
                metric.instance_count += 1

    return totals


def format_metric_line(granularity: str, counts: MetricCounts) -> str:
    precision, recall, f1 = precision_recall_f1(counts)
    return (
        f"{granularity}: "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"(tp={counts.tp} fp={counts.fp} fn={counts.fn})"
    )


def format_macro_metric_line(granularity: str, totals: MacroMetricTotals) -> str:
    instance_count = totals.instance_count
    precision = totals.precision / instance_count if instance_count else 0.0
    recall = totals.recall / instance_count if instance_count else 0.0
    f1 = totals.f1 / instance_count if instance_count else 0.0
    return (
        f"{granularity}: "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"(instances={instance_count})"
    )


def render_evaluation_text(
    result_folder: Path,
    evaluations: list[InstanceEvaluation],
    skipped_instance_ids: list[str],
) -> str:
    failed_instance_ids = [evaluation.instance_id for evaluation in evaluations if evaluation.is_failure]
    successful_instance_ids = [evaluation.instance_id for evaluation in evaluations if not evaluation.is_failure]
    missing_result_instance_ids = [
        evaluation.instance_id
        for evaluation in evaluations
        if evaluation.failure_reason == "instance result directory is missing"
    ]

    sections: list[str] = [
        f"result_folder: {result_folder}",
        f"sure_ground_truth_instances: {len(evaluations)}",
        f"total_instance_dirs: {len(evaluations) - len(missing_result_instance_ids) + len(skipped_instance_ids)}",
        f"matched_instances: {len(evaluations) - len(missing_result_instance_ids)}",
        f"missing_result_instance_dirs: {len(missing_result_instance_ids)}",
        f"successful_instances: {len(successful_instance_ids)}",
        f"failed_instances: {len(failed_instance_ids)}",
        f"skipped_instances_without_ground_truth: {len(skipped_instance_ids)}",
        "",
    ]

    if failed_instance_ids:
        sections.append("failed_instance_reasons:")
        for evaluation in evaluations:
            if evaluation.is_failure:
                sections.append(f"{evaluation.instance_id}: {evaluation.failure_reason}")
        sections.append("")

    if skipped_instance_ids:
        sections.append("skipped_instances_without_ground_truth:")
        sections.extend(skipped_instance_ids)
        sections.append("")

    for ignore_failures in (True, False):
        mode_name = "ignore_failed_instances" if ignore_failures else "include_failed_instances"
        included_count = sum(1 for evaluation in evaluations if not (ignore_failures and evaluation.is_failure))
        sections.append(f"[{mode_name}]")
        sections.append(f"included_instances: {included_count}")
        sections.append("")

        sections.append("micro:")
        counts = build_counts(evaluations, ignore_failures=ignore_failures)
        for change_type in CHANGE_TYPES:
            sections.append(f"  {change_type}:")
            for granularity in GRANULARITIES:
                sections.append(f"    {format_metric_line(granularity, counts[change_type][granularity])}")
            sections.append("")

        sections.append("macro:")
        macro_metrics = build_macro_metrics(evaluations, ignore_failures=ignore_failures)
        for change_type in CHANGE_TYPES:
            sections.append(f"  {change_type}:")
            for granularity in GRANULARITIES:
                sections.append(
                    f"    {format_macro_metric_line(granularity, macro_metrics[change_type][granularity])}"
                )
            sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root().resolve()
    result_folder = Path(args.result_folder).expanduser()
    if not result_folder.is_absolute():
        result_folder = (repo_root / result_folder).resolve()

    if not result_folder.exists():
        raise ValueError(f"Result folder does not exist: {result_folder}")
    if not result_folder.is_dir():
        raise ValueError(f"Result folder is not a directory: {result_folder}")

    truth_path = repo_root / TRUTH_RELATIVE_PATH
    if not truth_path.exists():
        raise ValueError(f"Ground truth file does not exist: {truth_path}")

    ground_truth_map = load_ground_truth_map(truth_path)
    evaluations, skipped_instance_ids = load_instance_evaluations(result_folder, ground_truth_map)
    evaluation_text = render_evaluation_text(result_folder, evaluations, skipped_instance_ids)
    (result_folder / "evaluation.txt").write_text(evaluation_text, encoding="utf-8")


if __name__ == "__main__":
    main()
