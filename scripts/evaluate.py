#!/usr/bin/env python3
"""Evaluate REAL-FL, Baseline and no-runtime runs using one scoring implementation.

run writes evaluation.json and evaluation.txt. compare consumes two run JSON
files and writes comparison.json/txt with --output-dir, or prints JSON otherwise.
Old invocations without a subcommand still select run. All evaluation logic is
contained here; compare and paper load NumPy lazily for resampling. Paper mode
recomputes the main VSM comparisons, ablation and repeated-run statistics from
the released data. Restore the patched repositories before scoring raw runs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (CODE_ROOT.parent / "Experiment Data/Benchmark data").is_dir()
DATA_ROOT = CODE_ROOT.parent / "Experiment Data" if PACKAGED else CODE_ROOT / "dataset"
BENCHMARK_ROOT = DATA_ROOT / "Benchmark data" if PACKAGED else DATA_ROOT
DEFAULT_INPUT_PATH = BENCHMARK_ROOT / "swebench_feature_with_truth.jsonl"
DEFAULT_SELECTED_INSTANCES_PATH = BENCHMARK_ROOT / "trace_completed_instance_validated.toml"
DEFAULT_WORKBENCH_PATH = Path("dataset/agent_workbench/baseline")
DEFAULT_RESULT_FOLDER = DATA_ROOT / "agent_results/baseline/Qwen35-9B"
GRANULARITIES = ("files", "classes", "functions")
CHANGE_TYPES = ("added", "edited", "total")
FEATURES = (*GRANULARITIES, "avg3")
SCHEMA_VERSION = 1
SCORING_VERSION = "dalic-agent-metrics-1"
BOOTSTRAP_DRAWS = 20_000
PERMUTATION_DRAWS = 100_000
BOOTSTRAP_SEED = 20260910
PERMUTATION_SEED = 20260911
# One mapping for all conditions. Names also bind the published random seeds.
# The released Baseline run C directory has an uppercase B; its historical
# model label is retained below so its confidence intervals remain reproducible.
PAPER_RUNS = {
    "Qwen3.5-4B": {
        "dalic_vsm": "Qwen35-4B-a(main)", "baseline": "Qwen3.5-4B-a(main)",
        "dalic_no_runtime": "Qwen35-4B-no-runtime",
        "repeats": {"dalic_vsm": ("Qwen35-4B-b", "Qwen35-4B-c"),
                    "baseline": ("Qwen35-4B-b", "Qwen35-4B-c")},
    },
    "Qwen3.5-9B": {
        "dalic_vsm": "Qwen35-9B", "baseline": "Qwen35-9B",
        "dalic_no_runtime": "Qwen35-9B-no-runtime",
    },
    "Qwen3.5-27B": {
        "dalic_vsm": "Qwen35-27B", "baseline": "Qwen3.5-27B",
        "dalic_no_runtime": "Qwen35-27B-no-runtime",
    },
    "GPT-5.4": {
        "dalic_vsm": "gpt54-0305-openai-a(main)", "baseline": "gpt54-0305-openai-a(main)",
        "dalic_no_runtime": "gpt54-0305-openai",
        "repeats": {"dalic_vsm": ("gpt54-0305-openai-b", "gpt54-0305-openai-c"),
                    "baseline": ("gpt54-0305-openai-b", "gpt54-0305-openai-c")},
    },
    "DeepSeek-v4-Flash": {
        "dalic_vsm": "DeepSeek-v4-flash-0423", "baseline": "DeepSeek-v4-flash-0423",
        "dalic_no_runtime": "DeepSeek-v4-flash-0423",
    },
    "DeepSeek-v4-Pro": {
        "dalic_vsm": "DeepSeek-v4-pro-0423", "baseline": "DeepSeek-v4-pro-0423",
        "dalic_no_runtime": "DeepSeek-v4-pro-0423",
    },
    "DeepSeek-v4-Pro-32K": {
        "dalic_vsm": "DeepSeek-v4-pro-0423-32K", "baseline": "DeepSeek-v4-pro-0423-32K",
        "dalic_no_runtime": "DeepSeek-v4-pro-0423-32K",
    },
}
PAPER_MODELS = tuple(PAPER_RUNS)
RUN_LABEL_ALIASES = {("baseline", "Qwen35-4B-c"): "Qwen35-4b-c"}
STATIC_METHODS = ("static_vsm", "static_lsi", "static_bm25")
PAPER_CUTOFFS = (3, 5, 10, 20)
RETRIEVAL_LABELS = {
    "static_vsm": "Static VSM", "static_lsi": "Static LSI",
    "static_bm25": "Static BM25", "trace_vsm": "Trace VSM",
}


@dataclass(frozen=True)
class InstanceEvaluation:
    instance_id: str
    predictions: dict[str, set[str]]
    ground_truth: dict[str, dict[str, set[str]]]
    is_failure: bool
    failure_reason: str | None = None
    # Preserve the set-based API for existing callers, and the original order
    # for AP. Never reconstruct a ranking from a set.
    ranked_predictions: dict[str, list[str]] | None = None
    result_sha256: str | None = None


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
    # Never discover a parent development checkout outside this package.
    return CODE_ROOT


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("run", help="Score one agent result directory.")
    run.add_argument("--input", type=Path, default=repo_root / DEFAULT_INPUT_PATH)
    run.add_argument(
        "--selected-instances", "--selected_instances", type=Path,
        default=repo_root / DEFAULT_SELECTED_INSTANCES_PATH,
        help="Validated benchmark manifest or a nonempty subset of it.",
    )
    run.add_argument(
        "--workbench-path", "--workbench_path", type=Path,
        default=repo_root / DEFAULT_WORKBENCH_PATH,
        help="Workbench containing each instance's checked-out repository.",
    )
    run.add_argument("--result-folder", "--result_folder", type=Path, default=DEFAULT_RESULT_FOLDER)
    run.add_argument("--method", help="Defaults to the result directory's parent name.")
    run.add_argument("--model", help="Defaults to the result directory name.")
    run.add_argument("--output-dir", type=Path,
                     help="Defaults to the result directory for custom runs, or dataset/evaluation/runs for archived runs.")
    compare = modes.add_parser("compare", help="Compare two run-mode evaluation.json files.")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path,
                         help="Differences are candidate minus baseline.")
    compare.add_argument("--output-dir", type=Path,
                         help="Write comparison.json/txt here; otherwise print JSON.")
    compare.add_argument("--tests", choices=("none", "paired"), default="none",
                         help="paired adds unadjusted All Instances McNemar and Avg3 sign-flip tests.")
    compare.add_argument("--bootstrap-draws", type=positive_int, default=BOOTSTRAP_DRAWS)
    compare.add_argument("--permutation-draws", type=positive_int, default=PERMUTATION_DRAWS)
    compare.add_argument("--bootstrap-seed", type=nonnegative_int, default=BOOTSTRAP_SEED)
    compare.add_argument("--permutation-seed", type=nonnegative_int, default=PERMUTATION_SEED)
    paper = modes.add_parser("paper", help="Compute RQ1-RQ3 from agent outputs and cached IR rankings.")
    paper.add_argument("--input", type=Path, default=repo_root / DEFAULT_INPUT_PATH)
    paper.add_argument("--selected-instances", "--selected_instances", type=Path,
                       default=repo_root / DEFAULT_SELECTED_INSTANCES_PATH)
    paper.add_argument("--agent-results-root", type=Path,
                       default=DATA_ROOT / "agent_results")
    paper.add_argument("--rankings-root", type=Path,
                       default=DATA_ROOT / "trad_methods_results/rankings")
    paper.add_argument("--workbench-path", type=Path, default=repo_root / DEFAULT_WORKBENCH_PATH,
                       help="Restored patched repositories, shared by all three conditions.")
    paper.add_argument("--models", nargs="+", choices=PAPER_MODELS, default=list(PAPER_MODELS),
                       help="Human-readable model names from PAPER_RUNS. Subsets change Holm family sizes.")
    paper.add_argument("--output-dir", type=Path, default=repo_root / "dataset/evaluation/paper")
    paper.add_argument("--bootstrap-draws", type=positive_int, default=BOOTSTRAP_DRAWS)
    paper.add_argument("--permutation-draws", type=positive_int, default=PERMUTATION_DRAWS)
    paper.add_argument("--bootstrap-seed", type=nonnegative_int, default=BOOTSTRAP_SEED)
    paper.add_argument("--permutation-seed", type=nonnegative_int, default=PERMUTATION_SEED)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or (arguments[0].startswith("-") and arguments[0] not in ("-h", "--help")):
        arguments.insert(0, "run")
    args = parser.parse_args(arguments)
    if args.mode == "run":
        validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    args.input = args.input.expanduser().resolve()
    args.selected_instances = args.selected_instances.expanduser().resolve()
    args.workbench_path = args.workbench_path.expanduser().resolve()

    if not args.input.exists():
        raise ValueError(f"Input file does not exist: {args.input}")
    if not args.input.is_file():
        raise ValueError(f"Input path is not a file: {args.input}")
    if args.input.suffix != ".jsonl":
        raise ValueError(f"Input file must be a .jsonl file: {args.input}")
    if not args.selected_instances.is_file():
        raise ValueError(f"Selected-instance manifest is missing: {args.selected_instances}")
    if not args.workbench_path.exists():
        raise ValueError(f"Workbench path does not exist: {args.workbench_path}")
    if not args.workbench_path.is_dir():
        raise ValueError(f"Workbench path is not a directory: {args.workbench_path}")


def find_instance_repository_root(workbench_dir: Path, instance_id: str) -> Path:
    repository_parent = workbench_dir / instance_id / "repo"
    if not repository_parent.is_dir():
        raise ValueError(f"Repository parent directory is missing: {repository_parent}")

    repository_roots = sorted(
        path for path in repository_parent.iterdir() if path.is_dir()
    )
    if len(repository_roots) != 1:
        raise ValueError(
            f"Expected exactly one repository root under {repository_parent}, "
            f"found {len(repository_roots)}"
        )
    return repository_roots[0]


def validate_repo_relative_path(
    path_text: str, repository_root: Path | None
) -> str | None:
    if not path_text:
        return "path is empty"
    if path_text.startswith("/"):
        return "path must be repo-relative, not absolute"
    if path_text.startswith("repo/"):
        return "path must be repo-relative and must not start with 'repo/'"
    if "\\" in path_text:
        return "path must use forward slashes"

    path_parts = path_text.split("/")
    if any(part in (".", "..") for part in path_parts):
        return "path must not contain '.' or '..' segments"
    if any(not part for part in path_parts):
        return "path must not contain empty segments"
    if repository_root is None:
        return None

    candidate_path = repository_root.joinpath(*path_parts)
    if (
        path_parts
        and path_parts[0] == repository_root.name
        and not candidate_path.is_file()
    ):
        stripped_path = repository_root.joinpath(*path_parts[1:])
        if stripped_path.is_file():
            return (
                f"path contains redundant checkout directory prefix "
                f"'{repository_root.name}/'; use "
                f"'{PurePosixPath(*path_parts[1:])}'"
            )
    return None


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_ranked_results(
    results_path: Path, repository_root: Path,
) -> tuple[dict[str, list[str]] | None, str | None, str | None]:
    """Invalid content is an agent failure; filesystem access errors propagate."""
    try:
        raw = results_path.read_bytes()
    except FileNotFoundError:
        return None, "results.txt is missing", None
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_json_object)
    except (ValueError, UnicodeError) as exc:
        return None, f"results.txt is not valid UTF-8 JSON: {exc}", digest
    ranked, reason = validate_ranked_payload(payload, repository_root)
    return ranked, reason, digest


def parse_results_file(
    results_path: Path, repository_root: Path,
) -> tuple[dict[str, set[str]] | None, str | None]:
    """Compatibility API for callers that only need membership predictions."""
    ranked, reason, _ = read_ranked_results(results_path, repository_root)
    return (None if ranked is None else {g: set(ranked[g]) for g in GRANULARITIES}), reason


def validate_ranked_payload(
    payload: Any, repository_root: Path | None = None,
) -> tuple[dict[str, list[str]] | None, str | None]:
    if not isinstance(payload, dict):
        return None, "top-level JSON value must be an object"
    if set(payload) != {"results"}:
        return None, "top-level JSON object must contain exactly one key: 'results'"

    results = payload["results"]
    if not isinstance(results, dict):
        return None, "'results' must be a JSON object"
    if set(results) != set(GRANULARITIES):
        return None, "'results' must contain exactly these keys: files, classes, functions"

    parsed: dict[str, list[str]] = {}
    for field_name in GRANULARITIES:
        entries = results[field_name]
        if not isinstance(entries, list):
            return None, f"'{field_name}' must be a list"
        if any(not isinstance(entry, str) for entry in entries):
            return None, f"'{field_name}' must contain only strings"
        if len(entries) != len(set(entries)):
            return None, f"'{field_name}' contains duplicate entries"
        parsed[field_name] = list(entries)

    for file_path in parsed["files"]:
        path_error = validate_repo_relative_path(file_path, repository_root)
        if path_error:
            return None, f"invalid file entry '{file_path}': {path_error}"

    for field_name in ("classes", "functions"):
        for entry in parsed[field_name]:
            if entry.count("::") != 1:
                return None, f"invalid {field_name[:-1]} entry '{entry}': expected '<file_path>::<qualified_name>'"
            file_path, qualified_name = entry.split("::", 1)
            path_error = validate_repo_relative_path(file_path, repository_root)
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
            raise ValueError(f"ground_truth.{change_type} must be an object")
        for granularity in GRANULARITIES:
            values = change_payload.get(granularity, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"ground_truth.{change_type}.{granularity} must be a string list")
            normalized[change_type][granularity] = set(values)

    for granularity in GRANULARITIES:
        normalized["total"][granularity] = (
            normalized["added"][granularity] | normalized["edited"][granularity]
        )

    return normalized


def load_ground_truth_map(
    truth_path: Path, selected_instance_ids: set[str] | None = None,
) -> dict[str, dict[str, dict[str, set[str]]]]:
    ground_truth_map: dict[str, dict[str, dict[str, set[str]]]] = {}
    with truth_path.open("r", encoding="utf-8") as truth_file:
        for line_number, raw_line in enumerate(truth_file, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if selected_instance_ids is not None and row.get("instance_id") not in selected_instance_ids:
                continue
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
            if instance_id in ground_truth_map:
                raise ValueError(f"Duplicate ground truth for {instance_id}")
            ground_truth_map[str(instance_id)] = normalize_ground_truth(ground_truth)
    if selected_instance_ids is not None:
        return filter_selected_ground_truth(ground_truth_map, selected_instance_ids)
    return ground_truth_map


def load_selected_instance_ids(selected_instances_path: Path) -> set[str]:
    with selected_instances_path.open("rb") as selected_instances_file:
        payload = tomllib.load(selected_instances_file)
    ids = payload.get("instance_ids")
    if not isinstance(ids, list) or not ids or any(
        not isinstance(i, str) or not i or "/" in i or "\\" in i or i in (".", "..") for i in ids
    ):
        raise ValueError("instance_ids must be a nonempty list of directory names")
    if len(ids) != len(set(ids)):
        raise ValueError("Selected-instance manifest contains duplicate IDs")
    return set(ids)


def validate_benchmark_ids(ids: set[str]) -> None:
    allowed = load_selected_instance_ids(get_repo_root() / DEFAULT_SELECTED_INSTANCES_PATH)
    outside = ids - allowed
    if outside:
        raise ValueError(f"Instances outside the validated benchmark: {', '.join(sorted(outside))}")


def filter_selected_ground_truth(
    ground_truth_map: dict[str, dict[str, dict[str, set[str]]]],
    selected_instance_ids: set[str],
) -> dict[str, dict[str, dict[str, set[str]]]]:
    missing = selected_instance_ids - ground_truth_map.keys()
    if missing:
        raise ValueError(f"Selected instances lack sure ground truth: {', '.join(sorted(missing))}")
    return {
        instance_id: ground_truth
        for instance_id, ground_truth in ground_truth_map.items()
        if instance_id in selected_instance_ids
    }


def iter_instance_dirs(result_folder: Path) -> list[Path]:
    return sorted(path for path in result_folder.iterdir() if path.is_dir())


def load_instance_evaluations(
    result_folder: Path,
    ground_truth_map: dict[str, dict[str, dict[str, set[str]]]],
    workbench_dir: Path,
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

        if not (instance_dir / "results.txt").exists():
            evaluations.append(InstanceEvaluation(
                instance_id, {g: set() for g in GRANULARITIES}, ground_truth,
                True, "results.txt is missing",
            ))
            continue
        repository_root = find_instance_repository_root(workbench_dir, instance_id)
        ranked, failure_reason, result_sha256 = read_ranked_results(
            instance_dir / "results.txt", repository_root
        )
        if ranked is None:
            predictions = {granularity: set() for granularity in GRANULARITIES}
            evaluations.append(
                InstanceEvaluation(
                    instance_id=instance_id,
                    predictions=predictions,
                    ground_truth=ground_truth,
                    is_failure=True,
                    failure_reason=failure_reason,
                    result_sha256=result_sha256,
                )
            )
            continue

        evaluations.append(
            InstanceEvaluation(
                instance_id=instance_id,
                predictions={g: set(ranked[g]) for g in GRANULARITIES},
                ground_truth=ground_truth,
                is_failure=False,
                ranked_predictions=ranked,
                result_sha256=result_sha256,
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
                predicted = set() if evaluation.is_failure else evaluation.predictions[granularity]
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
                precision, recall, f1 = (
                    (0.0, 0.0, 0.0) if evaluation.is_failure
                    else instance_precision_recall_f1(
                        evaluation.predictions[granularity],
                        evaluation.ground_truth[change_type][granularity],
                    )
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
    if not instance_count:
        return f"{granularity}: precision=undefined recall=undefined f1=undefined (instances=0)"
    precision = totals.precision / instance_count
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


def average_precision(ranked: list[str], truth: set[str]) -> float | None:
    """Use the entire GT as denominator; empty GT has no defined AP."""
    if not truth:
        return None
    hits = 0
    total = 0.0
    for rank, entity in enumerate(ranked, 1):
        if entity in truth:
            hits += 1
            total += hits / rank
    return total / len(truth)


def score_ranking(ranked: list[str], truth: set[str], valid: bool) -> dict[str, Any]:
    if len(ranked) != len(set(ranked)):
        raise ValueError("A ranking must not contain duplicate entities")
    predicted = set(ranked) if valid else set()
    precision, recall, f1 = (
        instance_precision_recall_f1(predicted, truth) if valid else (0.0, 0.0, 0.0)
    )
    return {
        "prediction_count": len(predicted), "ground_truth_count": len(truth),
        "tp": len(predicted & truth), "fp": len(predicted - truth),
        "fn": len(truth - predicted),
        "precision": precision, "recall": recall, "f1": f1,
        "ranking_eligible": bool(truth),
        "ap": average_precision(ranked, truth) if valid else (0.0 if truth else None),
    }


def instance_record(evaluation: InstanceEvaluation) -> dict[str, Any]:
    valid = not evaluation.is_failure
    ranked = evaluation.ranked_predictions
    if valid and ranked is None:
        raise ValueError(f"Ordered predictions are missing for {evaluation.instance_id}")
    if not valid:
        ranked = {g: [] for g in GRANULARITIES}
    assert ranked is not None
    truth = {
        change: {g: sorted(evaluation.ground_truth[change][g]) for g in GRANULARITIES}
        for change in CHANGE_TYPES
    }
    metrics = {
        g: score_ranking(ranked[g], evaluation.ground_truth["total"][g], valid)
        for g in GRANULARITIES
    }
    complete = all(metrics[g]["ranking_eligible"] for g in GRANULARITIES)
    metrics["avg3"] = {
        **{m: sum(metrics[g][m] for g in GRANULARITIES) / 3
           for m in ("precision", "recall", "f1")},
        "ranking_eligible": complete,
        "ap": sum(metrics[g]["ap"] for g in GRANULARITIES) / 3 if complete else None,
    }
    return {
        "instance_id": evaluation.instance_id, "valid": valid,
        "failure_reason": evaluation.failure_reason, "result_sha256": evaluation.result_sha256,
        "predictions": ranked, "ground_truth": truth, "metrics": metrics,
    }


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_records(records: list[dict]) -> dict:
    metrics = {}
    for g in FEATURES:
        eligible = [r for r in records if r["metrics"][g]["ranking_eligible"]]
        metrics[g] = {
            "n": len(records), "ranking_n": len(eligible),
            "ranking_instance_ids": [r["instance_id"] for r in eligible],
            **{m: mean_or_none([r["metrics"][g][m] for r in records])
               for m in ("precision", "recall", "f1")},
            "map": mean_or_none([r["metrics"][g]["ap"] for r in eligible]),
        }
    valid_n = sum(r["valid"] for r in records)
    return {
        "n": len(records), "instance_ids": [r["instance_id"] for r in records],
        "valid_n": valid_n, "invalid_n": len(records) - valid_n,
        "valid_rate": valid_n / len(records) if records else None, "metrics": metrics,
    }


def digest_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def record_fingerprints(records: list[dict]) -> dict[str, str]:
    return {
        "cohort_sha256": digest_json([r["instance_id"] for r in records]),
        "ground_truth_sha256": digest_json({r["instance_id"]: r["ground_truth"] for r in records}),
        "records_sha256": digest_json(records),
    }


def scoring_policy() -> dict[str, str]:
    return {
        "ground_truth": "total = added union edited; exact entity matching",
        "membership": "instance macro mean; valid double-empty P/R/F1=1; one-empty=0",
        "invalid": "zero P/R/F1; AP=0 when GT is nonempty, otherwise undefined",
        "ranking": "returned order; AP denominator is full GT size; exclude empty GT",
        "avg3": "unweighted within-instance mean; Avg3 AP requires three nonempty GT sets",
        "valid_only": "single-run valid subset, not pairwise common success",
        "units": "proportions; multiply differences by 100 for percentage points",
    }


def build_run_report(
    evaluations: list[InstanceEvaluation], *, method: str, model: str,
    sources: dict | None = None, skipped_ids: list[str] | None = None,
) -> dict:
    if not evaluations:
        raise ValueError("Cannot evaluate an empty run cohort")
    if not isinstance(method, str) or not method or not isinstance(model, str) or not model:
        raise ValueError("Method and model labels must be nonempty strings")
    records = [instance_record(e) for e in sorted(evaluations, key=lambda e: e.instance_id)]
    ids = [r["instance_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate instance IDs in run")
    return {
        "schema_version": SCHEMA_VERSION, "kind": "run",
        "metadata": {
            "scoring_version": SCORING_VERSION, "method": method, "model": model,
            "policy": scoring_policy(), "sources": sources or {},
            **record_fingerprints(records),
            "skipped_result_instance_ids": sorted(skipped_ids or []),
        },
        "instances": records,
        "scopes": {
            "all_instances": summarize_records(records),
            "valid_only": summarize_records([r for r in records if r["valid"]]),
        },
    }


def load_run_report(path: Path) -> dict:
    """Validate a saved run and recompute its scores from its ordered predictions.

    No checkout or raw results are needed during comparison. Serialized
    aggregates are never used as inputs to a statistical calculation.
    """
    report = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_json_object)
    if not isinstance(report, dict) or report.get("kind") != "run" or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} is not a supported run-mode evaluation JSON")
    metadata = report.get("metadata", {})
    if metadata.get("scoring_version") != SCORING_VERSION or metadata.get("policy") != scoring_policy():
        raise ValueError(f"{path} uses a different scoring policy; regenerate it with run")
    records = report.get("instances")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} has no instance records")
    evaluations = []
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
            raise ValueError(f"Invalid instance record in {path}")
        instance_id = row["instance_id"]
        valid = row.get("valid")
        if type(valid) is not bool:
            raise ValueError(f"Invalid completion status for {instance_id}")
        reason = row.get("failure_reason")
        if (valid and reason is not None) or (not valid and (not isinstance(reason, str) or not reason)):
            raise ValueError(f"Inconsistent failure reason for {instance_id}")
        ranked, error = validate_ranked_payload({"results": row.get("predictions")})
        if error or ranked is None:
            raise ValueError(f"Invalid saved predictions for {instance_id}: {error}")
        if not valid and any(ranked.values()):
            raise ValueError(f"Failed instance has nonempty saved predictions: {instance_id}")
        raw_truth = row.get("ground_truth")
        if not isinstance(raw_truth, dict) or set(raw_truth) != set(CHANGE_TYPES):
            raise ValueError(f"Invalid saved ground truth for {instance_id}")
        truth = normalize_ground_truth(raw_truth)
        canonical = {c: {g: sorted(truth[c][g]) for g in GRANULARITIES} for c in CHANGE_TYPES}
        if raw_truth != canonical:
            raise ValueError(f"Inconsistent ground-truth union or noncanonical entities for {instance_id}")
        item = InstanceEvaluation(
            instance_id, {g: set(ranked[g]) for g in GRANULARITIES}, truth,
            not valid, reason, ranked, row.get("result_sha256"),
        )
        if row != instance_record(item):
            raise ValueError(f"Saved scores disagree with predictions/ground truth for {instance_id}")
        evaluations.append(item)
    rebuilt = build_run_report(
        evaluations, method=metadata.get("method"), model=metadata.get("model"),
        sources=metadata.get("sources"), skipped_ids=metadata.get("skipped_result_instance_ids"),
    )
    validate_benchmark_ids({e.instance_id for e in evaluations})
    for key, value in record_fingerprints(rebuilt["instances"]).items():
        if metadata.get(key) != value:
            raise ValueError(f"{path} has an inconsistent {key}")
    return rebuilt


def derived_seed(base: int, key: str) -> int:
    return base + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % 10**9


def bootstrap_differences(
    differences: list[list[float]], columns: list[str], *, draws: int, seed: int,
) -> dict:
    """Resample paired instance differences, sharing indices across columns."""
    import numpy as np

    if draws <= 0 or seed < 0:
        raise ValueError("Bootstrap draws must be positive and seed nonnegative")
    if not differences:
        return {g: {"n": 0, "estimate": None, "ci_95": None, "seed": seed} for g in columns}
    delta = np.asarray(differences, dtype=float)
    if delta.shape != (len(differences), len(columns)) or not np.isfinite(delta).all():
        raise ValueError("Invalid paired difference matrix")
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, len(columns)))
    for start in range(0, draws, 500):
        size = min(500, draws - start)
        indices = rng.integers(0, len(delta), (size, len(delta)))
        samples[start:start + size] = delta[indices].mean(axis=1)
    return {
        g: {
            "n": len(delta), "estimate": float(delta[:, j].mean()),
            "ci_95": [float(np.percentile(samples[:, j], q)) for q in (2.5, 97.5)],
            "seed": seed,
        }
        for j, g in enumerate(columns)
    }


def sign_flip_test(differences: list[float], *, draws: int, seed: int) -> dict:
    import numpy as np

    if draws <= 0 or seed < 0:
        raise ValueError("Sign-flip draws must be positive and seed nonnegative")
    delta = np.asarray(differences, dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite paired differences")
    result = {"n": len(delta), "draws": draws, "seed": seed, "raw_p": None, "estimate": None}
    if not len(delta):
        return result
    observed = abs(float(delta.mean()))
    result["estimate"] = float(delta.mean())
    if not np.any(delta):
        result["raw_p"] = 1.0
        return result
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, draws, 2000):
        size = min(2000, draws - start)
        values = np.abs((rng.choice((-1.0, 1.0), (size, len(delta))) * delta).mean(axis=1))
        exceed += int(np.count_nonzero(values >= observed - 1e-15))
    result["raw_p"] = (exceed + 1) / (draws + 1)
    return result


def exact_mcnemar(baseline_only: int, candidate_only: int) -> float:
    n = baseline_only + candidate_only
    if baseline_only < 0 or candidate_only < 0:
        raise ValueError("McNemar cell counts must be nonnegative")
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(baseline_only, candidate_only) + 1)) / 2**n)


def aligned_records(baseline: dict, candidate: dict) -> tuple[list[dict], list[dict]]:
    if baseline["metadata"]["scoring_version"] != candidate["metadata"]["scoring_version"]:
        raise ValueError("Cannot compare different scoring versions")
    left = {r["instance_id"]: r for r in baseline["instances"]}
    right = {r["instance_id"]: r for r in candidate["instances"]}
    if len(left) != len(baseline["instances"]) or len(right) != len(candidate["instances"]):
        raise ValueError("Duplicate comparison instance IDs")
    if not left or left.keys() != right.keys():
        raise ValueError("Comparison requires identical, nonempty instance cohorts")
    ids = sorted(left)
    for i in ids:
        if left[i]["ground_truth"] != right[i]["ground_truth"]:
            raise ValueError(f"Ground truth differs between runs for {i}")
    return [left[i] for i in ids], [right[i] for i in ids]


def compare_scope(
    baseline: list[dict], candidate: list[dict], *, key: str,
    draws: int = BOOTSTRAP_DRAWS, seed_base: int = BOOTSTRAP_SEED,
) -> dict:
    """Inputs are already paired, including any common-success restriction."""
    import numpy as np

    if [r["instance_id"] for r in baseline] != [r["instance_id"] for r in candidate]:
        raise ValueError("Scope records must be paired by instance ID")
    left, right = summarize_records(baseline), summarize_records(candidate)
    effects = {
        m: {g: {"n": len(baseline), "estimate": mean_or_none([
            c["metrics"][g][m] - b["metrics"][g][m] for b, c in zip(baseline, candidate)
        ])} for g in FEATURES}
        for m in ("precision", "recall")
    }
    # This matches the paper script's resampling and within-instance reduction.
    delta = np.asarray([
        [c["metrics"][g]["f1"] - b["metrics"][g]["f1"] for g in GRANULARITIES]
        for b, c in zip(baseline, candidate)
    ], dtype=float).reshape((-1, 3))
    four = np.column_stack((delta, delta.mean(axis=1))) if len(delta) else np.empty((0, 4))
    effects["f1"] = bootstrap_differences(
        four.tolist(), list(FEATURES), draws=draws, seed=derived_seed(seed_base, key + ":f1"),
    )
    effects["map"] = {}
    for g in FEATURES:
        differences = []
        for b, c in zip(baseline, candidate):
            if not b["metrics"][g]["ranking_eligible"]:
                continue
            if g == "avg3":
                difference = sum(c["metrics"][k]["ap"] - b["metrics"][k]["ap"] for k in GRANULARITIES) / 3
            else:
                difference = c["metrics"][g]["ap"] - b["metrics"][g]["ap"]
            differences.append([difference])
        effects["map"][g] = bootstrap_differences(
            differences, [g], draws=draws, seed=derived_seed(seed_base, key + ":map:" + g),
        )[g]
    return {"baseline": left, "candidate": right, "effects": effects}


def inference_metadata(draws: int, seed: int, permutation_draws: int, permutation_seed: int) -> dict:
    return {
        "direction": "candidate minus baseline", "units": "proportions",
        "bootstrap": {"draws": draws, "seed_base": seed, "unit": "paired instance",
                      "ci": "pointwise percentile 95%", "seed_derivation": "SHA-256 key plus base"},
        "randomization": {"draws": permutation_draws, "seed_base": permutation_seed,
                          "test": "two-sided paired sign flip", "p": "(exceedances+1)/(draws+1)"},
        "common_success": "pairwise valid intersection; ranking also requires nonempty GT",
        "empty_cohort": "n=0 and null estimates, intervals, and p-values",
    }


def compare_run_reports(
    baseline: dict, candidate: dict, *, tests: str = "none",
    bootstrap_draws: int = BOOTSTRAP_DRAWS, permutation_draws: int = PERMUTATION_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED, permutation_seed: int = PERMUTATION_SEED,
) -> dict:
    if tests not in ("none", "paired"):
        raise ValueError("tests must be none or paired")
    left, right = aligned_records(baseline, candidate)
    bm, cm = baseline["metadata"], candidate["metadata"]
    # Preserve existing paper seeds for same-model Baseline/DALIC comparisons.
    if bm["method"] == "baseline" and cm["method"] == "dalic" and bm["model"] == cm["model"]:
        key = f"rq3:{bm['model']}"
    else:
        key = "compare:" + digest_json([bm["method"], bm["model"], cm["method"], cm["model"]])
    common = [(b, c) for b, c in zip(left, right) if b["valid"] and c["valid"]]
    common_left, common_right = [p[0] for p in common], [p[1] for p in common]
    scopes = {
        name: compare_scope(b, c, key=key + suffix, draws=bootstrap_draws, seed_base=bootstrap_seed)
        for name, b, c, suffix in (
            ("all_instances", left, right, ":all"),
            ("common_success", common_left, common_right, ":common"),
        )
    }
    cells = {
        "both_valid": sum(b["valid"] and c["valid"] for b, c in zip(left, right)),
        "baseline_only": sum(b["valid"] and not c["valid"] for b, c in zip(left, right)),
        "candidate_only": sum(not b["valid"] and c["valid"] for b, c in zip(left, right)),
        "both_invalid": sum(not b["valid"] and not c["valid"] for b, c in zip(left, right)),
    }
    completion = {
        "n": len(left), "baseline_valid": sum(r["valid"] for r in left),
        "candidate_valid": sum(r["valid"] for r in right), "cells": cells,
        "effect": bootstrap_differences(
            [[int(c["valid"]) - int(b["valid"])] for b, c in zip(left, right)],
            ["completion"], draws=bootstrap_draws,
            seed=derived_seed(bootstrap_seed, key + ":completion"),
        )["completion"],
    }
    completion["baseline_rate"] = completion["baseline_valid"] / len(left)
    completion["candidate_rate"] = completion["candidate_valid"] / len(left)
    test_results = {}
    if tests == "paired":
        test_results["completion"] = {
            "n": len(left), "test": "exact two-sided McNemar",
            "raw_p": exact_mcnemar(cells["baseline_only"], cells["candidate_only"]),
        }
        for metric, field in (("f1", "f1"), ("map", "ap")):
            differences = [
                c["metrics"]["avg3"][field] - b["metrics"]["avg3"][field]
                for b, c in zip(left, right)
                if metric == "f1" or b["metrics"]["avg3"]["ranking_eligible"]
            ]
            test_results[metric] = sign_flip_test(
                differences, draws=permutation_draws,
                seed=derived_seed(permutation_seed, key + ":" + metric),
            )
    return {
        "schema_version": SCHEMA_VERSION, "kind": "compare",
        "metadata": {
            "scoring_version": SCORING_VERSION,
            "baseline": {k: bm[k] for k in ("method", "model", "records_sha256")},
            "candidate": {k: cm[k] for k in ("method", "model", "records_sha256")},
            **inference_metadata(bootstrap_draws, bootstrap_seed, permutation_draws, permutation_seed),
            "tests": tests, "multiple_testing": "none; raw p-values for this pair only",
        },
        "completion": completion, "scopes": scopes, "tests": test_results,
    }


def display_number(value: float | None) -> str:
    return "undefined" if value is None else f"{100 * value:.2f}"


def summary_lines(scope: dict) -> list[str]:
    lines = [f"instances: {scope['n']}", f"valid: {scope['valid_n']} ({display_number(scope['valid_rate'])}%)",
             "granularity  N  ranking_N  P(%)  R(%)  F1(%)  MAP(%)"]
    for g, values in scope["metrics"].items():
        lines.append(f"{g}  {values['n']}  {values['ranking_n']}  " + "  ".join(
            display_number(values[m]) for m in ("precision", "recall", "f1", "map")
        ))
    return lines


def effect_text(effect: dict) -> str:
    if effect["estimate"] is None:
        return "undefined (N=0)"
    interval = effect.get("ci_95")
    ci = "" if interval is None else f" [{display_number(interval[0])}, {display_number(interval[1])}]"
    return display_number(effect["estimate"]) + ci


def comparison_scope_lines(name: str, scope: dict) -> list[str]:
    lines = [f"[{name}]", "Baseline:", *summary_lines(scope["baseline"]),
             "Candidate:", *summary_lines(scope["candidate"]),
             "Differences in percentage points; F1/MAP have pointwise 95% CIs:",
             "granularity  delta_P  delta_R  delta_F1 [CI]  delta_MAP [CI]"]
    for g in FEATURES:
        lines.append(g + "  " + "  ".join(
            effect_text(scope["effects"][m][g]) for m in ("precision", "recall", "f1", "map")
        ))
    return lines


def render_compare_text(report: dict) -> str:
    lines = ["candidate minus baseline", report["metadata"]["multiple_testing"],
             f"completion delta (pp) [95% CI]: {effect_text(report['completion']['effect'])}",
             f"completion cells: {report['completion']['cells']}", ""]
    for name, scope in report["scopes"].items():
        lines.extend(comparison_scope_lines(name, scope))
        lines.append("")
    for metric, result in report["tests"].items():
        lines.append(f"{metric}: N={result['n']} raw_p={result['raw_p']}")
    return "\n".join(lines) + "\n"


def write_report(report: dict, text: str, output_dir: Path, stem: str) -> None:
    output_dir = output_dir.expanduser().resolve()
    protected = [DATA_ROOT] if PACKAGED else [DATA_ROOT / p for p in (
        "agent_results", "results_archive", "trad_methods_results")]
    if any(output_dir.is_relative_to(p.resolve()) for p in protected):
        raise ValueError("Evaluation output must be outside the input archives")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    (output_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    print(f"Wrote {output_dir / (stem + '.json')} and {output_dir / (stem + '.txt')}")


def source_file(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_from_paths(
    result_folder: Path, workbench: Path, truth: dict, *, method: str, model: str,
    sources: dict | None = None,
) -> tuple[dict, list[InstanceEvaluation], list[str]]:
    if not result_folder.is_dir():
        raise ValueError(f"Result folder does not exist: {result_folder}")
    if not workbench.is_dir():
        raise ValueError(f"Workbench does not exist: {workbench}")
    evaluations, skipped = load_instance_evaluations(result_folder, truth, workbench)
    report = build_run_report(
        evaluations, method=method, model=model, skipped_ids=skipped,
        sources={**(sources or {}), "result_folder": str(result_folder.resolve()),
                 "workbench": str(workbench.resolve())},
    )
    return report, evaluations, skipped


def run_command(args: argparse.Namespace) -> None:
    result_folder = args.result_folder.expanduser()
    if not result_folder.is_absolute():
        result_folder = get_repo_root() / result_folder
    result_folder = result_folder.resolve()
    selected = load_selected_instance_ids(args.selected_instances)
    validate_benchmark_ids(selected)
    truth = load_ground_truth_map(args.input, selected)
    report, evaluations, skipped = run_from_paths(
        result_folder, args.workbench_path, truth,
        method=args.method if args.method is not None else result_folder.parent.name,
        model=args.model if args.model is not None else RUN_LABEL_ALIASES.get(
            (args.method or result_folder.parent.name, result_folder.name), result_folder.name),
        sources={"manifest": source_file(args.selected_instances), "truth": source_file(args.input)},
    )
    lines = ["Primary metrics: total ground truth; instance macro averages.",
             "Ranking excludes empty GT; failed outputs score zero.", ""]
    for name, scope in report["scopes"].items():
        lines.extend([f"[{name}]", *summary_lines(scope), ""])
    lines.extend(["Legacy membership diagnostics (micro and added/edited are not the primary metrics):",
                  render_evaluation_text(result_folder, evaluations, skipped)])
    protected = [DATA_ROOT] if PACKAGED else [DATA_ROOT / p for p in (
        "agent_results", "results_archive", "trad_methods_results")]
    archived = any(result_folder.is_relative_to(p.resolve()) for p in protected)
    default_output = (CODE_ROOT / "dataset/evaluation/runs" / report["metadata"]["method"] / result_folder.name
                      if archived else result_folder)
    output_dir = args.output_dir or default_output
    write_report(report, "\n".join(lines), output_dir, "evaluation")


def compare_command(args: argparse.Namespace) -> None:
    baseline_path, candidate_path = args.baseline.expanduser().resolve(), args.candidate.expanduser().resolve()
    report = compare_run_reports(
        load_run_report(baseline_path), load_run_report(candidate_path), tests=args.tests,
        bootstrap_draws=args.bootstrap_draws, permutation_draws=args.permutation_draws,
        bootstrap_seed=args.bootstrap_seed, permutation_seed=args.permutation_seed,
    )
    report["metadata"]["sources"] = {
        "baseline": source_file(baseline_path), "candidate": source_file(candidate_path),
    }
    if args.output_dir is None:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        write_report(report, render_compare_text(report), args.output_dir, "comparison")


def holm_adjust(raw_p: dict[str, float]) -> dict[str, float]:
    """Step-down FWER correction for exactly the declared family."""
    if any(p is None or not math.isfinite(p) or not 0 <= p <= 1 for p in raw_p.values()):
        raise ValueError("Every declared Holm comparison needs a finite p-value")
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted = {}
    running = 0.0
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * raw_p[key]))
        adjusted[key] = running
    return adjusted


def load_retrieval_rankings(root: Path, ids: list[str]) -> tuple[dict, dict]:
    rankings, sources = {}, {}
    for method in (*STATIC_METHODS, "trace_vsm"):
        rankings[method], sources[method] = {}, {}
        for instance_id in ids:
            path = root / method / f"{instance_id}.json.gz"
            # Missing or failed retrieval artifacts are input errors, not
            # successful empty rankings and not agent completion failures.
            raw = path.read_bytes()
            row = json.loads(gzip.decompress(raw), object_pairs_hook=unique_json_object)
            if not isinstance(row, dict) or row.get("instance_id") != instance_id:
                raise ValueError(f"Retrieval instance ID mismatch: {path}")
            if row.get("method") not in (method, RETRIEVAL_LABELS[method]) or "failure_reason" not in row or row["failure_reason"] is not None:
                raise ValueError(f"Wrong method or failed retrieval cache: {path}")
            ranked, error = validate_ranked_payload({"results": row.get("rankings")})
            if ranked is None:
                raise ValueError(f"Invalid retrieval ranking in {path}: {error}")
            rankings[method][instance_id] = ranked
            sources[method][instance_id] = {
                "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                "configuration": row.get("configuration"), "signature": row.get("signature"),
            }
    return rankings, sources


def retrieval_records(rankings: dict, truth: dict, cutoff: int | dict) -> list[dict]:
    records = []
    for instance_id in sorted(truth):
        ranked = {}
        for g in GRANULARITIES:
            k = cutoff if isinstance(cutoff, int) else cutoff[instance_id][g]
            if type(k) is not int or k < 0:
                raise ValueError("Retrieval cutoffs must be nonnegative integers")
            ranked[g] = rankings[instance_id][g][:k]
        item = InstanceEvaluation(
            instance_id, {g: set(ranked[g]) for g in GRANULARITIES},
            truth[instance_id], False, ranked_predictions=ranked,
        )
        records.append(instance_record(item))
    return records


def matched_cutoffs(dalic_records: list[dict]) -> dict:
    return {
        r["instance_id"]: {
            g: len(r["predictions"][g]) if r["valid"] else 3 for g in GRANULARITIES
        }
        for r in dalic_records
    }


def adjust_paper_family(comparisons: list[dict], family: str) -> dict:
    """One Holm family per metric and declared comparison set."""
    adjusted = {}
    for metric in ("completion", "f1", "map"):
        raw = {p["id"]: p["report"]["tests"][metric]["raw_p"] for p in comparisons}
        adjusted[metric] = holm_adjust(raw)
        for pair in comparisons:
            value = adjusted[metric][pair["id"]]
            pair["report"]["tests"][metric].update({
                "holm_p": value, "reject": value <= .05, "family_size": len(comparisons),
                "family": family + ":" + metric,
            })
            pair["report"]["metadata"]["multiple_testing"] = (
                f"{family}: separate {len(comparisons)}-comparison Holm families for completion, F1 and MAP"
            )
    return adjusted


def build_paper_report(
    agents: dict, rankings: dict, truth: dict, *, models: list[str], repeated_runs: dict | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS, permutation_draws: int = PERMUTATION_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED, permutation_seed: int = PERMUTATION_SEED,
) -> dict:
    if not models or len(models) != len(set(models)) or any(m not in PAPER_RUNS for m in models):
        raise ValueError("Paper models must be distinct names from PAPER_RUNS")
    ids = sorted(truth)
    canonical_ids = load_selected_instance_ids(DEFAULT_SELECTED_INSTANCES_PATH)
    if set(ids) != canonical_ids or len(ids) != 226:
        raise ValueError("Paper mode requires the complete validated 226-instance benchmark")
    ranking_counts = {g: sum(bool(truth[i]["total"][g]) for i in ids) for g in GRANULARITIES}
    ranking_counts["avg3"] = sum(all(truth[i]["total"][g] for g in GRANULARITIES) for i in ids)
    if ranking_counts != {"files": 226, "classes": 172, "functions": 217, "avg3": 170}:
        raise ValueError("Ground-truth ranking cohorts differ from the released benchmark")
    for model in models:
        for method in ("baseline", "dalic_vsm", "dalic_no_runtime"):
            run = agents[method][model]
            if run["metadata"]["method"] != method or run["metadata"]["model"] != PAPER_RUNS[model][method]:
                raise ValueError(f"Run metadata differs from the paper condition: {method}/{model}")
            if [r["instance_id"] for r in run["instances"]] != ids:
                raise ValueError(f"Agent cohort differs from the benchmark: {method}/{model}")
            for record in run["instances"]:
                expected = {c: {g: sorted(truth[record["instance_id"]][c][g]) for g in GRANULARITIES}
                            for c in CHANGE_TYPES}
                if record["ground_truth"] != expected:
                    raise ValueError("Agent and retrieval ground truth differ")
    for i in ids:
        for g in GRANULARITIES:
            if not set(rankings["trace_vsm"][i][g]) <= set(rankings["static_vsm"][i][g]):
                raise ValueError(f"Trace candidates are not a subset of Static VSM: {i}/{g}")
    inference = dict(bootstrap_draws=bootstrap_draws, permutation_draws=permutation_draws,
                     bootstrap_seed=bootstrap_seed, permutation_seed=permutation_seed)
    def scope(left, right, key):
        return compare_scope(left, right, key=key, draws=bootstrap_draws, seed_base=bootstrap_seed)

    fixed = {method: {k: retrieval_records(rankings[method], truth, k) for k in PAPER_CUTOFFS}
             for method in (*STATIC_METHODS, "trace_vsm")}
    report = {
        "schema_version": SCHEMA_VERSION, "kind": "paper",
        "metadata": {
            "scoring_version": SCORING_VERSION, "policy": scoring_policy(), "models": models,
            "instance_ids": ids, "instances": len(ids), "ranking_instances_by_granularity": ranking_counts,
            **inference_metadata(bootstrap_draws, bootstrap_seed, permutation_draws, permutation_seed),
            "units": "proportions; multiply by 100 for percentages or differences in percentage points",
            "cutoffs": list(PAPER_CUTOFFS), "primary_fixed_cutoff": 3,
            "matched": "valid REAL-FL length per instance/granularity; invalid REAL-FL uses IR k=3",
            "short_ranking": "available prefix, no padding; precision uses actual returned length",
            "agent_output": "complete returned ranking, never truncated to the IR cutoff",
            "retrieval_inference": "paired estimates and pointwise CIs; no hypothesis tests",
            "common_success": "pairwise valid intersection, with CIs and no hypothesis tests",
            "run_selection": {model: PAPER_RUNS[model] for model in models},
            "historical_run_labels": {method + "/" + name: label
                                      for (method, name), label in RUN_LABEL_ALIASES.items()},
            "seed_policy": "published VSM analysis keys and historical run labels, independent of filesystem location",
            "display_policy": "paper_estimate_pp uses rounded endpoint scores for Static vs Trace VSM, as in Table 2; estimate and CIs retain raw calculations",
            "scope": "main evaluation and repeated-run inference; excludes trajectory-access audit and token usage",
        },
        "agent_runs": agents,
        "retrieval_fixed": {method: {str(k): summarize_records(records) for k, records in cutoffs.items()}
                            for method, cutoffs in fixed.items()},
        "static_trace": {}, "trace_realfl": {}, "matched_static_ir": {},
        "baseline": [], "ablation": [], "repeated_runs": [],
    }
    for k in PAPER_CUTOFFS:
        report["static_trace"][str(k)] = scope(
            fixed["static_vsm"][k], fixed["trace_vsm"][k], f"static_vsm_trace_vsm:@{k}")
    for number, model in enumerate(models, 1):
        print(f"Paper comparisons: {model}", file=sys.stderr, flush=True)
        candidate = agents["dalic_vsm"][model]
        records = candidate["instances"]
        name = PAPER_RUNS[model]["dalic_vsm"]
        lengths = matched_cutoffs(records)
        report["trace_realfl"][model] = {}
        for cutoff, reference in (("top3", fixed["trace_vsm"][3]),
                                  ("matched_length", retrieval_records(rankings["trace_vsm"], truth, lengths))):
            key = "rq1_vsm_supplement:" + digest_json(["trace_vsm", cutoff, "dalic_vsm", name]) + ":all"
            report["trace_realfl"][model][cutoff] = scope(reference, records, key)
        report["matched_static_ir"][model] = {
            method: scope(retrieval_records(rankings[method], truth, lengths), records,
                          f"rq2:{name}:{method}:matched") for method in STATIC_METHODS}
        for method, section, prefix in (("baseline", "baseline", "M"),
                                         ("dalic_no_runtime", "ablation", "NRM")):
            report[section].append({
                "id": f"{prefix}{number:02d}", "model": model, "reference_method": method,
                "reference_run": PAPER_RUNS[model][method], "candidate_run": name,
                "report": compare_run_reports(agents[method][model], candidate, tests="paired", **inference),
            })
    for model in models:
        for method in ("dalic_vsm", "baseline"):
            for run in (repeated_runs or {}).get(model, {}).get(method, []):
                baseline = run if method == "baseline" else agents["baseline"][model]
                candidate = run if method == "dalic_vsm" else agents["dalic_vsm"][model]
                report["repeated_runs"].append({
                    "id": f"S{len(report['repeated_runs']) + 1:02d}", "model": model,
                    "replacement": method, "reference_run": baseline["metadata"]["model"],
                    "candidate_run": candidate["metadata"]["model"],
                    "report": compare_run_reports(baseline, candidate, tests="paired", **inference),
                })
    report["holm"] = {section: adjust_paper_family(report[section], section)
                      for section in ("baseline", "ablation", "repeated_runs")}
    return report


def paper_csv_rows(report: dict) -> tuple[list[dict], list[dict]]:
    """Export one long-form metric table and one difference/test table."""
    metrics, effects = [], []
    def add_summary(summary, **labels):
        for g in FEATURES:
            values = summary["metrics"][g]
            metrics.append({**labels, "granularity": g, "n": values["n"],
                            "ranking_n": values["ranking_n"], "valid_n": summary["valid_n"],
                            "valid_rate": summary["valid_rate"],
                            **{m: values[m] for m in ("precision", "recall", "f1", "map")},
                            **{m + "_percent": f"{100*values[m]:.2f}" if values[m] is not None else ""
                               for m in ("precision", "recall", "f1", "map")}})

    def add_effect(effect, metric, granularity, labels, test=None, paper_display=None):
        test = test or {}
        ci = effect.get("ci_95") or [None, None]
        effects.append({**labels, "granularity": granularity, "metric": metric,
                        "n": effect["n"], "estimate": effect["estimate"],
                        "paper_estimate_pp": (paper_display if paper_display is not None else
                                              f"{100*effect['estimate']:+.2f}" if effect["estimate"] is not None else ""),
                        "ci_low": ci[0], "ci_high": ci[1], "bootstrap_seed": effect.get("seed"),
                        "raw_p": test.get("raw_p"), "holm_p": test.get("holm_p"),
                        "family_size": test.get("family_size"), "family": test.get("family"),
                        "reject": test.get("reject"), "test_seed": test.get("seed")})

    def add_scope(comparison, labels, tests=None):
        for side in ("baseline", "candidate"):
            add_summary(comparison[side], **labels, side=side)
        for metric in ("precision", "recall", "f1", "map"):
            for g in FEATURES:
                test = (tests or {}).get(metric) if g == "avg3" else None
                display = None
                if labels["section"] == "static_trace":
                    left = float(f"{100*comparison['baseline']['metrics'][g][metric]:.2f}")
                    right = float(f"{100*comparison['candidate']['metrics'][g][metric]:.2f}")
                    display = f"{right-left:+.2f}"
                add_effect(comparison["effects"][metric][g], metric, g, labels, test, display)

    for method, runs in report["agent_runs"].items():
        for model, run in runs.items():
            for name, summary in run["scopes"].items():
                add_summary(summary, section="agent_runs", model=model, comparator=method, condition="main", scope=name)
    for method, cutoffs in report["retrieval_fixed"].items():
        for cutoff, summary in cutoffs.items():
            add_summary(summary, section="retrieval_fixed", model="", comparator=method, condition=cutoff, scope="all_instances")
    for cutoff, comparison in report["static_trace"].items():
        add_scope(comparison, dict(section="static_trace", model="", comparator="static_vsm", condition=cutoff, scope="all_instances"))
    for section in ("trace_realfl", "matched_static_ir"):
        for model, comparisons in report[section].items():
            for condition, comparison in comparisons.items():
                add_scope(comparison, dict(section=section, model=model,
                          comparator="trace_vsm" if section == "trace_realfl" else condition,
                          condition=condition, scope="all_instances"))
    for section in ("baseline", "ablation", "repeated_runs"):
        for pair in report[section]:
            comparison = pair["report"]
            labels = dict(section=section, model=pair["model"], comparator=pair.get("reference_method", "baseline"),
                          condition=pair["id"], scope="all_instances")
            add_effect(comparison["completion"]["effect"], "completion", "run", labels, comparison["tests"]["completion"])
            for name, comparison_scope in comparison["scopes"].items():
                add_scope(comparison_scope, {**labels, "scope": name},
                          comparison["tests"] if name == "all_instances" else None)
    return metrics, effects


def render_paper_text(report: dict) -> str:
    meta = report["metadata"]
    lines = ["REAL-FL evaluation from released raw results", "",
             f"Instances: {meta['instances']}; Avg3 MAP: {meta['ranking_instances_by_granularity']['avg3']}",
             "Scores are percentages; differences are percentage points.",
             "Confidence intervals are pointwise. Common Success has no hypothesis tests.",
             "Holm families are separate for each metric and comparison set.", "",
             "REAL-FL main results: valid %, Avg3 F1 %, Avg3 MAP %"]
    for model in meta["models"]:
        s = report["agent_runs"]["dalic_vsm"][model]["scopes"]["all_instances"]
        lines.append(f"{model}: {100*s['valid_rate']:.2f}, {100*s['metrics']['avg3']['f1']:.2f}, {100*s['metrics']['avg3']['map']:.2f}")
    for section in ("baseline", "ablation", "repeated_runs"):
        lines.extend(["", section])
        for pair in report[section]:
            comparison = pair["report"]
            cells = [pair["id"], pair["model"]]
            for metric in ("completion", "f1", "map"):
                effect = (comparison["completion"]["effect"] if metric == "completion"
                          else comparison["scopes"]["all_instances"]["effects"][metric]["avg3"])
                cells.append(f"{metric}: {effect_text(effect)}; Holm p={comparison['tests'][metric]['holm_p']:.6g}")
            lines.append(" | ".join(cells))
    lines.extend(["", "All granularities, retrieval comparisons and Common Success: paper_metrics.csv and paper_differences.csv.",
                  "Full provenance, instance records and seeds: paper_evaluation.json."])
    return "\n".join(lines) + "\n"


def paper_command(args: argparse.Namespace) -> None:
    if not args.models or len(args.models) != len(set(args.models)):
        raise ValueError("Paper model list must be nonempty without duplicates")
    for name in ("input", "selected_instances", "agent_results_root", "rankings_root", "workbench_path", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if (PACKAGED and args.output_dir.is_relative_to(DATA_ROOT.resolve())) or args.output_dir.is_relative_to(args.agent_results_root) or args.output_dir.is_relative_to(args.rankings_root):
        raise ValueError("Choose an output directory outside the input archives")
    selected = load_selected_instance_ids(args.selected_instances)
    validate_benchmark_ids(selected)
    if selected != load_selected_instance_ids(DEFAULT_SELECTED_INSTANCES_PATH):
        raise ValueError("Paper mode requires all 226 instances; use run for a benchmark subset")
    truth = load_ground_truth_map(args.input, selected)
    sources = {"manifest": source_file(args.selected_instances), "truth": source_file(args.input),
               "evaluator": source_file(Path(__file__))}
    agents = {method: {} for method in ("baseline", "dalic_vsm", "dalic_no_runtime")}
    runs, repeated = {}, {}

    def read_run(method, directory):
        if (method, directory) not in runs:
            label = RUN_LABEL_ALIASES.get((method, directory), directory)
            folder = args.agent_results_root / method / directory
            if not folder.is_dir() and label != directory:
                folder = args.agent_results_root / method / label
            run, _, _ = run_from_paths(folder,
                                      args.workbench_path, truth, method=method, model=label, sources=sources)
            runs[method, directory] = run
            print(f"Scored {method}/{directory}: {run['scopes']['all_instances']['valid_n']}/226 valid", file=sys.stderr, flush=True)
        return runs[method, directory]

    for model in args.models:
        config = PAPER_RUNS[model]
        for method in agents:
            agents[method][model] = read_run(method, config[method])
        repeated[model] = {method: [read_run(method, name) for name in names]
                           for method, names in config.get("repeats", {}).items()}
    rankings, ranking_sources = load_retrieval_rankings(args.rankings_root, sorted(selected))
    report = build_paper_report(agents, rankings, truth, models=args.models, repeated_runs=repeated,
        bootstrap_draws=args.bootstrap_draws, permutation_draws=args.permutation_draws,
        bootstrap_seed=args.bootstrap_seed, permutation_seed=args.permutation_seed)
    report["metadata"]["sources"] = {**sources, "retrieval": ranking_sources}
    # Fail if any input changed while scores and statistics were computed.
    for (method, directory), run in runs.items():
        for record in run["instances"]:
            path = Path(run["metadata"]["sources"]["result_folder"]) / record["instance_id"] / "results.txt"
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if actual != record["result_sha256"]:
                raise ValueError(f"Raw input changed during evaluation: {path}")
    for source in [*sources.values(), *(s for method in ranking_sources.values() for s in method.values())]:
        if source_file(Path(source["path"]))["sha256"] != source["sha256"]:
            raise ValueError(f"Source changed during evaluation: {source['path']}")
    write_report(report, render_paper_text(report), args.output_dir, "paper_evaluation")
    for (method, directory), run in runs.items():
        write_report(run, "\n".join(summary_lines(run["scopes"]["all_instances"])),
                     args.output_dir / "runs" / method / directory, "evaluation")
    for name, rows in zip(("paper_metrics.csv", "paper_differences.csv"), paper_csv_rows(report)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved evaluation: {args.output_dir}")


def main() -> None:
    try:
        args = parse_args()
        if args.mode == "run":
            run_command(args)
        elif args.mode == "compare":
            compare_command(args)
        else:
            paper_command(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(f"Evaluation error: {exc}") from exc


if __name__ == "__main__":
    main()
