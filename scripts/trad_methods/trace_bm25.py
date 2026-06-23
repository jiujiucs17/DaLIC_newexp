#!/usr/bin/env python3
"""Evaluate DALIC dynamic evidence candidates against SWE-bench ground truth.
   needs to be executed from the root of the repository, e.g. `python scripts/trad_methods_results/trace_bm25.py`"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHANGE_TYPES = ("added", "edited")
GRANULARITIES = ("files", "classes", "functions")
EVAL_GRANULARITIES = (*GRANULARITIES, "all")
CUTOFFS: tuple[int | None, ...] = (3, 5, 10, 20, None)
EVIDENCE_FILES = {
    "files": "file.json",
    "classes": "class.json",
    "functions": "function.json",
}

EntityMap = dict[str, list[str]]
TruthMap = dict[str, set[str]]


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int
    candidate_count: int
    ground_truth_count: int
    precision: float
    recall: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("dataset/swebench_feature_with_truth.jsonl"),
        help="Ground-truth JSONL file.",
    )
    parser.add_argument(
        "--workbench",
        type=Path,
        default=Path("dataset/agent_workbench/dalic"),
        help="DALIC workbench containing per-instance dynamic_evidence folders.",
    )
    parser.add_argument(
        "--instances",
        "--selected-instances",
        dest="instances",
        type=Path,
        default=None,
        help=(
            "Optional TOML file containing instance_ids. By default all "
            "instances under --workbench are evaluated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/trad_methods_results"),
        help="Directory where summary.md is written.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        help="Evaluate only this instance; may be supplied more than once.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> None:
    args.truth = args.truth.expanduser()
    args.workbench = args.workbench.expanduser()
    args.output_dir = args.output_dir.expanduser()
    if args.instances is not None:
        args.instances = args.instances.expanduser()


def empty_entity_map() -> EntityMap:
    return {granularity: [] for granularity in GRANULARITIES}


def empty_truth_map() -> TruthMap:
    return {granularity: set() for granularity in GRANULARITIES}


def cutoff_label(cutoff: int | None) -> str:
    return "@Full" if cutoff is None else f"@{cutoff}"


def cutoff_labels() -> list[str]:
    return [cutoff_label(cutoff) for cutoff in CUTOFFS]


def load_instance_ids_from_toml(path: Path) -> list[str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    instance_ids = data.get("instance_ids")
    if not isinstance(instance_ids, list) or not all(
        isinstance(value, str) for value in instance_ids
    ):
        raise ValueError(f"{path} must contain a string array named instance_ids")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"{path} contains duplicate instance ids")
    return instance_ids


def load_instance_ids_from_workbench(workbench: Path) -> list[str]:
    if not workbench.is_dir():
        raise ValueError(f"Workbench directory does not exist: {workbench}")

    instance_ids = [
        path.name
        for path in sorted(workbench.iterdir())
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / "dynamic_evidence").is_dir()
    ]
    if not instance_ids:
        raise ValueError(f"No dynamic_evidence directories found under {workbench}")
    return instance_ids


def selected_instance_ids(args: argparse.Namespace) -> list[str]:
    if args.instances is None:
        instance_ids = load_instance_ids_from_workbench(args.workbench)
    else:
        instance_ids = load_instance_ids_from_toml(args.instances)

    if args.instance_ids:
        requested = set(args.instance_ids)
        unknown = requested - set(instance_ids)
        if unknown:
            raise ValueError(f"Unknown instance ids: {', '.join(sorted(unknown))}")
        instance_ids = [value for value in instance_ids if value in requested]
    return instance_ids


def load_truth_rows(path: Path, instance_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            if instance_id not in instance_ids:
                continue
            if instance_id in rows:
                raise ValueError(f"Duplicate {instance_id} at {path}:{line_number}")
            if not isinstance(row.get("ground_truth"), dict):
                raise ValueError(f"Invalid ground_truth for {instance_id}")
            rows[str(instance_id)] = row

    missing = instance_ids - rows.keys()
    if missing:
        raise ValueError(f"Ground truth missing for: {', '.join(sorted(missing))}")
    return rows


def truth_entities(row: dict[str, Any]) -> TruthMap:
    ground_truth = row["ground_truth"]
    truth = empty_truth_map()
    for change_type in CHANGE_TYPES:
        change_payload = ground_truth.get(change_type, {})
        if not isinstance(change_payload, dict):
            continue
        for granularity in GRANULARITIES:
            values = change_payload.get(granularity, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) for value in values
            ):
                instance_id = row.get("instance_id")
                raise ValueError(
                    f"Invalid ground_truth.{change_type}.{granularity} "
                    f"for {instance_id}"
                )
            truth[granularity].update(values)
    return truth


def load_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, list) or any(
        not isinstance(item, dict) for item in payload
    ):
        raise ValueError(f"{path} must contain a JSON array of objects")
    return payload


def load_candidates(instance_dir: Path) -> EntityMap:
    evidence_dir = instance_dir / "dynamic_evidence"
    if not evidence_dir.is_dir():
        raise FileNotFoundError(f"Missing dynamic_evidence directory: {evidence_dir}")

    candidates = empty_entity_map()
    for granularity, filename in EVIDENCE_FILES.items():
        evidence_path = evidence_dir / filename
        if not evidence_path.is_file():
            raise FileNotFoundError(f"Missing evidence file: {evidence_path}")

        seen: set[str] = set()
        for rank, record in enumerate(load_json_array(evidence_path), 1):
            entity_id = record.get("qualified_name")
            if not isinstance(entity_id, str) or not entity_id:
                raise ValueError(
                    f"{evidence_path} record #{rank} has invalid qualified_name"
                )
            if entity_id not in seen:
                candidates[granularity].append(entity_id)
                seen.add(entity_id)
    return candidates


def take_top(values: list[str], cutoff: int | None) -> list[str]:
    return values if cutoff is None else values[:cutoff]


def candidates_at(candidates: EntityMap, granularity: str, cutoff: int | None) -> set[str]:
    if granularity == "all":
        return {
            f"{kind}\0{entity}"
            for kind in GRANULARITIES
            for entity in take_top(candidates[kind], cutoff)
        }
    return set(take_top(candidates[granularity], cutoff))


def truth_at(truth: TruthMap, granularity: str) -> set[str]:
    if granularity == "all":
        return {
            f"{kind}\0{entity}"
            for kind in GRANULARITIES
            for entity in truth[kind]
        }
    return truth[granularity]


def score(candidate_ids: set[str], truth_ids: set[str]) -> Counts:
    tp = len(candidate_ids & truth_ids)
    fp = len(candidate_ids - truth_ids)
    fn = len(truth_ids - candidate_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Counts(
        tp=tp,
        fp=fp,
        fn=fn,
        candidate_count=len(candidate_ids),
        ground_truth_count=len(truth_ids),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def score_from_totals(tp: int, fp: int, fn: int) -> Counts:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Counts(
        tp=tp,
        fp=fp,
        fn=fn,
        candidate_count=tp + fp,
        ground_truth_count=tp + fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def aggregate_metrics(
    instance_ids: list[str],
    truth_by_instance: dict[str, TruthMap],
    candidates_by_instance: dict[str, EntityMap],
) -> dict[str, dict[str, dict[str, Any]]]:
    aggregate: dict[str, dict[str, dict[str, Any]]] = {}

    for cutoff in CUTOFFS:
        cutoff_key = cutoff_label(cutoff)
        aggregate[cutoff_key] = {}
        for granularity in EVAL_GRANULARITIES:
            rows = [
                score(
                    candidates_at(candidates_by_instance[instance_id], granularity, cutoff),
                    truth_at(truth_by_instance[instance_id], granularity),
                )
                for instance_id in instance_ids
            ]
            micro = score_from_totals(
                sum(row.tp for row in rows),
                sum(row.fp for row in rows),
                sum(row.fn for row in rows),
            )
            rows_with_truth = [row for row in rows if row.ground_truth_count > 0]
            if rows_with_truth:
                macro = {
                    metric: sum(getattr(row, metric) for row in rows_with_truth)
                    / len(rows_with_truth)
                    for metric in ("precision", "recall", "f1")
                }
            else:
                macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

            aggregate[cutoff_key][granularity] = {
                "micro": micro,
                "macro": macro,
                "instances_with_truth": len(rows_with_truth),
            }
    return aggregate


def render_markdown(
    *,
    truth_path: Path,
    workbench: Path,
    instance_count: int,
    aggregate: dict[str, dict[str, dict[str, Any]]],
) -> str:
    lines = [
        "# DALIC Dynamic Evidence Ground-Truth Metrics",
        "",
        f"- Workbench: `{workbench}`",
        f"- Ground truth: `{truth_path}`",
        f"- Evaluated instances: {instance_count}",
        "",
        "## Micro Metrics",
        "",
        "| Cutoff | Granularity | Candidates | Truth | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cutoff in cutoff_labels():
        for granularity in EVAL_GRANULARITIES:
            metric = aggregate[cutoff][granularity]["micro"]
            lines.append(
                f"| {cutoff} | {granularity} | {metric.candidate_count} | "
                f"{metric.ground_truth_count} | {metric.tp} | {metric.fp} | "
                f"{metric.fn} | {metric.precision:.4%} | "
                f"{metric.recall:.4%} | {metric.f1:.4%} |"
            )

    lines.extend(
        [
            "",
            "## Macro Metrics",
            "",
            "Macro averages include only instances with ground truth at that granularity.",
            "",
            "| Cutoff | Granularity | Instances | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for cutoff in cutoff_labels():
        for granularity in EVAL_GRANULARITIES:
            result = aggregate[cutoff][granularity]
            macro = result["macro"]
            lines.append(
                f"| {cutoff} | {granularity} | {result['instances_with_truth']} | "
                f"{macro['precision']:.4%} | {macro['recall']:.4%} | "
                f"{macro['f1']:.4%} |"
            )

    lines.extend(
        [
            "",
            "## Matching",
            "",
            "- `@k` is applied independently to `file.json`, `class.json`, and `function.json` rankings.",
            "- `all` combines tagged file/class/function candidates at the same cutoff.",
            "- Ground truth is the union of `added` and `edited` entities.",
            "- Files match by repository-relative path; classes/functions match by exact `<file_path>::<qualified_name>`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    normalize_args(args)
    instance_ids = selected_instance_ids(args)
    truth_rows = load_truth_rows(args.truth, set(instance_ids))

    truth_by_instance: dict[str, TruthMap] = {}
    candidates_by_instance: dict[str, EntityMap] = {}
    for index, instance_id in enumerate(instance_ids, 1):
        truth_by_instance[instance_id] = truth_entities(truth_rows[instance_id])
        candidates_by_instance[instance_id] = load_candidates(
            args.workbench / instance_id
        )
        full_score = score(
            candidates_at(candidates_by_instance[instance_id], "all", None),
            truth_at(truth_by_instance[instance_id], "all"),
        )
        print(
            f"[{index:03d}/{len(instance_ids):03d}] {instance_id}: "
            f"P/R/F1@Full={full_score.precision:.2%}/"
            f"{full_score.recall:.2%}/{full_score.f1:.2%}",
            flush=True,
        )

    aggregate = aggregate_metrics(instance_ids, truth_by_instance, candidates_by_instance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "trace_bm25.md"
    summary_path.write_text(
        render_markdown(
            truth_path=args.truth,
            workbench=args.workbench,
            instance_count=len(instance_ids),
            aggregate=aggregate,
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
