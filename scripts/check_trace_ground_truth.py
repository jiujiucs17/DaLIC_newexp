#!/usr/bin/env python3
"""Evaluate VarTracer trace and dependency candidates against ground truth.

For each selected instance this script independently extracts project-local
file/function candidates from exec_stack.json and file/class/function
candidates from dep_tree.json. It evaluates trace, dependency, and their union
with micro and macro precision, recall, and F1.

VarTracer records only an unqualified function name in both artifacts and does
not record a class in execution-stack events. Named entities are therefore
evaluated as path::leaf_name. Trace class candidates are intentionally empty
rather than being inferred from dependency data.
"""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CHANGE_TYPES = ("added", "edited")
GRANULARITIES = ("files", "classes", "functions")
EVAL_GRANULARITIES = (*GRANULARITIES, "all")
SOURCES = ("trace", "dependency", "union")
CALL_MARKER = b'"type": "CALL"'
FILE_FIELD = b'"file_path"'
FUNC_FIELD = b'"func"'

CandidateSets = dict[str, set[str]]
EvidenceMap = dict[tuple[str, str], dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path("dataset/trace_completed_instance.toml"),
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("dataset/swebench_feature_with_truth.jsonl"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("dataset/vartracer_results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/trace_ground_truth_check"),
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        help="Evaluate only this instance; may be supplied more than once.",
    )
    return parser.parse_args()


def empty_candidates() -> CandidateSets:
    return {granularity: set() for granularity in GRANULARITIES}


def project_relative_path(observed: Any) -> str | None:
    if not isinstance(observed, str):
        return None
    normalized = observed.replace("\\", "/")
    marker = "/testbed/"
    if marker not in normalized:
        return None
    relative = normalized.split(marker, 1)[1].lstrip("/")
    if not relative or relative.startswith("<"):
        return None
    return relative


def leaf_name(name: Any) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    if name in {"<module>", "<external>"}:
        return None
    return name.rsplit(".", 1)[-1]


def named_entity_id(source_path: str, name: Any) -> str | None:
    leaf = leaf_name(name)
    return f"{source_path}::{leaf}" if leaf else None


def load_instance_ids(path: Path) -> list[str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    instance_ids = data.get("instance_ids")
    if not isinstance(instance_ids, list) or not all(
        isinstance(value, str) for value in instance_ids
    ):
        raise ValueError(f"{path} must contain a string array named instance_ids")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"{path} contains duplicate instance ids")
    return instance_ids


def load_truth(path: Path, selected: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            if instance_id in selected:
                if instance_id in rows:
                    raise ValueError(f"Duplicate {instance_id} at {path}:{line_number}")
                rows[instance_id] = row
    missing = selected - rows.keys()
    if missing:
        raise ValueError(f"Ground truth missing for: {', '.join(sorted(missing))}")
    return rows


def make_records(instance_id: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError(f"Invalid ground_truth for {instance_id}")
    records: list[dict[str, Any]] = []
    for change_type in CHANGE_TYPES:
        payload = ground_truth.get(change_type, {})
        for granularity in GRANULARITIES:
            entities = payload.get(granularity, [])
            if not isinstance(entities, list):
                raise ValueError(
                    f"Invalid ground_truth.{change_type}.{granularity} for {instance_id}"
                )
            for entity_id in entities:
                source_path = entity_id
                qualified_name = None
                normalized_id = entity_id
                if granularity != "files":
                    if "::" not in entity_id:
                        raise ValueError(f"Invalid {granularity} id: {entity_id}")
                    source_path, qualified_name = entity_id.split("::", 1)
                    normalized_id = named_entity_id(source_path, qualified_name)
                    if normalized_id is None:
                        raise ValueError(f"Invalid named entity id: {entity_id}")
                records.append(
                    {
                        "instance_id": instance_id,
                        "change_type": change_type,
                        "granularity": granularity,
                        "ground_truth_id": entity_id,
                        "normalized_id": normalized_id,
                        "source_path": source_path,
                        "qualified_name": qualified_name,
                        "leaf_name": leaf_name(qualified_name),
                        "trace_found": False,
                        "trace_check": (
                            "not_representable" if granularity == "classes" else "pending"
                        ),
                        "trace_evidence": None,
                        "dependency_found": False,
                        "dependency_evidence": None,
                    }
                )
    return records


def truth_sets(records: list[dict[str, Any]]) -> CandidateSets:
    result = empty_candidates()
    for record in records:
        result[record["granularity"]].add(record["normalized_id"])
    return result


def add_candidate(
    candidates: CandidateSets,
    evidence: EvidenceMap,
    granularity: str,
    entity_id: str | None,
    detail: dict[str, Any],
) -> None:
    if entity_id is None:
        return
    candidates[granularity].add(entity_id)
    evidence.setdefault((granularity, entity_id), detail)


def extract_dependency_candidates(
    instance_dir: Path,
) -> tuple[CandidateSets, EvidenceMap, int]:
    candidates = empty_candidates()
    evidence: EvidenceMap = {}
    loaded = 0
    for dep_path in sorted(instance_dir.glob("tc_*/dep_tree.json")):
        with dep_path.open(encoding="utf-8") as handle:
            graph = json.load(handle)
        loaded += 1
        test_case_id = dep_path.parent.name
        files = graph.get("files", {})
        relative_files: dict[str, str] = {}
        for file_id, observed_path in files.items():
            relative = project_relative_path(observed_path)
            if relative is None:
                continue
            relative_files[file_id] = relative
            add_candidate(
                candidates,
                evidence,
                "files",
                relative,
                {
                    "test_case_id": test_case_id,
                    "dep_tree": str(dep_path),
                    "match": "file_table",
                    "file_id": file_id,
                    "observed_path": observed_path,
                },
            )

        for symbol_id, symbol in graph.get("symbols", {}).items():
            if not isinstance(symbol, list) or len(symbol) < 5:
                continue
            kind, name, scope, file_id, line = symbol[:5]
            relative = relative_files.get(file_id)
            if relative is None:
                continue
            common = {
                "test_case_id": test_case_id,
                "dep_tree": str(dep_path),
                "symbol_id": symbol_id,
                "symbol": symbol,
                "source_path": relative,
                "line": line,
            }
            if kind == "class":
                add_candidate(
                    candidates,
                    evidence,
                    "classes",
                    named_entity_id(relative, name),
                    {**common, "match": "class_symbol"},
                )
            if kind == "func":
                add_candidate(
                    candidates,
                    evidence,
                    "functions",
                    named_entity_id(relative, name),
                    {**common, "match": "function_symbol"},
                )
            add_candidate(
                candidates,
                evidence,
                "functions",
                named_entity_id(relative, scope),
                {**common, "match": "function_scope"},
            )
    return candidates, evidence, loaded


def read_json_string_field(
    mapped: mmap.mmap,
    marker: bytes,
    start: int,
    end: int,
) -> str | None:
    field = mapped.find(marker, start, end)
    if field < 0:
        return None
    colon = mapped.find(b":", field + len(marker), end)
    if colon < 0:
        return None
    quote_start = mapped.find(b'"', colon + 1, end)
    if quote_start < 0:
        return None
    position = quote_start + 1
    while position < end:
        byte = mapped[position]
        if byte == 92:
            position += 2
            continue
        if byte == 34:
            try:
                raw = mapped[quote_start : position + 1].decode("utf-8")
                return json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
        position += 1
    return None


def add_trace_event(
    candidates: CandidateSets,
    evidence: EvidenceMap,
    trace_path: Path,
    offset: int,
    observed_path: str | None,
    function_name: str | None,
) -> None:
    relative = project_relative_path(observed_path)
    if relative is None:
        return
    detail = {
        "test_case_id": trace_path.parent.name,
        "exec_stack": str(trace_path),
        "byte_offset": offset,
        "source_path": relative,
    }
    add_candidate(
        candidates,
        evidence,
        "files",
        relative,
        {**detail, "match": "file_path"},
    )
    add_candidate(
        candidates,
        evidence,
        "functions",
        named_entity_id(relative, function_name),
        {
            **detail,
            "match": "file_path_and_unqualified_func",
            "function_name": function_name,
        },
    )


def trace_paths(instance_dir: Path) -> list[Path]:
    paths = {
        *instance_dir.glob("tc_*/exec_stack.json"),
        *instance_dir.glob("tc_*/exec_trace.json"),
    }
    return sorted(paths)


def extract_trace_candidates(
    instance_dir: Path,
) -> tuple[CandidateSets, EvidenceMap, int, int]:
    candidates = empty_candidates()
    evidence: EvidenceMap = {}
    scanned = 0
    bytes_scanned = 0
    for trace_path in trace_paths(instance_dir):
        size = trace_path.stat().st_size
        if size == 0:
            continue
        scanned += 1
        bytes_scanned += size
        with trace_path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                # Tracing can begin inside the root test function, so capture
                # its first event even when no CALL event exists for that frame.
                first_file = mapped.find(FILE_FIELD)
                if first_file >= 0:
                    end = min(size, first_file + 4096)
                    add_trace_event(
                        candidates,
                        evidence,
                        trace_path,
                        first_file,
                        read_json_string_field(
                            mapped, FILE_FIELD, first_file, end
                        ),
                        read_json_string_field(mapped, FUNC_FIELD, first_file, end),
                    )

                position = 0
                while True:
                    call_at = mapped.find(CALL_MARKER, position)
                    if call_at < 0:
                        break
                    end = min(size, call_at + 4096)
                    add_trace_event(
                        candidates,
                        evidence,
                        trace_path,
                        call_at,
                        read_json_string_field(mapped, FILE_FIELD, call_at, end),
                        read_json_string_field(mapped, FUNC_FIELD, call_at, end),
                    )
                    position = call_at + len(CALL_MARKER)
    return candidates, evidence, scanned, bytes_scanned


def apply_candidates_to_records(
    records: list[dict[str, Any]],
    trace: CandidateSets,
    trace_evidence: EvidenceMap,
    dependency: CandidateSets,
    dependency_evidence: EvidenceMap,
) -> None:
    for record in records:
        granularity = record["granularity"]
        entity_id = record["normalized_id"]
        record["dependency_found"] = entity_id in dependency[granularity]
        record["dependency_evidence"] = dependency_evidence.get(
            (granularity, entity_id)
        )
        if granularity == "classes":
            record["trace_found"] = False
            record["trace_check"] = "not_representable"
        else:
            record["trace_found"] = entity_id in trace[granularity]
            record["trace_check"] = "searched_full"
        record["trace_evidence"] = trace_evidence.get((granularity, entity_id))
        record["found"] = bool(record["trace_found"] or record["dependency_found"])
        if record["trace_found"] and record["dependency_found"]:
            record["evidence_source"] = "both"
        elif record["trace_found"]:
            record["evidence_source"] = "trace"
        elif record["dependency_found"]:
            record["evidence_source"] = "dependency"
        else:
            record["evidence_source"] = None


def union_candidates(
    trace: CandidateSets, dependency: CandidateSets
) -> CandidateSets:
    return {
        granularity: trace[granularity] | dependency[granularity]
        for granularity in GRANULARITIES
    }


def tagged_all(candidates: CandidateSets) -> set[str]:
    return {
        f"{granularity}\\0{entity_id}"
        for granularity in GRANULARITIES
        for entity_id in candidates[granularity]
    }


def candidates_at(candidates: CandidateSets, granularity: str) -> set[str]:
    return tagged_all(candidates) if granularity == "all" else candidates[granularity]


def score(candidate_ids: set[str], truth_ids: set[str]) -> dict[str, Any]:
    true_positive = len(candidate_ids & truth_ids)
    false_positive = len(candidate_ids - truth_ids)
    false_negative = len(truth_ids - candidate_ids)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "candidate_count": len(candidate_ids),
        "ground_truth_count": len(truth_ids),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_from_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "candidate_count": tp + fp,
        "ground_truth_count": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_metric_rows(
    instance_ids: list[str],
    truth_by_instance: dict[str, CandidateSets],
    candidates_by_instance: dict[str, dict[str, CandidateSets]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        truth = truth_by_instance[instance_id]
        for source in SOURCES:
            candidates = candidates_by_instance[instance_id][source]
            for granularity in EVAL_GRANULARITIES:
                rows.append(
                    {
                        "instance_id": instance_id,
                        "source": source,
                        "granularity": granularity,
                        **score(
                            candidates_at(candidates, granularity),
                            candidates_at(truth, granularity),
                        ),
                    }
                )

    aggregate: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in SOURCES:
        for granularity in EVAL_GRANULARITIES:
            subset = [
                row
                for row in rows
                if row["source"] == source
                and row["granularity"] == granularity
            ]
            tp = sum(row["tp"] for row in subset)
            fp = sum(row["fp"] for row in subset)
            fn = sum(row["fn"] for row in subset)
            with_truth = [row for row in subset if row["ground_truth_count"] > 0]
            macro = {
                metric: (
                    sum(row[metric] for row in with_truth) / len(with_truth)
                    if with_truth
                    else 0.0
                )
                for metric in ("precision", "recall", "f1")
            }
            aggregate[source][granularity] = {
                "micro": score_from_counts(tp, fp, fn),
                "macro_over_instances_with_truth": macro,
                "instances_with_truth": len(with_truth),
            }
    return rows, dict(aggregate)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def write_candidate_sets(
    path: Path,
    instance_ids: list[str],
    truth_by_instance: dict[str, CandidateSets],
    candidates_by_instance: dict[str, dict[str, CandidateSets]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for instance_id in instance_ids:
            payload: dict[str, Any] = {
                "instance_id": instance_id,
                "ground_truth": {
                    key: sorted(value)
                    for key, value in truth_by_instance[instance_id].items()
                },
            }
            for source in SOURCES:
                payload[source] = {
                    key: sorted(value)
                    for key, value in candidates_by_instance[instance_id][
                        source
                    ].items()
                }
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def write_metrics_csv(
    path: Path,
    rows: list[dict[str, Any]],
    aggregate: dict[str, dict[str, dict[str, Any]]],
) -> None:
    fields = (
        "instance_id",
        "source",
        "granularity",
        "candidate_count",
        "ground_truth_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
        for source in SOURCES:
            for granularity in EVAL_GRANULARITIES:
                result = aggregate[source][granularity]
                writer.writerow(
                    {
                        "instance_id": "__micro__",
                        "source": source,
                        "granularity": granularity,
                        **result["micro"],
                    }
                )
                writer.writerow(
                    {
                        "instance_id": "__macro__",
                        "source": source,
                        "granularity": granularity,
                        **result["macro_over_instances_with_truth"],
                    }
                )


def write_missing_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ("instance_id", "change_type", "granularity", "ground_truth_id")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if not record["found"]:
                writer.writerow({field: record[field] for field in fields})


def write_instance_summary(
    path: Path, instance_ids: list[str], records: list[dict[str, Any]]
) -> None:
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_instance[record["instance_id"]].append(record)
    fields = ("instance_id", "trace_found", "dependency_found", "union_found", "total")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for instance_id in instance_ids:
            items = by_instance[instance_id]
            writer.writerow(
                {
                    "instance_id": instance_id,
                    "trace_found": sum(item["trace_found"] for item in items),
                    "dependency_found": sum(
                        item["dependency_found"] for item in items
                    ),
                    "union_found": sum(item["found"] for item in items),
                    "total": len(items),
                }
            )


def write_summary(
    path: Path,
    instance_ids: list[str],
    records: list[dict[str, Any]],
    aggregate: dict[str, dict[str, dict[str, Any]]],
    dependency_files: int,
    trace_files: int,
    trace_bytes: int,
) -> None:
    lines = [
        "# Trace/dependency candidate retrieval metrics",
        "",
        f"- Selected instances: {len(instance_ids)}",
        f"- Ground-truth records: {len(records)}",
        f"- Dependency files read: {dependency_files}",
        f"- Trace files scanned: {trace_files}",
        f"- Trace bytes scanned: {trace_bytes}",
        "",
        "## Micro metrics",
        "",
        "| Source | Granularity | Candidates | Truth | TP | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCES:
        for granularity in EVAL_GRANULARITIES:
            result = aggregate[source][granularity]["micro"]
            lines.append(
                f"| {source} | {granularity} | {result['candidate_count']} | "
                f"{result['ground_truth_count']} | {result['tp']} | "
                f"{result['precision']:.4%} | {result['recall']:.4%} | "
                f"{result['f1']:.4%} |"
            )
    lines.extend(
        [
            "",
            "## Macro metrics",
            "",
            "Macro averages include only instances with ground truth at that granularity.",
            "",
            "| Source | Granularity | Instances | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for source in SOURCES:
        for granularity in EVAL_GRANULARITIES:
            result = aggregate[source][granularity]
            macro = result["macro_over_instances_with_truth"]
            lines.append(
                f"| {source} | {granularity} | {result['instances_with_truth']} | "
                f"{macro['precision']:.4%} | {macro['recall']:.4%} | "
                f"{macro['f1']:.4%} |"
            )
    source_counts = Counter(record["evidence_source"] or "not_found" for record in records)
    lines.extend(
        [
            "",
            "## Ground-truth overlap",
            "",
            f"- Both trace and dependency: {source_counts['both']}",
            f"- Trace only: {source_counts['trace']}",
            f"- Dependency only: {source_counts['dependency']}",
            f"- Neither: {source_counts['not_found']}",
            "",
            "## Normalization and limitations",
            "",
            "- Project files are paths below /testbed; external files and <string> are excluded.",
            "- Files use their project-relative path.",
            "- Classes and functions use path::leaf_name; enclosing qualification is unavailable.",
            "- exec_stack.json has no class field, so trace class candidates are empty.",
            "- Trace candidates come from CALL events plus the first root event and are deduplicated per instance.",
            "- Dependency candidates come from the file table, class/function symbols, and symbol scopes.",
            "- Precision includes test files and other project-local candidates absent from ground truth as false positives.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    instance_ids = load_instance_ids(args.instances)
    if args.instance_ids:
        requested = set(args.instance_ids)
        unknown = requested - set(instance_ids)
        if unknown:
            raise ValueError(f"Unknown instance ids: {', '.join(sorted(unknown))}")
        instance_ids = [value for value in instance_ids if value in requested]
    truth_rows = load_truth(args.truth, set(instance_ids))

    all_records: list[dict[str, Any]] = []
    truth_by_instance: dict[str, CandidateSets] = {}
    candidates_by_instance: dict[str, dict[str, CandidateSets]] = {}
    dependency_files = 0
    trace_files = 0
    trace_bytes = 0

    for index, instance_id in enumerate(instance_ids, 1):
        instance_dir = args.results / instance_id
        if not instance_dir.is_dir():
            raise FileNotFoundError(f"Missing result directory: {instance_dir}")
        records = make_records(instance_id, truth_rows[instance_id])
        dependency, dependency_evidence, loaded = extract_dependency_candidates(
            instance_dir
        )
        trace, trace_evidence, scanned, searched = extract_trace_candidates(
            instance_dir
        )
        combined = union_candidates(trace, dependency)
        apply_candidates_to_records(
            records,
            trace,
            trace_evidence,
            dependency,
            dependency_evidence,
        )
        ground_truth = truth_sets(records)
        truth_by_instance[instance_id] = ground_truth
        candidates_by_instance[instance_id] = {
            "trace": trace,
            "dependency": dependency,
            "union": combined,
        }
        all_records.extend(records)
        dependency_files += loaded
        trace_files += scanned
        trace_bytes += searched
        union_score = score(tagged_all(combined), tagged_all(ground_truth))
        print(
            f"[{index:03d}/{len(instance_ids):03d}] {instance_id}: "
            f"trace={sum(len(v) for v in trace.values())}, "
            f"dependency={sum(len(v) for v in dependency.values())}, "
            f"union_recall={union_score['recall']:.2%}",
            flush=True,
        )

    metric_rows, aggregate = build_metric_rows(
        instance_ids, truth_by_instance, candidates_by_instance
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "entity_results.jsonl", all_records)
    write_candidate_sets(
        args.output_dir / "candidate_sets.jsonl",
        instance_ids,
        truth_by_instance,
        candidates_by_instance,
    )
    write_metrics_csv(
        args.output_dir / "retrieval_metrics.csv", metric_rows, aggregate
    )
    (args.output_dir / "retrieval_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "trace-dependency-retrieval.v1",
                "identifier_normalization": "project_path_plus_leaf_name",
                "trace_class_candidates": "not_representable",
                "aggregate": aggregate,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_missing_csv(args.output_dir / "missing_entities.csv", all_records)
    write_instance_summary(
        args.output_dir / "instance_summary.csv", instance_ids, all_records
    )
    write_summary(
        args.output_dir / "summary.md",
        instance_ids,
        all_records,
        aggregate,
        dependency_files,
        trace_files,
        trace_bytes,
    )
    print(f"Wrote reports to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
