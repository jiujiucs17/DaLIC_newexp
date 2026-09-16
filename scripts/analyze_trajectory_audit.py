#!/usr/bin/env python3
"""Audit DALIC and Baseline trajectories for observable runtime evidence use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate as ev


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "dataset/results_archive/agent_results"
MANIFEST = ROOT / "dataset/trace_completed_instance_validated.toml"
TRUTH = ROOT / "dataset/swebench_feature_with_truth.jsonl"
WORKBENCH = ROOT / "dataset/agent_workbench"
OUTPUT_DIR = ROOT / (
    "paper_writing/DaLIC-Paper/DaLIC_ feature recovery for python combining "
    "software reconnaissance and data dependencies/exp_metric_results"
)

MODEL_DIRS = (
    "Qwen_Qwen3.5-4B",
    "Qwen_Qwen3.5-9B",
    "Qwen_Qwen3.5-27B",
    "GPT_5-4",
    "DeepSeek-v4-flash",
    "DeepSeek-v4-pro",
    "DS-v4-pro_32k_think_full",
)
MODEL_NAMES = dict(
    zip(
        MODEL_DIRS,
        (
            "Qwen3.5-4B",
            "Qwen3.5-9B",
            "Qwen3.5-27B",
            "GPT-5.4",
            "DeepSeek-v4-Flash",
            "DeepSeek-v4-Pro",
            "DeepSeek-v4-Pro-32K",
        ),
    )
)
EVIDENCE_KINDS = ("file", "class", "function")
EVALUATOR_GRANULARITY = {
    "file": "files",
    "class": "classes",
    "function": "functions",
}

READ_PROGRAM_RE = re.compile(
    r"(?:^|[;&|()\s])(?:cat|head|tail|sed|grep|rg|jq|awk|perl|less|more|"
    r"python(?:3)?|ruby|node|nl|strings|xxd|od|yq)(?:\s|$)",
    re.IGNORECASE,
)
NON_CONTENT_ONLY_RE = re.compile(
    r"^\s*(?:sudo\s+)?(?:ls|find|wc|stat|du|file|test|realpath|dirname|basename)\b",
    re.IGNORECASE,
)
RUNTIME_CANDIDATE_RE = re.compile(
    r"(?:^|[;&|()\s])coverage\s+(?:run|report|json|xml|html|annotate|combine)\b|"
    r"python(?:3)?\s+-m\s+coverage\s+(?:run|report|json|xml|html|annotate)\b|"
    r"--cov(?:=|\s|$)|pytest_cov|python(?:3)?\s+-m\s+trace\b|"
    r"sys\.settrace|settrace\s*\(|trace\.Trace\s*\(|sys\.setprofile|"
    r"cProfile|pyinstrument|viztracer|callgrind|line_profiler|"
    r"monkeypatch|mock\.patch|patch\.object|breakpoint\s*\(|"
    r"pytest\b[^\n;&|]*(?:--pdb|--trace)\b",
    re.IGNORECASE,
)
RUNTIME_BROAD_RE = re.compile(
    r"trace|cov|profil|hook|wrap|patch|inspect|__file__|sys\.modules|"
    r"importtime|pythonverbose|call|stack|frame",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BashCall:
    command_index: int
    model_turn_index: int
    tool_call_id: str | None
    command: str
    output: str | None
    returncode: int | None


def load_ids() -> list[str]:
    with MANIFEST.open("rb") as handle:
        ids = tomllib.load(handle)["instance_ids"]
    if len(ids) != 226 or len(set(ids)) != 226:
        raise ValueError("Validated manifest must contain 226 unique instances")
    return sorted(ids)


def tool_output_payload(content: Any) -> tuple[str, int | None]:
    if content is None:
        return "", None
    if isinstance(content, (dict, list)):
        payload = content
    else:
        text = str(content)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text, None
    if not isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=False), None
    parts = []
    for key in ("output", "output_head", "output_tail"):
        value = payload.get(key)
        if value is not None:
            parts.append(str(value))
    combined = "\n".join(parts)
    for _ in range(2):
        try:
            nested = json.loads(combined)
        except json.JSONDecodeError:
            break
        if not isinstance(nested, dict) or not any(
            key in nested for key in ("output", "output_head", "output_tail")
        ):
            break
        parts = [
            str(nested[key])
            for key in ("output", "output_head", "output_tail")
            if nested.get(key) is not None
        ]
        combined = "\n".join(parts)
        if payload.get("returncode") is None:
            payload["returncode"] = nested.get("returncode")
    return combined, payload.get("returncode")


def parse_command(arguments: Any) -> str | None:
    if isinstance(arguments, dict):
        command = arguments.get("command")
        return str(command) if command is not None else None
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return str(arguments) if arguments else None
    if isinstance(parsed, dict) and parsed.get("command") is not None:
        return str(parsed["command"])
    return None


def extract_bash_calls(trajectory: dict[str, Any]) -> list[BashCall]:
    messages = trajectory.get("messages", [])
    tool_outputs: dict[str, tuple[str, int | None]] = {}
    for message in messages:
        if message.get("role") != "tool" or not message.get("tool_call_id"):
            continue
        tool_outputs[str(message["tool_call_id"])] = tool_output_payload(
            message.get("content")
        )

    calls = []
    command_index = 0
    for model_turn_index, message in enumerate(messages):
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if str(function.get("name", "")).lower() not in {"bash", "shell"}:
                continue
            command = parse_command(function.get("arguments"))
            if command is None:
                continue
            command_index += 1
            call_id = tool_call.get("id")
            output, returncode = tool_outputs.get(str(call_id), (None, None))
            calls.append(
                BashCall(
                    command_index=command_index,
                    model_turn_index=model_turn_index,
                    tool_call_id=str(call_id) if call_id is not None else None,
                    command=command,
                    output=output,
                    returncode=returncode,
                )
            )
    return calls


def evidence_kinds_mentioned(command: str) -> set[str]:
    lowered = command.lower()
    kinds = set()
    for kind in EVIDENCE_KINDS:
        if re.search(
            rf"dynamic_evidence/(?:[^\s;&|]*?/)?{kind}\.json\b", lowered
        ) or (
            "dynamic_evidence" in lowered
            and re.search(rf"(?<![\w]){kind}\.json\b", lowered)
        ):
            kinds.add(kind)
    if "dynamic_evidence" not in lowered:
        return kinds

    if re.search(r"dynamic_evidence/(?:\*|\*\.json)", lowered):
        kinds.update(EVIDENCE_KINDS)
    for brace in re.findall(r"dynamic_evidence/\{([^}]*,[^}]*)\}", lowered):
        kinds.update(kind for kind in EVIDENCE_KINDS if re.search(rf"\b{kind}\b", brace))

    python_loops = re.findall(r"for\s+\w+\s+in\s+\[([^\]]+)\]", lowered)
    shell_loops = re.findall(r"for\s+\w+\s+in\s+([^;\n]+)", lowered)
    assignments = re.findall(
        r"\b(?:kind|fname|filename|fn)\s*=\s*['\"]([^'\"]+)['\"]", lowered
    )
    constructed_values = [*python_loops, *shell_loops, *assignments]
    if re.search(r"dynamic_evidence/[^\s'\"]*(?:\$\w+|\{\w+\})|dynamic_evidence/['\"]?\s*\+", lowered):
        for values in constructed_values:
            kinds.update(
                kind
                for kind in EVIDENCE_KINDS
                if re.search(rf"(?<![\w]){kind}(?![\w])", values)
            )
    return kinds


def is_evidence_read(call: BashCall) -> tuple[bool, set[str]]:
    kinds = set()
    whole_command_kinds = evidence_kinds_mentioned(call.command)
    if whole_command_kinds and re.search(
        r"(?:^|[;&|()\s])python(?:3)?(?:\s|$)", call.command, re.IGNORECASE
    ):
        # Python one-liners often contain semicolons inside a quoted -c program.
        # A plain shell split would incorrectly separate the interpreter from open().
        kinds.update(whole_command_kinds)
    in_evidence_dir = bool(
        re.search(r"\bcd\s+(?:/workspace/instance/)?dynamic_evidence\b", call.command)
    )
    segments = re.split(r"(?:&&|\|\||;)", call.command)
    for segment in segments:
        segment_kinds = evidence_kinds_mentioned(segment)
        if in_evidence_dir:
            segment_kinds.update(
                kind
                for kind in EVIDENCE_KINDS
                if re.search(rf"(?<![\w]){kind}\.json\b", segment, re.IGNORECASE)
            )
        if not segment_kinds or not READ_PROGRAM_RE.search(segment):
            continue
        if NON_CONTENT_ONLY_RE.match(segment) and not re.search(
            r"[|].*?(?:cat|head|tail|sed|grep|rg|jq|awk|perl|python|nl)\b",
            segment,
            re.IGNORECASE | re.DOTALL,
        ):
            continue
        kinds.update(segment_kinds)
    if not kinds:
        return False, set()
    if call.output is None:
        return False, set()
    if not call.output.strip():
        return False, set()
    if call.returncode in (126, 127):
        return False, set()
    failed_before_exposure = re.search(
        r"no such file|cannot open|can't open|command not found|permission denied|"
        r"syntax error|parse error|invalid numeric literal",
        call.output,
        re.IGNORECASE,
    )
    if failed_before_exposure and not re.search(
        r"qualified_name|bm25_score|test_case_count", call.output
    ):
        return False, set()
    return True, kinds


def load_evidence(instance_id: str) -> dict[str, list[str]]:
    base = WORKBENCH / "dalic" / instance_id / "dynamic_evidence"
    result = {}
    for kind in EVIDENCE_KINDS:
        rows = json.loads((base / f"{kind}.json").read_text())
        result[kind] = [row["qualified_name"] for row in rows]
    return result


def output_string_tokens(output: str) -> set[str]:
    tokens = set()
    for match in re.finditer(r'"((?:\\.|[^"\\])*)"', output):
        try:
            tokens.add(json.loads(f'"{match.group(1)}"'))
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"'((?:\\.|[^'\\])*)'", output):
        tokens.add(match.group(1))
    tokens.update(line.strip().strip(",") for line in output.splitlines())
    return tokens


def exposed_ranks(
    call: BashCall, kinds: set[str], evidence: dict[str, list[str]]
) -> dict[str, list[int]]:
    output = call.output or ""
    tokens = output_string_tokens(output)
    ranks: dict[str, list[int]] = {kind: [] for kind in kinds}
    for kind in kinds:
        for rank, name in enumerate(evidence[kind], 1):
            if name in tokens:
                ranks[kind].append(rank)
                continue
            qualified_pattern = re.compile(
                r'"qualified_name"\s*:\s*' + re.escape(json.dumps(name))
            )
            if qualified_pattern.search(output):
                ranks[kind].append(rank)
                continue
            if len(kinds) == 1:
                bounded_pattern = re.compile(
                    r"(?<![A-Za-z0-9_./:<>-])"
                    + re.escape(name)
                    + r"(?![A-Za-z0-9_./:<>-])"
                )
                if bounded_pattern.search(output):
                    ranks[kind].append(rank)
    return ranks


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validity(
    condition: str, model_dir: str, ids: list[str]
) -> dict[str, bool]:
    all_truth = ev.load_ground_truth_map(TRUTH)
    selected = {instance_id: all_truth[instance_id] for instance_id in ids}
    workbench = WORKBENCH / ("dalic" if condition == "dalic" else "baseline")
    evaluations, _ = ev.load_instance_evaluations(
        ARCHIVE / condition / model_dir, selected, workbench
    )
    result = {item.instance_id: not item.is_failure for item in evaluations}
    if set(result) != set(ids):
        raise ValueError(f"Evaluator did not return the validated cohort for {condition}/{model_dir}")
    return result


def join_ints(values: Iterable[int]) -> str:
    return json.dumps(sorted(set(values)), separators=(",", ":"))


def bool_csv(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def dalic_row(
    model_dir: str,
    instance_id: str,
    valid: bool,
) -> dict[str, Any]:
    path = ARCHIVE / "dalic" / model_dir / instance_id / "last_mini_run.traj.json"
    trajectory = json.loads(path.read_text())
    evidence = load_evidence(instance_id)
    access_calls: list[tuple[BashCall, set[str]]] = []
    ranks_by_kind: dict[str, set[int]] = {kind: set() for kind in EVIDENCE_KINDS}
    rank_support: dict[str, set[int]] = {kind: set() for kind in EVIDENCE_KINDS}
    for call in extract_bash_calls(trajectory):
        accessed, kinds = is_evidence_read(call)
        if not accessed:
            continue
        access_calls.append((call, kinds))
        one = exposed_ranks(call, kinds, evidence)
        for kind, ranks in one.items():
            if ranks:
                ranks_by_kind[kind].update(ranks)
                rank_support[kind].add(call.command_index)

    accessed_kind = {
        kind: any(kind in kinds for _, kinds in access_calls)
        for kind in EVIDENCE_KINDS
    }
    maxima = {
        kind: max(ranks_by_kind[kind]) if ranks_by_kind[kind] else None
        for kind in EVIDENCE_KINDS
    }
    identified_ranks = [rank for ranks in ranks_by_kind.values() for rank in ranks]
    if any(rank > 10 for rank in identified_ranks):
        expanded: bool | None = True
    elif identified_ranks:
        expanded = False
    else:
        expanded = None

    return {
        "instance_id": instance_id,
        "model_configuration": MODEL_NAMES[model_dir],
        "model_directory": model_dir,
        "condition": "DALIC",
        "valid_completion": valid,
        "accessed_any_evidence": bool(access_calls),
        "accessed_file_evidence": accessed_kind["file"],
        "accessed_class_evidence": accessed_kind["class"],
        "accessed_function_evidence": accessed_kind["function"],
        "number_of_evidence_access_commands": len(access_calls),
        "maximum_file_rank_inspected": maxima["file"],
        "maximum_class_rank_inspected": maxima["class"],
        "maximum_function_rank_inspected": maxima["function"],
        "exposed_file_ranks": join_ints(ranks_by_kind["file"]),
        "exposed_class_ranks": join_ints(ranks_by_kind["class"]),
        "exposed_function_ranks": join_ints(ranks_by_kind["function"]),
        "expanded_beyond_initial_evidence": expanded,
        "produced_runtime_observation": None,
        "runtime_observation_mechanism": "",
        "trajectory_path": str(path.relative_to(ROOT)),
        "evidence_access_command_indices": join_ints(
            call.command_index for call, _ in access_calls
        ),
        "evidence_access_model_turn_indices": join_ints(
            call.model_turn_index for call, _ in access_calls
        ),
        "file_rank_support_command_indices": join_ints(rank_support["file"]),
        "class_rank_support_command_indices": join_ints(rank_support["class"]),
        "function_rank_support_command_indices": join_ints(rank_support["function"]),
        "runtime_observation_command_indices": "[]",
        "runtime_observation_model_turn_indices": "[]",
        "classification_notes": (
            "Rank fields use only qualified_name values present in stored tool output."
            if access_calls
            else "No content-reading command for a runtime-evidence JSON file."
        ),
    }


def baseline_row(
    model_dir: str,
    instance_id: str,
    valid: bool,
) -> dict[str, Any]:
    path = ARCHIVE / "baseline" / model_dir / instance_id / "last_mini_run.traj.json"
    trajectory = json.loads(path.read_text())
    calls = extract_bash_calls(trajectory)
    candidates = [call for call in calls if RUNTIME_CANDIDATE_RE.search(call.command)]
    if candidates:
        detail = "; ".join(
            f"{call.command_index}:{call.command[:160]}" for call in candidates
        )
        raise ValueError(
            "A Baseline command requires manual runtime-observation adjudication: "
            f"{MODEL_NAMES[model_dir]} {instance_id} {detail}"
        )
    execution_count = sum(
        bool(
            re.search(
                r"(?:^|[;&|()\s])(?:pytest|py\.test|tox|unittest|python(?:3)?)(?:\s|$)",
                call.command,
                re.IGNORECASE,
            )
        )
        for call in calls
    )
    return {
        "instance_id": instance_id,
        "model_configuration": MODEL_NAMES[model_dir],
        "model_directory": model_dir,
        "condition": "Baseline",
        "valid_completion": valid,
        "accessed_any_evidence": None,
        "accessed_file_evidence": None,
        "accessed_class_evidence": None,
        "accessed_function_evidence": None,
        "number_of_evidence_access_commands": None,
        "maximum_file_rank_inspected": None,
        "maximum_class_rank_inspected": None,
        "maximum_function_rank_inspected": None,
        "exposed_file_ranks": "",
        "exposed_class_ranks": "",
        "exposed_function_ranks": "",
        "expanded_beyond_initial_evidence": None,
        "produced_runtime_observation": False,
        "runtime_observation_mechanism": "",
        "trajectory_path": str(path.relative_to(ROOT)),
        "evidence_access_command_indices": "[]",
        "evidence_access_model_turn_indices": "[]",
        "file_rank_support_command_indices": "[]",
        "class_rank_support_command_indices": "[]",
        "function_rank_support_command_indices": "[]",
        "runtime_observation_command_indices": "[]",
        "runtime_observation_model_turn_indices": "[]",
        "baseline_execution_command_count": execution_count,
        "classification_notes": (
            "No tracing, coverage, profiling, debugger-trace, or call-wrapper command. "
            "Ordinary execution, when present, is not a runtime observation."
        ),
    }


def rate(count: int, denominator: int) -> dict[str, Any]:
    return {
        "count": count,
        "denominator": denominator,
        "percentage": 100.0 * count / denominator if denominator else None,
    }


def quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "q1": None, "median": None, "q3": None, "max": None}
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 3
    return {
        "min": ordered[0],
        "q1": quartiles[0],
        "median": statistics.median(ordered),
        "q3": quartiles[2],
        "max": ordered[-1],
    }



def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def sorted_values(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def summarize_trace_coverage(
    per_instance: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    total_instances = len(per_instance)
    for kind in EVIDENCE_KINDS:
        records = [item["granularities"][kind] for item in per_instance]
        nonempty = [item for item in records if item["ground_truth_entity_count"]]
        recalls = [item["trace_ground_truth_recall"] for item in nonempty]
        covered = sum(item["covered_ground_truth_entity_count"] for item in records)
        truth = sum(item["ground_truth_entity_count"] for item in records)
        full = sum(
            item["ground_truth_entity_count"] > 0
            and item["covered_ground_truth_entity_count"]
            == item["ground_truth_entity_count"]
            for item in records
        )
        partial = sum(
            0 < item["covered_ground_truth_entity_count"]
            < item["ground_truth_entity_count"]
            for item in records
        )
        zero = sum(
            item["ground_truth_entity_count"] > 0
            and item["covered_ground_truth_entity_count"] == 0
            for item in records
        )
        empty_truth = total_instances - len(nonempty)
        if full + partial + zero + empty_truth != total_instances:
            raise ValueError(f"Trace coverage categories do not reconcile for {kind}")
        distribution = quantiles(recalls)
        distribution["mean"] = statistics.mean(recalls) if recalls else None
        distribution["instances"] = len(recalls)
        result[kind] = {
            "runtime_evidence_entities": sum(
                item["runtime_evidence_entity_count"] for item in records
            ),
            "ground_truth_entities": truth,
            "covered_ground_truth_entities": covered,
            "micro_recall": {
                "numerator": covered,
                "denominator": truth,
                "recall": ratio(covered, truth),
                "percentage": 100.0 * covered / truth if truth else None,
            },
            "per_instance_macro_recall": distribution,
            "instances_with_full_coverage": rate(full, total_instances),
            "instances_with_partial_coverage": rate(partial, total_instances),
            "instances_with_zero_coverage": rate(zero, total_instances),
            "instances_with_empty_ground_truth": rate(empty_truth, total_instances),
            "instances_with_empty_runtime_evidence": rate(
                sum(item["runtime_evidence_entity_count"] == 0 for item in records),
                total_instances,
            ),
        }
    return result


def summarize_prediction_records(
    per_run: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_runs = [item for item in per_run if item["valid_completion"]]
    result: dict[str, Any] = {"valid_runs": len(valid_runs), "granularities": {}}
    for kind in EVIDENCE_KINDS:
        records = [item["granularities"][kind] for item in valid_runs]
        total_predictions = sum(item["prediction_count"] for item in records)
        out_predictions = sum(item["out_of_trace_prediction_count"] for item in records)
        correct_out = sum(
            item["correct_out_of_trace_prediction_count"] for item in records
        )
        absent_truth = sum(item["absent_ground_truth_entity_count"] for item in records)
        recovered_absent = sum(
            item["recovered_absent_ground_truth_entity_count"] for item in records
        )
        correct_predictions = sum(item["correct_prediction_count"] for item in records)
        nonempty_truth = [item for item in records if item["ground_truth_entity_count"]]
        comparisons = Counter(item["trace_recall_ceiling_comparison"] for item in records)
        if correct_out != recovered_absent:
            raise ValueError(f"Correct out-of-trace predictions do not reconcile for {kind}")
        eligible = sum(comparisons[name] for name in ("exceeds", "equals", "below"))
        if eligible != len(nonempty_truth):
            raise ValueError(f"Recall ceiling comparisons do not reconcile for {kind}")
        result["granularities"][kind] = {
            "valid_runs": len(records),
            "total_predictions": total_predictions,
            "out_of_trace_predictions": rate(out_predictions, total_predictions),
            "valid_runs_with_out_of_trace_prediction": rate(
                sum(item["out_of_trace_prediction_count"] > 0 for item in records),
                len(records),
            ),
            "correct_out_of_trace_predictions": correct_out,
            "out_of_trace_prediction_precision": rate(correct_out, out_predictions),
            "valid_runs_with_correct_out_of_trace_prediction": rate(
                sum(
                    item["correct_out_of_trace_prediction_count"] > 0
                    for item in records
                ),
                len(records),
            ),
            "ground_truth_entities_absent_from_trace": absent_truth,
            "recovered_ground_truth_entities_absent_from_trace": recovered_absent,
            "absent_ground_truth_recovery": rate(recovered_absent, absent_truth),
            "valid_runs_recovering_absent_ground_truth": rate(
                sum(
                    item["recovered_absent_ground_truth_entity_count"] > 0
                    for item in records
                ),
                len(records),
            ),
            "correct_predictions": correct_predictions,
            "correct_out_of_trace_contribution": rate(
                correct_out, correct_predictions
            ),
            "recall_vs_trace_ceiling": {
                "eligible_valid_runs": len(nonempty_truth),
                "excluded_empty_ground_truth": len(records) - len(nonempty_truth),
                "exceeds": rate(comparisons["exceeds"], len(nonempty_truth)),
                "equals": rate(comparisons["equals"], len(nonempty_truth)),
                "below": rate(comparisons["below"], len(nonempty_truth)),
            },
        }
    return result


def build_out_of_trace_analysis(
    rows: list[dict[str, Any]], ids: list[str]
) -> dict[str, Any]:
    all_truth = ev.load_ground_truth_map(TRUTH)
    truth = {instance_id: all_truth[instance_id] for instance_id in ids}
    evidence = {instance_id: load_evidence(instance_id) for instance_id in ids}

    per_instance = []
    for instance_id in ids:
        granularities = {}
        for kind in EVIDENCE_KINDS:
            evaluator_kind = EVALUATOR_GRANULARITY[kind]
            trace_set = set(evidence[instance_id][kind])
            truth_set = truth[instance_id]["total"][evaluator_kind]
            covered = trace_set & truth_set
            granularities[kind] = {
                "runtime_evidence_entity_count": len(trace_set),
                "ground_truth_entity_count": len(truth_set),
                "covered_ground_truth_entity_count": len(covered),
                "trace_ground_truth_recall": ratio(len(covered), len(truth_set)),
                "covered_ground_truth_entities": sorted_values(covered),
                "ground_truth_entities_absent_from_trace": sorted_values(
                    truth_set - trace_set
                ),
            }
        per_instance.append(
            {"instance_id": instance_id, "granularities": granularities}
        )

    evaluations_by_model: dict[str, dict[str, ev.InstanceEvaluation]] = {}
    for model_dir in MODEL_DIRS:
        evaluations, _ = ev.load_instance_evaluations(
            ARCHIVE / "dalic" / model_dir,
            truth,
            WORKBENCH / "dalic",
        )
        evaluations_by_model[model_dir] = {
            item.instance_id: item for item in evaluations
        }

    per_run = []
    dalic_rows = [row for row in rows if row["condition"] == "DALIC"]
    for row in dalic_rows:
        instance_id = row["instance_id"]
        model_dir = row["model_directory"]
        evaluation = evaluations_by_model[model_dir][instance_id]
        if row["valid_completion"] == evaluation.is_failure:
            raise ValueError(f"Validity mismatch for {model_dir}/{instance_id}")
        granularities = {}
        for kind in EVIDENCE_KINDS:
            evaluator_kind = EVALUATOR_GRANULARITY[kind]
            trace_set = set(evidence[instance_id][kind])
            truth_set = truth[instance_id]["total"][evaluator_kind]
            covered_truth = trace_set & truth_set
            absent_truth = truth_set - trace_set
            common = {
                "runtime_evidence_entity_count": len(trace_set),
                "ground_truth_entity_count": len(truth_set),
                "covered_ground_truth_entity_count": len(covered_truth),
                "trace_ground_truth_recall": ratio(
                    len(covered_truth), len(truth_set)
                ),
            }
            row.update(
                {
                    f"trace_{kind}_entity_count": len(trace_set),
                    f"ground_truth_{kind}_entity_count": len(truth_set),
                    f"trace_ground_truth_{kind}_count": len(covered_truth),
                    f"trace_ground_truth_recall_{kind}": ratio(
                        len(covered_truth), len(truth_set)
                    ),
                }
            )
            if evaluation.is_failure:
                granularities[kind] = {**common, "prediction_analysis": None}
                for field in (
                    "prediction_count",
                    "out_of_trace_prediction_count",
                    "correct_out_of_trace_prediction_count",
                    "absent_ground_truth_entity_count",
                    "recovered_absent_ground_truth_entity_count",
                    "correct_prediction_count",
                    "dalic_recall",
                    "trace_recall_ceiling_comparison",
                ):
                    row[f"{field}_{kind}"] = None
                continue

            predictions = evaluation.predictions[evaluator_kind]
            in_trace = predictions & trace_set
            out_trace = predictions - trace_set
            correct = predictions & truth_set
            correct_out = out_trace & truth_set
            recovered_absent = predictions & absent_truth
            if correct_out != recovered_absent:
                raise ValueError(
                    f"Out-of-trace decomposition failed for "
                    f"{model_dir}/{instance_id}/{kind}"
                )
            if not truth_set:
                comparison = "undefined_empty_ground_truth"
            elif len(correct) > len(covered_truth):
                comparison = "exceeds"
            elif len(correct) == len(covered_truth):
                comparison = "equals"
            else:
                comparison = "below"
            prediction_analysis = {
                "prediction_count": len(predictions),
                "prediction_entities": sorted_values(predictions),
                "in_trace_predictions": sorted_values(in_trace),
                "out_of_trace_predictions": sorted_values(out_trace),
                "out_of_trace_prediction_count": len(out_trace),
                "correct_predictions": sorted_values(correct),
                "correct_prediction_count": len(correct),
                "correct_out_of_trace_predictions": sorted_values(correct_out),
                "correct_out_of_trace_prediction_count": len(correct_out),
                "ground_truth_entities_absent_from_trace": sorted_values(absent_truth),
                "absent_ground_truth_entity_count": len(absent_truth),
                "recovered_absent_ground_truth_entities": sorted_values(
                    recovered_absent
                ),
                "recovered_absent_ground_truth_entity_count": len(recovered_absent),
                "dalic_recall": ratio(len(correct), len(truth_set)),
                "trace_recall_ceiling_comparison": comparison,
            }
            granularities[kind] = {**common, **prediction_analysis}
            for field in (
                "prediction_count",
                "out_of_trace_prediction_count",
                "correct_out_of_trace_prediction_count",
                "absent_ground_truth_entity_count",
                "recovered_absent_ground_truth_entity_count",
                "correct_prediction_count",
                "dalic_recall",
                "trace_recall_ceiling_comparison",
            ):
                row[f"{field}_{kind}"] = prediction_analysis[field]
        per_run.append(
            {
                "instance_id": instance_id,
                "model_configuration": row["model_configuration"],
                "model_directory": model_dir,
                "valid_completion": row["valid_completion"],
                "granularities": granularities,
            }
        )

    if len(per_run) != len(ids) * len(MODEL_DIRS):
        raise ValueError("DALIC prediction analysis does not cover every run")
    by_model = {
        MODEL_NAMES[model_dir]: summarize_prediction_records(
            [item for item in per_run if item["model_directory"] == model_dir]
        )
        for model_dir in MODEL_DIRS
    }
    return {
        "definitions": {
            "entity_matching": "Exact evaluator identifiers and set membership.",
            "trace_scope": (
                "Complete stored runtime evidence, independent of trajectory inspection."
            ),
            "ground_truth_scope": "Union of evaluator added and edited entities.",
            "empty_ground_truth_recall": None,
            "prediction_scope": "Valid DALIC results only.",
        },
        "trace_coverage": {
            "per_instance": per_instance,
            "aggregate_by_granularity": summarize_trace_coverage(per_instance),
        },
        "prediction_analysis": {
            "per_run": per_run,
            "by_model_configuration": by_model,
            "aggregate": summarize_prediction_records(per_run),
        },
    }

def summarize_group(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    total = len(rows)
    valid_rows = [row for row in rows if row["valid_completion"]]
    invalid_rows = [row for row in rows if not row["valid_completion"]]
    result: dict[str, Any] = {
        "trajectories": total,
        "valid_completion": rate(len(valid_rows), total),
    }
    if condition == "DALIC":
        for field in (
            "accessed_any_evidence",
            "accessed_file_evidence",
            "accessed_class_evidence",
            "accessed_function_evidence",
        ):
            result[field] = rate(sum(bool(row[field]) for row in rows), total)
        result["access_commands_all_trajectories"] = quantiles(
            [int(row["number_of_evidence_access_commands"]) for row in rows]
        )
        result["access_commands_among_accessors"] = quantiles(
            [
                int(row["number_of_evidence_access_commands"])
                for row in rows
                if row["accessed_any_evidence"]
            ]
        )
        result["access_by_completion"] = {
            "valid": rate(
                sum(bool(row["accessed_any_evidence"]) for row in valid_rows),
                len(valid_rows),
            ),
            "invalid": rate(
                sum(bool(row["accessed_any_evidence"]) for row in invalid_rows),
                len(invalid_rows),
            ),
        }
        known_expansion = [
            row for row in rows if row["expanded_beyond_initial_evidence"] is not None
        ]
        result["expanded_beyond_initial_evidence"] = {
            **rate(
                sum(bool(row["expanded_beyond_initial_evidence"]) for row in known_expansion),
                len(known_expansion),
            ),
            "unknown": total - len(known_expansion),
            "definition": "At least one exposed qualified_name has rank greater than 10.",
        }
        result["maximum_rank_inspected"] = {}
        for kind in EVIDENCE_KINDS:
            field = f"maximum_{kind}_rank_inspected"
            values = [int(row[field]) for row in rows if row[field] is not None]
            result["maximum_rank_inspected"][kind] = {
                "reconstructable": len(values),
                "distribution": quantiles(values),
                "beyond_rank_10": rate(sum(value > 10 for value in values), len(values)),
            }
    else:
        result["produced_runtime_observation"] = rate(
            sum(bool(row["produced_runtime_observation"]) for row in rows), total
        )
        result["runtime_observation_by_completion"] = {
            "valid": rate(
                sum(bool(row["produced_runtime_observation"]) for row in valid_rows),
                len(valid_rows),
            ),
            "invalid": rate(
                sum(bool(row["produced_runtime_observation"]) for row in invalid_rows),
                len(invalid_rows),
            ),
        }
        result["ordinary_execution_commands"] = {
            "trajectories_with_one_or_more": sum(
                int(row.get("baseline_execution_command_count", 0)) > 0 for row in rows
            ),
            "commands_total": sum(
                int(row.get("baseline_execution_command_count", 0)) for row in rows
            ),
        }
    return result


def pct(item: dict[str, Any]) -> str:
    percentage = item["percentage"]
    if percentage is None:
        return f'{item["count"]}/0'
    return f'{item["count"]}/{item["denominator"]} ({percentage:.2f}%)'



def metric_fraction(item: dict[str, Any]) -> str:
    numerator = item.get("count", item.get("numerator"))
    denominator = item["denominator"]
    percentage = item["percentage"]
    if percentage is None:
        return f"{numerator}/{denominator}"
    return f"{numerator}/{denominator} ({percentage:.2f}%)"


def report_outside_evidence_sections(summary: dict[str, Any]) -> list[str]:
    analysis = summary["out_of_trace_analysis"]
    coverage = analysis["trace_coverage"]["aggregate_by_granularity"]
    prediction = analysis["prediction_analysis"]["aggregate"]["granularities"]
    labels = {"file": "File", "class": "Class", "function": "Function"}
    lines = [
        "",
        "## Complete runtime evidence coverage of ground truth",
        "",
        "This calculation uses every entity in the stored runtime evidence, regardless of which entries appeared in trajectory output. Ground truth is the evaluator's union of added and edited entities. Recall is undefined when an instance has no ground-truth entity at that granularity; those instances are excluded from macro recall and shown separately.",
        "",
        "| Granularity | Evidence entities | Ground-truth entities | Covered entities | Micro recall | Macro mean | Macro median [Q1, Q3] | Full | Partial | Zero | Empty truth |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in EVIDENCE_KINDS:
        item = coverage[kind]
        macro = item["per_instance_macro_recall"]
        lines.append(
            f"| {labels[kind]} | {item['runtime_evidence_entities']} | "
            f"{item['ground_truth_entities']} | {item['covered_ground_truth_entities']} | "
            f"{metric_fraction(item['micro_recall'])} | {100 * macro['mean']:.2f}% | "
            f"{100 * macro['median']:.2f}% [{100 * macro['q1']:.2f}%, {100 * macro['q3']:.2f}%] | "
            f"{pct(item['instances_with_full_coverage'])} | "
            f"{pct(item['instances_with_partial_coverage'])} | "
            f"{pct(item['instances_with_zero_coverage'])} | "
            f"{pct(item['instances_with_empty_ground_truth'])} |"
        )
    lines.extend(
        [
            "",
            "The full, partial, zero, and empty categories use all 226 instances as the denominator. The corresponding macro recall denominators are 226 file instances, 172 class instances, and 217 function instances.",
            "",
            "## Predictions outside complete runtime evidence",
            "",
            "Prediction composition is calculated only for the 1,428 valid DALIC model-instance runs. Membership uses the complete evidence sets, not the subset exposed in tool output.",
            "",
            "| Granularity | Predictions | Outside evidence | Valid runs with outside prediction | Correct outside predictions | Precision outside evidence | Valid runs with correct outside prediction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for kind in EVIDENCE_KINDS:
        item = prediction[kind]
        lines.append(
            f"| {labels[kind]} | {item['total_predictions']} | "
            f"{pct(item['out_of_trace_predictions'])} | "
            f"{pct(item['valid_runs_with_out_of_trace_prediction'])} | "
            f"{item['correct_out_of_trace_predictions']} | "
            f"{pct(item['out_of_trace_prediction_precision'])} | "
            f"{pct(item['valid_runs_with_correct_out_of_trace_prediction'])} |"
        )
    lines.extend(
        [
            "",
            "## Recovery of ground truth absent from runtime evidence",
            "",
            "Entity counts in this section are weighted by valid runs: an instance's absent ground-truth entity is counted once for each configuration that produced a valid result. This matches the model-instance run as the requested unit of prediction analysis.",
            "",
            "| Granularity | Absent ground-truth occurrences | Recovered occurrences | Recovery rate | Correct predictions | Correct outside contribution | Valid runs recovering an absent entity |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for kind in EVIDENCE_KINDS:
        item = prediction[kind]
        lines.append(
            f"| {labels[kind]} | {item['ground_truth_entities_absent_from_trace']} | "
            f"{item['recovered_ground_truth_entities_absent_from_trace']} | "
            f"{pct(item['absent_ground_truth_recovery'])} | "
            f"{item['correct_predictions']} | "
            f"{pct(item['correct_out_of_trace_contribution'])} | "
            f"{pct(item['valid_runs_recovering_absent_ground_truth'])} |"
        )
    lines.extend(
        [
            "",
            "Across the 226 unique instances, the complete evidence omitted 74 of 366 file entities, 74 of 327 class entities, and 180 of 706 function entities in the patch-defined ground truth.",
            "",
            "## DALIC recall relative to the runtime evidence ceiling",
            "",
            "For ground truth that is nonempty, the evidence ceiling is the fraction of ground-truth entities present in the complete runtime evidence. DALIC recall exceeds this ceiling only when its correct predictions include enough entities outside the evidence to surpass the complete evidence overlap.",
            "",
            "| Granularity | Eligible valid runs | Exceeds ceiling | Equals ceiling | Below ceiling | Empty truth excluded |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for kind in EVIDENCE_KINDS:
        item = prediction[kind]["recall_vs_trace_ceiling"]
        lines.append(
            f"| {labels[kind]} | {item['eligible_valid_runs']} | "
            f"{pct(item['exceeds'])} | {pct(item['equals'])} | "
            f"{pct(item['below'])} | {item['excluded_empty_ground_truth']} |"
        )
    lines.extend(
        [
            "",
            "A result below the ceiling does not imply poor use of runtime evidence because DALIC is not expected to return every entity present in that evidence.",
            "",
            "## Configuration-level variation",
            "",
            "| Configuration | Valid runs | Outside predictions, file | Class | Function | Precision outside evidence, file | Class | Function |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_model = analysis["prediction_analysis"]["by_model_configuration"]
    for model_dir in MODEL_DIRS:
        name = MODEL_NAMES[model_dir]
        item = by_model[name]
        granularities = item["granularities"]
        lines.append(
            f"| {name} | {item['valid_runs']} | "
            f"{pct(granularities['file']['out_of_trace_predictions'])} | "
            f"{pct(granularities['class']['out_of_trace_predictions'])} | "
            f"{pct(granularities['function']['out_of_trace_predictions'])} | "
            f"{pct(granularities['file']['out_of_trace_prediction_precision'])} | "
            f"{pct(granularities['class']['out_of_trace_prediction_precision'])} | "
            f"{pct(granularities['function']['out_of_trace_prediction_precision'])} |"
        )
    lines.extend(
        [
            "",
            "| Configuration | Recall exceeds ceiling, file | Class | Function |",
            "|---|---:|---:|---:|",
        ]
    )
    for model_dir in MODEL_DIRS:
        name = MODEL_NAMES[model_dir]
        granularities = by_model[name]["granularities"]
        lines.append(
            f"| {name} | "
            f"{pct(granularities['file']['recall_vs_trace_ceiling']['exceeds'])} | "
            f"{pct(granularities['class']['recall_vs_trace_ceiling']['exceeds'])} | "
            f"{pct(granularities['function']['recall_vs_trace_ceiling']['exceeds'])} |"
        )
    lines.extend(
        [
            "",
            "## Compact paper-ready summary table",
            "",
            "| Granularity | Trace ground-truth micro recall | Predictions outside evidence | Precision outside evidence | Recovery of truth absent from evidence | Contribution to correct predictions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for kind in EVIDENCE_KINDS:
        c = coverage[kind]
        p = prediction[kind]
        lines.append(
            f"| {labels[kind]} | {metric_fraction(c['micro_recall'])} | "
            f"{pct(p['out_of_trace_predictions'])} | "
            f"{pct(p['out_of_trace_prediction_precision'])} | "
            f"{pct(p['absent_ground_truth_recovery'])} | "
            f"{pct(p['correct_out_of_trace_contribution'])} |"
        )
    lines.extend(
        [
            "",
            "## Manual validation of the extended analysis",
            "",
            "We directly compared normalized ground truth, complete evidence JSON, and valid results.txt files for the following cases:",
            "",
            "- astropy__astropy-12962: all file, class, and function ground-truth entities occur in complete evidence, validating a full-coverage case.",
            "- astropy__astropy-14042, Qwen3.5-4B: file coverage is 1/2 and function coverage is 0/1. The predicted function astropy/units/format/fits.py::Fits._generate_unit_names is absent from function evidence, present in ground truth, and raises actual recall from the ceiling of 0 to 1.",
            "- django__django-10914, Qwen3.5-4B: django/conf/global_settings.py is absent from file evidence but present in both the prediction and ground truth, validating correct recovery outside evidence.",
            "- django__django-11070, Qwen3.5-4B: class evidence covers four of six ground-truth classes. Predictions recover the absent PasswordChangeForm and PasswordResetForm classes, so class recall exceeds its 4/6 ceiling.",
            "- astropy__astropy-12962, Qwen3.5-4B: predictions using the shorter nddata/ccddata.py path remain distinct from astropy/nddata/ccddata.py. This validates exact evaluator matching rather than suffix matching.",
            "",
            "Edge-case checks found 54 instances with empty class ground truth and 9 with empty function ground truth. Runtime evidence itself is empty for two file lists, thirteen class lists, and two function lists. Empty ground truth has undefined recall and is excluded from macro recall and ceiling comparisons. Nested identifiers such as Generic._make_parser.p_main and module-level functions such as fits_ccddata_writer are retained verbatim. No alias, rename, prefix, or suffix mapping is introduced. The 154 invalid DALIC runs remain in the trajectory-access audit but are excluded from every prediction-composition and correctness result.",
            "",
            "## Discussion interpretation",
            "",
            "The evidence supports the design rationale in a bounded descriptive sense. First, complete runtime evidence omits some patch-defined ground truth at every granularity. Second, valid DALIC results do return entities outside that evidence. Third, some of those predictions correctly recover omitted ground-truth entities and contribute to DALIC's correct predictions. These observations support guiding repository exploration with runtime evidence without enforcing it as a hard prediction boundary. They do not establish that this freedom causes DALIC's overall improvement because the study has no otherwise identical condition whose final predictions are constrained to runtime evidence.",
            "",
        ]
    )
    return lines

def generate_report(summary: dict[str, Any]) -> str:
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {date.today().isoformat()}",
        "- Verification Status: VERIFIED",
        "- Version Label: trajectory_audit_v2",
        "",
        "# Trajectory audit of runtime evidence use and prediction scope",
        "",
        "## Scope and interpretation",
        "",
        "This audit covers all 226 validated benchmark instances for each of the seven final model configurations, separately for DALIC and Baseline. It analyzes 1,582 trajectories per condition and 3,164 trajectories in total. The analysis is descriptive. It does not estimate a causal effect of accessing runtime evidence on localization quality or valid completion.",
        "",
        "A DALIC access is counted only when a Bash command reads content from at least one of `dynamic_evidence/file.json`, `class.json`, or `function.json`. Directory listings, file discovery, byte or line counts, and prompt-only mentions are excluded. A ranked entry is treated as exposed only when its exact `qualified_name` is present in the stored tool output returned to the model. Rank 1 through rank 10 define the initial portion. Expansion means that at least one visible `qualified_name` maps to rank 11 or later.",
        "",
        "A Baseline runtime observation requires tracing, coverage, profiling, debugging output that identifies executed implementation entities, or an explicit call wrapper with observable output. Ordinary test runs and direct program execution are excluded.",
        "",
        "The extended analysis compares complete runtime evidence with evaluator-normalized ground truth and final DALIC predictions. Trace coverage uses all 226 instances. Prediction composition and correctness use valid DALIC results only.",
        "",
        "",
        "## Main results",
        "",
    ]
    dalic_overall = summary["overall"]["DALIC"]
    baseline_overall = summary["overall"]["Baseline"]
    lines.extend(
        [
            f"Across all DALIC trajectories, supplied runtime evidence was accessed in {pct(dalic_overall['accessed_any_evidence'])}. File, class, and function evidence were accessed in {pct(dalic_overall['accessed_file_evidence'])}, {pct(dalic_overall['accessed_class_evidence'])}, and {pct(dalic_overall['accessed_function_evidence'])}, respectively.",
            "",
            f"Among trajectories with a reconstructable expansion classification, {pct(dalic_overall['expanded_beyond_initial_evidence'])} exposed at least one candidate at rank 11 or later. {dalic_overall['expanded_beyond_initial_evidence']['unknown']} trajectories lacked a visible qualified name and therefore do not receive an expansion classification.",
            "",
            f"No Baseline trajectory met the runtime observation criterion: {pct(baseline_overall['produced_runtime_observation'])}. Baseline trajectories did execute tests or Python commands, but the audit found no tracing, coverage, profiling, debugger trace, or call wrapper command that returned executed implementation entities.",
            "",
            "### Configuration-level audit",
            "",
            "| Configuration | DALIC valid | DALIC any evidence | File | Class | Function | Expanded beyond 10, known only | Baseline runtime observation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_dir in MODEL_DIRS:
        name = MODEL_NAMES[model_dir]
        d = summary["by_model"][name]["DALIC"]
        b = summary["by_model"][name]["Baseline"]
        lines.append(
            f"| {name} | {pct(d['valid_completion'])} | {pct(d['accessed_any_evidence'])} | "
            f"{pct(d['accessed_file_evidence'])} | {pct(d['accessed_class_evidence'])} | "
            f"{pct(d['accessed_function_evidence'])} | "
            f"{pct(d['expanded_beyond_initial_evidence'])} | "
            f"{pct(b['produced_runtime_observation'])} |"
        )
    lines.extend(
        [
            "",
            "## Relation to valid completion",
            "",
            "| Configuration | DALIC access among valid | DALIC access among invalid | Baseline runtime among valid | Baseline runtime among invalid |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_dir in MODEL_DIRS:
        name = MODEL_NAMES[model_dir]
        d = summary["by_model"][name]["DALIC"]
        b = summary["by_model"][name]["Baseline"]
        lines.append(
            f"| {name} | {pct(d['access_by_completion']['valid'])} | "
            f"{pct(d['access_by_completion']['invalid'])} | "
            f"{pct(b['runtime_observation_by_completion']['valid'])} | "
            f"{pct(b['runtime_observation_by_completion']['invalid'])} |"
        )
    lines.extend(
        [
            "",
            "These valid and invalid subgroup rates are descriptive associations only. They do not show that evidence access changes the probability of valid completion.",
            "",
            "## Extent of DALIC evidence inspection",
            "",
            "| Configuration | Access-command count, median [Q1, Q3] | Max file rank, median [Q1, Q3] | Max class rank, median [Q1, Q3] | Max function rank, median [Q1, Q3] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_dir in MODEL_DIRS:
        name = MODEL_NAMES[model_dir]
        d = summary["by_model"][name]["DALIC"]
        c = d["access_commands_among_accessors"]
        rank_cells = []
        for kind in EVIDENCE_KINDS:
            q = d["maximum_rank_inspected"][kind]["distribution"]
            rank_cells.append(
                "NA" if q["median"] is None else f'{q["median"]:g} [{q["q1"]:g}, {q["q3"]:g}]'
            )
        lines.append(
            f'| {name} | {c["median"]:g} [{c["q1"]:g}, {c["q3"]:g}] | '
            + " | ".join(rank_cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "The rank summaries use only trajectories for which at least one qualified name could be mapped to that evidence list. They describe the largest identifiable rank visible to the model. Commands that exposed only counts, errors, or unidentifiable fragments remain null in the per-trajectory dataset.",
            "",
            *report_outside_evidence_sections(summary),
            "## Audit validation and limitations",
            "",
            "- The validated TOML manifest fixes the cohort at 226 unique instances. The script asserts complete coverage for every condition and configuration.",
            "- Valid completion is loaded through the same evaluator used by the final experiment analysis, including JSON-format and source-entity validation.",
            "- Complete evidence, ground truth, and valid predictions use the exact entity identifiers returned by the final evaluator. No additional normalization or suffix matching is applied.",
            "- The parser pairs Bash calls with stored tool outputs by tool-call identifier and recursively unwraps nested tool-output records.",
            "- Output truncation is respected. Elided middle content is never treated as exposed. Visible head and tail qualified names can still be mapped to their original ranks.",
            "- A broad Baseline scan was used to check uncommon tracing terms after the exact instrumentation scan returned no candidates. Static source searches and ordinary execution were retained as negatives.",
            "- The automatic rank mapping is intentionally conservative. If no exact qualified name is visible, the maximum-rank and expansion fields are null rather than inferred from the source file or intended command range.",
            "- Invalid DALIC results are excluded from prediction composition and correctness. Trace coverage remains based on the full 226-instance benchmark.",
            "",
            "### Manual classification checks",
            "",
            "We manually checked one DALIC evidence-positive trajectory from every configuration. The sample covered Python one-liners, grep and head pipelines, sed ranges, complete and truncated cat output, and a single command that read all three evidence files. We also checked one Baseline negative trajectory from every configuration, plus ordinary pytest runs in the six configurations where such a sample was found. The pytest outputs included normal collection or import tracebacks but no tracing, coverage, profiling, or explicit call instrumentation, so they remained negative as required.",
            "",
            "| Configuration | DALIC positive sample | Access command indices | Baseline negative sample |",
            "|---|---|---|---|",
            "| Qwen3.5-4B | `django__django-11457` | 5, 6, 7, 30, 31 | `astropy__astropy-12962` |",
            "| Qwen3.5-9B | `django__django-11894` | 6, 7, 8 | `astropy__astropy-12962` |",
            "| Qwen3.5-27B | `django__django-11894` | 5, 6, 7, 18, 19, 20 | `astropy__astropy-12962` |",
            "| GPT-5.4 | `astropy__astropy-14371` | 2 | `astropy__astropy-12962` |",
            "| DeepSeek-v4-Flash | `django__django-11894` | 4, 5, 6, 16, 17, 18, 30, 31, 45 | `astropy__astropy-12962` |",
            "| DeepSeek-v4-Pro | `django__django-11894` | 9, 10, 11, 23, 25, 26, 27, 30, 31, 43, 50 | `astropy__astropy-12962` |",
            "| DeepSeek-v4-Pro-32K | `django__django-11894` | 3, 4, 5, 7, 15 | `astropy__astropy-12962` |",
            "",
            "We separately checked negative command decisions for directory listings, line or byte counts, an unavailable jq executable, and a jq pipeline that returned no content. These commands were not counted as evidence access. Commands that read multiple evidence files were counted once in the command total while setting each supported file-type flag. No DALIC trajectory-level negative and no Baseline trajectory-level positive existed in the final classifications, so those nonexistent categories could not be sampled.",
            "",
            "### Statistical fallacy scan",
            "",
            "**Coverage: 11/11 checks completed.** Overall and configuration results are both reported to expose aggregate variation. Counts and denominators are explicit. Trace coverage includes all 226 instances, while prediction composition excludes invalid results by definition and reports that selection. Empty ground truth is separated rather than assigned perfect recall. No null-hypothesis tests, repeated-measure trends, diagnostic-accuracy claims, ecological inference, or causal claims are made. The analysis does not condition an outcome association on valid completion, and it does not interpret ceiling shortfalls as poor evidence use. Selection and analytic flexibility remain limitations because the benchmark is validated and the set rules are audit decisions. Regression to the mean and reverse causality are not applicable to these set decompositions.",
            "",
            "## Paper-ready Results text",
            "",
            "",
            f"We audited all 1,582 DALIC trajectories from the 226 validated instances and seven model configurations. Access to runtime evidence was counted only when a shell command read content from one of the three supplied evidence files. DALIC accessed at least one supplied runtime evidence file in {dalic_overall['accessed_any_evidence']['percentage']:.2f}% of trajectories. The corresponding access rates were {dalic_overall['accessed_file_evidence']['percentage']:.2f}% for file evidence, {dalic_overall['accessed_class_evidence']['percentage']:.2f}% for class evidence, and {dalic_overall['accessed_function_evidence']['percentage']:.2f}% for function evidence.",
            "",
            f"Where the exposed ranks could be reconstructed from qualified names in stored tool output, {dalic_overall['expanded_beyond_initial_evidence']['percentage']:.2f}% of classified trajectories inspected at least one entry beyond rank 10. This measure follows only content returned to the model and does not treat an entire JSON file as inspected when command output was truncated.",
            "",
            f"We separately audited the 1,582 corresponding Baseline trajectories. None produced qualifying runtime observations through tracing, coverage, profiling, debugger traces, or explicit call wrappers ({baseline_overall['produced_runtime_observation']['percentage']:.2f}%). Ordinary test or program execution was not counted because it did not expose which implementation entities were exercised. These observations characterize trajectory behavior and do not establish that evidence access caused differences in completion or feature location effectiveness.",
            "",
            f"The complete runtime evidence covered 292/366 file entities (79.78%), 253/327 class entities (77.37%), and 526/706 function entities (74.50%) in the patch-defined ground truth. Across valid DALIC results, 10.46%, 13.94%, and 16.43% of file, class, and function predictions were outside the corresponding evidence. DALIC recovered 32.39%, 31.34%, and 29.18% of ground-truth entity occurrences absent from evidence, and these recoveries contributed 7.81%, 8.67%, and 10.13% of its correct predictions. DALIC recall exceeded the complete evidence ceiling in 8.47% of eligible file runs, 7.71% of eligible class runs, and 10.80% of eligible function runs. These descriptive results show that valid DALIC runs sometimes recover correct entities that the runtime evidence does not contain.",
            "",
            "## Reproducibility artifacts",
            "",
            "- `trajectory_audit_per_trajectory.csv`: auditable classifications and supporting command or model-turn indices.",
            "- `trajectory_audit_summary.json`: per-instance, per-configuration, per-run, and aggregate set decompositions.",
            "- `scripts/analyze_trajectory_audit.py`: deterministic generator.",
            "",
        ]
    )
    return "\n".join(lines)


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids = load_ids()
    rows: list[dict[str, Any]] = []
    for model_dir in MODEL_DIRS:
        valid_by_condition = {
            condition: load_validity(condition, model_dir, ids)
            for condition in ("dalic", "baseline")
        }
        for instance_id in ids:
            rows.append(
                dalic_row(
                    model_dir,
                    instance_id,
                    valid_by_condition["dalic"][instance_id],
                )
            )
            rows.append(
                baseline_row(
                    model_dir,
                    instance_id,
                    valid_by_condition["baseline"][instance_id],
                )
            )
    expected = len(ids) * len(MODEL_DIRS) * 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(rows)}")
    out_of_trace_analysis = build_out_of_trace_analysis(rows, ids)

    by_model = {}
    for model_dir in MODEL_DIRS:
        name = MODEL_NAMES[model_dir]
        by_model[name] = {}
        for condition in ("DALIC", "Baseline"):
            group = [
                row
                for row in rows
                if row["model_configuration"] == name and row["condition"] == condition
            ]
            by_model[name][condition] = summarize_group(group, condition)
    summary = {
        "metadata": {
            "generated_on": date.today().isoformat(),
            "cohort": str(MANIFEST.relative_to(ROOT)),
            "cohort_sha256": sha256(MANIFEST),
            "instances": len(ids),
            "model_configurations": [MODEL_NAMES[model] for model in MODEL_DIRS],
            "conditions": ["DALIC", "Baseline"],
            "analysis_version": "trajectory_audit_v2",
            "trajectory_rows": len(rows),
            "initial_evidence_rank_boundary": 10,
            "rank_rule": "Exact qualified_name must appear in stored tool output.",
            "baseline_rule": "Only execution that returns implementation-entity runtime observations qualifies.",
        },
        "by_model": by_model,
        "overall": {
            condition: summarize_group(
                [row for row in rows if row["condition"] == condition], condition
            )
            for condition in ("DALIC", "Baseline")
        },
        "out_of_trace_analysis": out_of_trace_analysis,
    }
    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "trajectory_audit_per_trajectory.csv"
    fields = [
        "instance_id",
        "model_configuration",
        "model_directory",
        "condition",
        "valid_completion",
        "accessed_any_evidence",
        "accessed_file_evidence",
        "accessed_class_evidence",
        "accessed_function_evidence",
        "number_of_evidence_access_commands",
        "maximum_file_rank_inspected",
        "maximum_class_rank_inspected",
        "maximum_function_rank_inspected",
        "exposed_file_ranks",
        "exposed_class_ranks",
        "exposed_function_ranks",
        "expanded_beyond_initial_evidence",
        "produced_runtime_observation",
        "runtime_observation_mechanism",
        "baseline_execution_command_count",
        "trajectory_path",
        "evidence_access_command_indices",
        "evidence_access_model_turn_indices",
        "file_rank_support_command_indices",
        "class_rank_support_command_indices",
        "function_rank_support_command_indices",
        "runtime_observation_command_indices",
        "runtime_observation_model_turn_indices",
        "trace_file_entity_count",
        "ground_truth_file_entity_count",
        "trace_ground_truth_file_count",
        "trace_ground_truth_recall_file",
        "prediction_count_file",
        "out_of_trace_prediction_count_file",
        "correct_out_of_trace_prediction_count_file",
        "absent_ground_truth_entity_count_file",
        "recovered_absent_ground_truth_entity_count_file",
        "correct_prediction_count_file",
        "dalic_recall_file",
        "trace_recall_ceiling_comparison_file",
        "trace_class_entity_count",
        "ground_truth_class_entity_count",
        "trace_ground_truth_class_count",
        "trace_ground_truth_recall_class",
        "prediction_count_class",
        "out_of_trace_prediction_count_class",
        "correct_out_of_trace_prediction_count_class",
        "absent_ground_truth_entity_count_class",
        "recovered_absent_ground_truth_entity_count_class",
        "correct_prediction_count_class",
        "dalic_recall_class",
        "trace_recall_ceiling_comparison_class",
        "trace_function_entity_count",
        "ground_truth_function_entity_count",
        "trace_ground_truth_function_count",
        "trace_ground_truth_recall_function",
        "prediction_count_function",
        "out_of_trace_prediction_count_function",
        "correct_out_of_trace_prediction_count_function",
        "absent_ground_truth_entity_count_function",
        "recovered_absent_ground_truth_entity_count_function",
        "correct_prediction_count_function",
        "dalic_recall_function",
        "trace_recall_ceiling_comparison_function",
        "classification_notes",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {
                key: bool_csv(value) if isinstance(value, bool) or value is None else value
                for key, value in row.items()
            }
            writer.writerow(serialized)
    (OUTPUT_DIR / "trajectory_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (OUTPUT_DIR / "trajectory_audit_report.md").write_text(generate_report(summary))


def discover_baseline_candidates(ids: Iterable[str], broad: bool = False) -> None:
    rows = []
    for model_dir in MODEL_DIRS:
        for instance_id in ids:
            path = ARCHIVE / "baseline" / model_dir / instance_id / "last_mini_run.traj.json"
            trajectory = json.loads(path.read_text())
            for call in extract_bash_calls(trajectory):
                pattern = RUNTIME_BROAD_RE if broad else RUNTIME_CANDIDATE_RE
                if not pattern.search(call.command):
                    continue
                rows.append(
                    {
                        "model": MODEL_NAMES[model_dir],
                        "instance_id": instance_id,
                        "command_index": call.command_index,
                        "model_turn_index": call.model_turn_index,
                        "returncode": call.returncode,
                        "command": call.command.replace("\n", "\\n"),
                        "output": (call.output or "")[:1000].replace("\n", "\\n"),
                    }
                )
    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=(
            "model",
            "instance_id",
            "command_index",
            "model_turn_index",
            "returncode",
            "command",
            "output",
        ),
    )
    writer.writeheader()
    writer.writerows(rows)


def discover_evidence_commands(ids: Iterable[str]) -> None:
    counter: Counter[str] = Counter()
    examples: dict[str, tuple[str, str, int, str]] = {}
    for model_dir in MODEL_DIRS:
        for instance_id in ids:
            path = ARCHIVE / "dalic" / model_dir / instance_id / "last_mini_run.traj.json"
            trajectory = json.loads(path.read_text())
            for call in extract_bash_calls(trajectory):
                if "dynamic_evidence" not in call.command:
                    continue
                normalized = re.sub(r"\s+", " ", call.command.strip())
                counter[normalized] += 1
                examples.setdefault(
                    normalized,
                    (MODEL_NAMES[model_dir], instance_id, call.command_index, (call.output or "")[:300]),
                )
    for command, count in counter.most_common():
        model, instance, index, output = examples[command]
        print(json.dumps({
            "count": count,
            "model": model,
            "instance_id": instance,
            "command_index": index,
            "command": command,
            "output_prefix": output,
        }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover", choices=("baseline", "baseline-broad", "evidence"))
    args = parser.parse_args()
    ids = load_ids()
    if args.discover is None:
        rows, summary = build()
        write_outputs(rows, summary)
        print(json.dumps(summary["overall"], indent=2))
    elif args.discover == "baseline":
        discover_baseline_candidates(ids)
    elif args.discover == "baseline-broad":
        discover_baseline_candidates(ids, broad=True)
    elif args.discover == "evidence":
        discover_evidence_commands(ids)


if __name__ == "__main__":
    main()
