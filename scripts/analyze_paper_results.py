#!/usr/bin/env python3
"""Generate reproducible statistics used by the DALIC paper.

The script deliberately reuses scripts/evaluate.py so that valid completion and
set-based localization metrics have exactly the same semantics as the paper's
authoritative evaluator.  It writes a JSON artifact that can be checked when
tables in the paper are updated.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate as evaluator


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "Qwen_Qwen3.5-4B",
    "Qwen_Qwen3.5-9B",
    "Qwen_Qwen3.5-27B",
    "GPT_5-4",
    "DeepSeek-v4-flash",
    "DeepSeek-v4-pro",
)
GRANULARITIES = evaluator.GRANULARITIES


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q, method="linear"))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> list[float]:
    if len(values) == 0:
        return [math.nan, math.nan]
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [percentile(samples, 2.5), percentile(samples, 97.5)]


def sign_flip_p(values: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    observed = abs(float(values.mean()))
    if not np.any(values):
        return 1.0
    exceed = 0
    done = 0
    chunk = 10_000
    while done < draws:
        size = min(chunk, draws - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
        done += size
    return (exceed + 1) / (draws + 1)


def exact_mcnemar_p(baseline_only: int, dalic_only: int) -> float:
    discordant = baseline_only + dalic_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(baseline_only, dalic_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (total - index) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def load_rows() -> tuple[dict[str, dict], list[str]]:
    with (ROOT / "dataset/trace_completed_instance.toml").open("rb") as handle:
        selected = list(tomllib.load(handle)["instance_ids"])
    selected_set = set(selected)
    rows: dict[str, dict] = {}
    with (ROOT / "dataset/swebench_feature_with_truth.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("instance_id") in selected_set:
                rows[row["instance_id"]] = row
    if set(rows) != selected_set:
        raise RuntimeError("Selected manifest and benchmark rows do not match")
    return rows, sorted(selected)


def load_evaluations(instance_ids: list[str]) -> dict[str, dict[str, dict[str, evaluator.InstanceEvaluation]]]:
    truth = evaluator.load_ground_truth_map(ROOT / "dataset/swebench_feature_with_truth.jsonl")
    truth = {instance_id: truth[instance_id] for instance_id in instance_ids}
    results: dict[str, dict[str, dict[str, evaluator.InstanceEvaluation]]] = {}
    archive = ROOT / "dataset/results_archive/agent_results"
    for method in ("original", "baseline", "dalic"):
        results[method] = {}
        for model in MODELS:
            folder = archive / method / model
            if not folder.is_dir():
                continue
            evaluations, _ = evaluator.load_instance_evaluations(
                folder, truth, ROOT / "dataset/agent_workbench" / ("dalic" if method == "dalic" else "baseline")
            )
            results[method][model] = {item.instance_id: item for item in evaluations}
    return results


def one_instance_metrics(item: evaluator.InstanceEvaluation, failure_zero: bool) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for granularity in GRANULARITIES:
        if failure_zero and item.is_failure:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            precision, recall, f1 = evaluator.instance_precision_recall_f1(
                item.predictions[granularity], item.ground_truth["total"][granularity]
            )
        output[granularity] = {"precision": precision, "recall": recall, "f1": f1}
    return output


def paired_values(
    baseline: dict[str, evaluator.InstanceEvaluation],
    dalic: dict[str, evaluator.InstanceEvaluation],
    ids: list[str],
    failure_zero: bool,
) -> tuple[
    dict[str, dict[str, float]],
    np.ndarray,
    dict[str, dict],
    dict[str, np.ndarray],
]:
    per_metric = {
        metric: {"baseline": [], "dalic": []}
        for metric in ("precision", "recall", "f1")
    }
    for instance_id in ids:
        base_metrics = one_instance_metrics(baseline[instance_id], failure_zero)
        dalic_metrics = one_instance_metrics(dalic[instance_id], failure_zero)
        for metric in per_metric:
            per_metric[metric]["baseline"].append(
                [base_metrics[granularity][metric] for granularity in GRANULARITIES]
            )
            per_metric[metric]["dalic"].append(
                [dalic_metrics[granularity][metric] for granularity in GRANULARITIES]
            )

    metric_means: dict[str, dict[str, dict[str, float]]] = {}
    metric_deltas: dict[str, np.ndarray] = {}
    for metric, values in per_metric.items():
        base_array = np.asarray(values["baseline"], dtype=float)
        dalic_array = np.asarray(values["dalic"], dtype=float)
        delta_array = dalic_array - base_array
        summary: dict[str, dict[str, float]] = {"baseline": {}, "dalic": {}, "delta": {}}
        for index, granularity in enumerate(GRANULARITIES):
            summary["baseline"][granularity] = float(base_array[:, index].mean())
            summary["dalic"][granularity] = float(dalic_array[:, index].mean())
            summary["delta"][granularity] = float(delta_array[:, index].mean())
        summary["baseline"]["avg3"] = float(base_array.mean())
        summary["dalic"]["avg3"] = float(dalic_array.mean())
        summary["delta"]["avg3"] = float(delta_array.mean())
        metric_means[metric] = summary
        metric_deltas[metric] = delta_array.mean(axis=1)
    decomposition = {metric: metric_means[metric] for metric in ("precision", "recall")}
    decomposition_deltas = {
        metric: metric_deltas[metric] for metric in ("precision", "recall")
    }
    return metric_means["f1"], metric_deltas["f1"], decomposition, decomposition_deltas


def paired_model_statistics(evaluations: dict, instance_ids: list[str], draws: int) -> dict:
    output: dict[str, dict] = {}
    raw_p_all: dict[str, float] = {}
    raw_p_common: dict[str, float] = {}
    raw_decomposition_p: dict[str, dict[str, float]] = {
        "all_instances": {},
        "common_success": {},
    }
    for model_index, model in enumerate(MODELS):
        baseline = evaluations["baseline"][model]
        dalic = evaluations["dalic"][model]
        rng = np.random.default_rng(42 + model_index)

        common_ids = [
            instance_id for instance_id in instance_ids
            if not baseline[instance_id].is_failure and not dalic[instance_id].is_failure
        ]
        model_output: dict[str, object] = {}
        for scope_index, (scope, ids, failure_zero) in enumerate((
            ("all_instances", instance_ids, True),
            ("common_success", common_ids, False),
        )):
            means, delta, decomposition, decomposition_deltas = paired_values(
                baseline, dalic, ids, failure_zero
            )
            ci = bootstrap_mean_ci(delta, rng, draws)
            p_value = sign_flip_p(delta, rng, 100_000)
            decomposition_inference: dict[str, dict[str, float]] = {}
            for metric_index, metric in enumerate(("precision", "recall")):
                decomposition_rng = np.random.default_rng(
                    1_042 + 100 * model_index + 10 * scope_index + metric_index
                )
                decomposition_p = sign_flip_p(
                    decomposition_deltas[metric], decomposition_rng, 100_000
                )
                decomposition_inference[metric] = {"sign_flip_p": decomposition_p}
                raw_decomposition_p[scope][f"{model}:{metric}"] = decomposition_p
            wins = int(np.count_nonzero(delta > 1e-12))
            losses = int(np.count_nonzero(delta < -1e-12))
            ties = len(delta) - wins - losses
            model_output[scope] = {
                "n": len(ids),
                "macro": means,
                "macro_precision_recall": decomposition,
                "precision_recall_inference": decomposition_inference,
                "avg3_delta_ci_95": ci,
                "sign_flip_p": p_value,
                "wins_ties_losses": [wins, ties, losses],
            }
            (raw_p_all if scope == "all_instances" else raw_p_common)[model] = p_value

        base_success = np.array([not baseline[i].is_failure for i in instance_ids], dtype=int)
        dalic_success = np.array([not dalic[i].is_failure for i in instance_ids], dtype=int)
        risk_delta = dalic_success - base_success
        base_only = int(np.count_nonzero((base_success == 1) & (dalic_success == 0)))
        dalic_only = int(np.count_nonzero((base_success == 0) & (dalic_success == 1)))
        model_output["completion"] = {
            "baseline": int(base_success.sum()),
            "dalic": int(dalic_success.sum()),
            "risk_difference": float(risk_delta.mean()),
            "risk_difference_ci_95": bootstrap_mean_ci(risk_delta.astype(float), rng, draws),
            "discordant_baseline_only": base_only,
            "discordant_dalic_only": dalic_only,
            "mcnemar_exact_p": exact_mcnemar_p(base_only, dalic_only),
        }
        if model in evaluations["original"]:
            original = evaluations["original"][model]
            model_output["original_completion"] = sum(not original[i].is_failure for i in instance_ids)
        output[model] = model_output

    adjusted_all = holm_adjust(raw_p_all)
    adjusted_common = holm_adjust(raw_p_common)
    for model in MODELS:
        output[model]["all_instances"]["holm_p_six_models"] = adjusted_all[model]
        output[model]["common_success"]["holm_p_six_models"] = adjusted_common[model]
    for scope, raw_p_values in raw_decomposition_p.items():
        adjusted = holm_adjust(raw_p_values)
        for key, adjusted_p in adjusted.items():
            model, metric = key.rsplit(":", 1)
            output[model][scope]["precision_recall_inference"][metric][
                "holm_p_12_tests"
            ] = adjusted_p
    return output


def normalize_truth(row: dict) -> dict[str, set[str]]:
    truth = {g: set() for g in GRANULARITIES}
    for change in ("added", "edited"):
        for granularity in GRANULARITIES:
            truth[granularity].update(row["ground_truth"].get(change, {}).get(granularity, []))
    return truth


def project_and_feature_statistics(rows: dict[str, dict], evaluations: dict, instance_ids: list[str], draws: int) -> dict:
    deltas_by_instance: dict[str, list[float]] = defaultdict(list)
    completion_by_instance: dict[str, list[int]] = defaultdict(list)
    for model in MODELS:
        baseline = evaluations["baseline"][model]
        dalic = evaluations["dalic"][model]
        for instance_id in instance_ids:
            base = one_instance_metrics(baseline[instance_id], True)
            d = one_instance_metrics(dalic[instance_id], True)
            deltas_by_instance[instance_id].append(
                float(np.mean([d[g]["f1"] - base[g]["f1"] for g in GRANULARITIES]))
            )
            completion_by_instance[instance_id].append(
                int(not dalic[instance_id].is_failure) - int(not baseline[instance_id].is_failure)
            )

    def direct_clue(row: dict, truth: dict[str, set[str]]) -> bool:
        text = row.get("problem_statement", "").lower()
        if any(path.lower() in text for path in truth["files"]):
            return True
        symbols = []
        for granularity in ("classes", "functions"):
            for item in truth[granularity]:
                symbol = item.split("::", 1)[-1].split(".")[-1]
                if len(symbol) >= 3:
                    symbols.append(symbol.lower())
        return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", text) for symbol in symbols)

    groups: dict[str, list[str]] = defaultdict(list)
    projects: dict[str, list[str]] = defaultdict(list)
    clue_flags: dict[str, bool] = {}
    for instance_id in instance_ids:
        row = rows[instance_id]
        truth = normalize_truth(row)
        project = row["repo"].split("/")[-1]
        projects[project].append(instance_id)
        function_count = len(truth["functions"])
        if function_count == 0:
            groups["functions_0"].append(instance_id)
        elif function_count == 1:
            groups["functions_1"].append(instance_id)
        elif function_count <= 5:
            groups["functions_2_to_5"].append(instance_id)
        else:
            groups["functions_over_5"].append(instance_id)
        has_added = any(row["ground_truth"].get("added", {}).get(g, []) for g in GRANULARITIES)
        groups["has_added" if has_added else "edited_only"].append(instance_id)
        clue_flags[instance_id] = direct_clue(row, truth)
        groups["direct_clue" if clue_flags[instance_id] else "no_direct_clue"].append(instance_id)

    def describe(ids: list[str]) -> dict[str, float | int]:
        values = [value for instance_id in ids for value in deltas_by_instance[instance_id]]
        completion = [value for instance_id in ids for value in completion_by_instance[instance_id]]
        return {
            "instances": len(ids),
            "model_instance_pairs": len(values),
            "pooled_avg3_delta": float(np.mean(values)),
            "pooled_completion_risk_difference": float(np.mean(completion)),
        }

    project_output = {project: describe(ids) for project, ids in sorted(projects.items())}
    group_output = {name: describe(ids) for name, ids in sorted(groups.items())}

    leave_one_out: dict[str, dict[str, float]] = {}
    for model in MODELS:
        model_values = []
        for removed_ids in projects.values():
            retained = [i for i in instance_ids if i not in set(removed_ids)]
            model_values.append(float(np.mean([deltas_by_instance[i][MODELS.index(model)] for i in retained])))
        leave_one_out[model] = {"minimum_avg3_delta": min(model_values), "maximum_avg3_delta": max(model_values)}

    project_names = sorted(projects)
    cluster_bootstrap: dict[str, dict[str, float | list[float]]] = {}
    for model_index, model in enumerate(MODELS):
        rng = np.random.default_rng(142 + model_index)
        sampled_means = []
        for _ in range(draws):
            sampled_projects = rng.choice(project_names, size=len(project_names), replace=True)
            sampled_values = [
                deltas_by_instance[instance_id][model_index]
                for project in sampled_projects
                for instance_id in projects[str(project)]
            ]
            sampled_means.append(float(np.mean(sampled_values)))
        observed = float(np.mean([deltas_by_instance[i][model_index] for i in instance_ids]))
        cluster_bootstrap[model] = {
            "avg3_delta": observed,
            "project_cluster_bootstrap_ci_95": [
                percentile(np.asarray(sampled_means), 2.5),
                percentile(np.asarray(sampled_means), 97.5),
            ],
        }

    per_model_no_clue: dict[str, dict[str, float | int]] = {}
    no_clue_ids = groups["no_direct_clue"]
    for model_index, model in enumerate(MODELS):
        values = [deltas_by_instance[i][model_index] for i in no_clue_ids]
        per_model_no_clue[model] = {"n": len(no_clue_ids), "avg3_delta": float(np.mean(values))}
    return {
        "projects": project_output,
        "feature_groups_pooled_descriptive": group_output,
        "leave_one_project_out": leave_one_out,
        "project_cluster_bootstrap": cluster_bootstrap,
        "no_direct_clue_by_model": per_model_no_clue,
        "direct_clue_rule": "exact ground-truth repository-relative path or terminal ground-truth class/function identifier in problem statement",
    }


def evidence_statistics(rows: dict[str, dict], instance_ids: list[str], evaluations: dict) -> dict:
    cutoffs = (1, 3, 5, 10, 20, 50)
    output: dict[str, dict] = {}
    per_instance_top10_recall: dict[str, list[float]] = defaultdict(list)
    for granularity in GRANULARITIES:
        candidate_sizes: list[int] = []
        total_truth = 0
        full_hits = 0
        total_candidates = 0
        cutoff_hits = {k: 0 for k in cutoffs}
        first_ranks: list[int] = []
        reciprocal_ranks: list[float] = []
        average_precisions: list[float] = []
        truth_instances = 0
        for instance_id in instance_ids:
            evidence_name = {"files": "file", "classes": "class", "functions": "function"}[granularity]
            path = ROOT / "dataset/agent_workbench/dalic" / instance_id / "dynamic_evidence" / f"{evidence_name}.json"
            candidates = json.loads(path.read_text(encoding="utf-8"))
            ranked = [str(record["qualified_name"]) for record in candidates]
            candidate_sizes.append(len(ranked))
            candidate_set = set(ranked)
            truth = normalize_truth(rows[instance_id])[granularity]
            total_candidates += len(ranked)
            total_truth += len(truth)
            full_hits += len(candidate_set & truth)
            if truth:
                truth_instances += 1
                per_instance_top10_recall[instance_id].append(len(set(ranked[:10]) & truth) / len(truth))
                relevant_seen = 0
                precision_sum = 0.0
                first = None
                for rank, candidate in enumerate(ranked, start=1):
                    if candidate in truth:
                        relevant_seen += 1
                        precision_sum += relevant_seen / rank
                        if first is None:
                            first = rank
                reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
                average_precisions.append(precision_sum / len(truth))
                if first is not None:
                    first_ranks.append(first)
            for cutoff in cutoffs:
                cutoff_hits[cutoff] += len(set(ranked[:cutoff]) & truth)
        size_array = np.asarray(candidate_sizes)
        output[granularity] = {
            "instances": len(instance_ids),
            "instances_with_nonempty_truth": truth_instances,
            "candidate_size_median": float(np.median(size_array)),
            "candidate_size_iqr": [percentile(size_array, 25), percentile(size_array, 75)],
            "candidate_size_range": [int(size_array.min()), int(size_array.max())],
            "full_micro_recall": full_hits / total_truth,
            "full_micro_precision": full_hits / total_candidates,
            "micro_recall_at_k": {str(k): cutoff_hits[k] / total_truth for k in cutoffs},
            "first_relevant_rank_median_when_found": float(np.median(first_ranks)) if first_ranks else math.nan,
            "instances_with_relevant_candidate": len(first_ranks),
            "mrr": float(np.mean(reciprocal_ranks)),
            "map": float(np.mean(average_precisions)),
        }

    # Descriptive association only: compare pooled Baseline->DALIC Avg3 deltas
    # above and below the median mean top-10 evidence recall for each instance.
    instance_coverage = {
        instance_id: float(np.mean(per_instance_top10_recall[instance_id]))
        for instance_id in instance_ids
        if per_instance_top10_recall[instance_id]
    }
    median_coverage = float(np.median(list(instance_coverage.values())))
    association: dict[str, dict] = {}
    for model in MODELS:
        baseline = evaluations["baseline"][model]
        dalic = evaluations["dalic"][model]
        buckets = {"below_median": [], "at_or_above_median": []}
        for instance_id, coverage in instance_coverage.items():
            base = one_instance_metrics(baseline[instance_id], True)
            d = one_instance_metrics(dalic[instance_id], True)
            delta = float(np.mean([d[g]["f1"] - base[g]["f1"] for g in GRANULARITIES]))
            key = "at_or_above_median" if coverage >= median_coverage else "below_median"
            buckets[key].append(delta)
        association[model] = {
            key: {"n": len(values), "avg3_delta": float(np.mean(values))}
            for key, values in buckets.items()
        }
    output["coverage_outcome_association"] = {
        "median_mean_top10_recall": median_coverage,
        "by_model": association,
        "interpretation": "descriptive association, not causal mediation",
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset/results_archive/agent_results/paper_statistics.json",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()

    rows, instance_ids = load_rows()
    evaluations = load_evaluations(instance_ids)
    payload = {
        "analysis_scope": {
            "instances": len(instance_ids),
            "manifest": "dataset/trace_completed_instance.toml",
            "failure_rule": "authoritative evaluator; invalid or missing output scores zero in all-instance analysis",
            "bootstrap_seed_family": "42 + model index",
            "bootstrap_draws": args.bootstrap_draws,
            "permutation_draws": 100_000,
            "precision_recall_permutation_seed_family": (
                "1042 + 100 * model index + 10 * scope index + metric index"
            ),
        },
        "paired_model_statistics": paired_model_statistics(evaluations, instance_ids, args.bootstrap_draws),
        "robustness": project_and_feature_statistics(rows, evaluations, instance_ids, args.bootstrap_draws),
        "runtime_evidence": evidence_statistics(rows, instance_ids, evaluations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
