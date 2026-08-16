#!/usr/bin/env python3
"""Compare MAP and MRR for ordered Original, Baseline, and DALIC outputs.

The authoritative evaluator determines whether an instance has a valid output.
Ranking metrics are then computed from the original JSON arrays in results.txt,
because scripts/evaluate.py intentionally converts those arrays to sets for its
precision, recall, and F1 calculations.

Two scopes are reported:

* all instances: invalid, missing, or failed outputs score zero;
* pairwise common success: Original vs Baseline and DALIC vs Baseline are
  evaluated on their respective intersections of valid outputs.

Instances with empty ground truth at a granularity are excluded from that
granularity's standalone ranking metrics.  Avg3 MAP/MRR is stricter: an
instance contributes only when its file, class, and function ground truth are
all nonempty.  Average precision divides by the full number of ground-truth
entities, so relevant entities omitted from the submitted list reduce AP.
Reciprocal rank is zero when a valid output contains no relevant entity.

For DALIC versus Baseline, the script also reports paired-bootstrap confidence
intervals and two-sided Monte Carlo sign-flip tests. Holm correction is
applied separately within the all-instance and common-success scopes across
the 12 tests formed by six models and two ranking metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate as evaluator


MODELS = (
    "Qwen_Qwen3.5-4B",
    "Qwen_Qwen3.5-9B",
    "Qwen_Qwen3.5-27B",
    "GPT_5-4",
    "DeepSeek-v4-flash",
    "DeepSeek-v4-pro",
)
METHODS = ("original", "baseline", "dalic")
PAIRWISE_COMPARISONS = (
    ("original_vs_baseline", "original", "baseline"),
    ("dalic_vs_baseline", "dalic", "baseline"),
)
GRANULARITIES = evaluator.GRANULARITIES
RANKING_METRICS = ("map", "mrr")
INFERENCE_SCOPES = ("all_instances", "common_success")
DISPLAY_NAMES = {
    "original": "Original",
    "baseline": "Baseline",
    "dalic": "DALIC",
}
COMPARISON_DISPLAY_NAMES = {
    "original_vs_baseline": "Original vs Baseline",
    "dalic_vs_baseline": "DALIC vs Baseline",
}


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


def parse_args() -> argparse.Namespace:
    root = get_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=root / "dataset/results_archive/agent_results",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=root / "dataset/swebench_feature_with_truth.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "dataset/trace_completed_instance.toml",
    )
    parser.add_argument(
        "--baseline-workbench",
        type=Path,
        default=root / "dataset/agent_workbench/baseline",
    )
    parser.add_argument(
        "--dalic-workbench",
        type=Path,
        default=root / "dataset/agent_workbench/dalic",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--permutation-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=500)
    return parser.parse_args()


def load_instance_ids(manifest_path: Path) -> list[str]:
    with manifest_path.open("rb") as handle:
        payload = tomllib.load(handle)
    instance_ids = payload.get("instance_ids")
    if not isinstance(instance_ids, list) or any(
        not isinstance(instance_id, str) for instance_id in instance_ids
    ):
        raise ValueError(f"Invalid instance_ids in {manifest_path}")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"Duplicate instance_ids in {manifest_path}")
    return sorted(instance_ids)


def load_evaluations(
    archive: Path,
    truth_path: Path,
    instance_ids: list[str],
    baseline_workbench: Path,
    dalic_workbench: Path,
) -> dict[str, dict[str, dict[str, evaluator.InstanceEvaluation]]]:
    all_truth = evaluator.load_ground_truth_map(truth_path)
    missing_truth = set(instance_ids) - all_truth.keys()
    if missing_truth:
        raise ValueError(f"Missing ground truth for {len(missing_truth)} instances")
    truth = {instance_id: all_truth[instance_id] for instance_id in instance_ids}

    output: dict[str, dict[str, dict[str, evaluator.InstanceEvaluation]]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        workbench = dalic_workbench if method == "dalic" else baseline_workbench
        for model in MODELS:
            result_folder = archive / method / model
            if not result_folder.is_dir():
                continue
            evaluations, _ = evaluator.load_instance_evaluations(
                result_folder,
                truth,
                workbench,
            )
            output[method][model] = {
                evaluation.instance_id: evaluation for evaluation in evaluations
            }
    return output


def load_ranked_predictions(
    archive: Path,
    method: str,
    model: str,
    instance_id: str,
) -> dict[str, list[str]]:
    results_path = archive / method / model / instance_id / "results.txt"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    results = payload["results"]
    ranked: dict[str, list[str]] = {}
    for granularity in GRANULARITIES:
        entries = results[granularity]
        if not isinstance(entries, list) or any(
            not isinstance(entry, str) for entry in entries
        ):
            raise RuntimeError(
                f"Authoritative evaluator accepted an unexpected ranking in {results_path}"
            )
        ranked[granularity] = entries
    return ranked


def average_precision_and_reciprocal_rank(
    ranked_predictions: list[str],
    ground_truth: set[str],
) -> tuple[float, float]:
    if not ground_truth:
        raise ValueError("Ranking metrics are undefined for empty ground truth")

    relevant_seen: set[str] = set()
    precision_sum = 0.0
    first_relevant_rank: int | None = None
    for rank, candidate in enumerate(ranked_predictions, start=1):
        if candidate not in ground_truth or candidate in relevant_seen:
            continue
        relevant_seen.add(candidate)
        precision_sum += len(relevant_seen) / rank
        if first_relevant_rank is None:
            first_relevant_rank = rank

    average_precision = precision_sum / len(ground_truth)
    reciprocal_rank = (
        0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
    )
    return average_precision, reciprocal_rank


def has_complete_ground_truth(
    evaluation: evaluator.InstanceEvaluation,
) -> bool:
    """Return whether all three granularities have nonempty ground truth."""
    return all(
        evaluation.ground_truth["total"][granularity]
        for granularity in GRANULARITIES
    )


def summarize_method(
    *,
    archive: Path,
    method: str,
    model: str,
    instance_ids: list[str],
    evaluations: dict[str, evaluator.InstanceEvaluation],
    failure_zero: bool,
) -> dict[str, Any]:
    values = {
        granularity: {"ap": [], "rr": []} for granularity in GRANULARITIES
    }
    mean3_values: dict[str, list[float]] = {"ap": [], "rr": []}
    for instance_id in instance_ids:
        evaluation = evaluations[instance_id]
        ranked = None
        if not evaluation.is_failure:
            ranked = load_ranked_predictions(archive, method, model, instance_id)
        elif not failure_zero:
            raise RuntimeError(
                f"Failure included in common-success scope: {method}/{model}/{instance_id}"
            )

        instance_values: dict[str, tuple[float, float]] = {}
        for granularity in GRANULARITIES:
            ground_truth = evaluation.ground_truth["total"][granularity]
            if not ground_truth:
                continue
            if evaluation.is_failure:
                average_precision, reciprocal_rank = 0.0, 0.0
            else:
                assert ranked is not None
                average_precision, reciprocal_rank = (
                    average_precision_and_reciprocal_rank(
                        ranked[granularity], ground_truth
                    )
                )
            instance_values[granularity] = (average_precision, reciprocal_rank)
            values[granularity]["ap"].append(average_precision)
            values[granularity]["rr"].append(reciprocal_rank)

        if has_complete_ground_truth(evaluation):
            mean3_values["ap"].append(
                sum(instance_values[g][0] for g in GRANULARITIES)
                / len(GRANULARITIES)
            )
            mean3_values["rr"].append(
                sum(instance_values[g][1] for g in GRANULARITIES)
                / len(GRANULARITIES)
            )

    summary: dict[str, Any] = {}
    for granularity in GRANULARITIES:
        average_precisions = values[granularity]["ap"]
        reciprocal_ranks = values[granularity]["rr"]
        summary[granularity] = {
            "n": len(average_precisions),
            "map": sum(average_precisions) / len(average_precisions),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        }
    summary["mean3"] = {
        "n": len(mean3_values["ap"]),
        "map": sum(mean3_values["ap"]) / len(mean3_values["ap"]),
        "mrr": sum(mean3_values["rr"]) / len(mean3_values["rr"]),
    }
    return summary


def rank_score_matrix(
    *,
    archive: Path,
    method: str,
    model: str,
    instance_ids: list[str],
    evaluations: dict[str, evaluator.InstanceEvaluation],
    failure_zero: bool,
) -> np.ndarray:
    """Return complete-GT instance x granularity x (AP, RR) scores."""
    scores = np.empty(
        (len(instance_ids), len(GRANULARITIES), len(RANKING_METRICS)),
        dtype=float,
    )
    for instance_index, instance_id in enumerate(instance_ids):
        evaluation = evaluations[instance_id]
        if not has_complete_ground_truth(evaluation):
            raise ValueError(
                f"Avg3 ranking requires nonempty ground truth at all three "
                f"granularities: {instance_id}"
            )
        ranked = None
        if not evaluation.is_failure:
            ranked = load_ranked_predictions(archive, method, model, instance_id)
        elif not failure_zero:
            raise RuntimeError(
                f"Failure included in common-success scope: {method}/{model}/{instance_id}"
            )

        for granularity_index, granularity in enumerate(GRANULARITIES):
            ground_truth = evaluation.ground_truth["total"][granularity]
            if evaluation.is_failure:
                scores[instance_index, granularity_index] = (0.0, 0.0)
            else:
                assert ranked is not None
                scores[instance_index, granularity_index] = (
                    average_precision_and_reciprocal_rank(
                        ranked[granularity], ground_truth
                    )
                )
    return scores


def cross_granularity_statistic(values: np.ndarray) -> np.ndarray:
    """Average complete-GT instances and the three granularities equally."""
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Avg3 ranking received an undefined score")
    return values.mean(axis=(-3, -2))


def paired_ranking_inference(
    delta: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap_draws: int,
    permutation_draws: int,
) -> dict[str, Any]:
    """Infer on DALIC-minus-Baseline complete-case Avg3 MAP and MRR."""
    if delta.ndim != 3 or delta.shape[1:] != (
        len(GRANULARITIES),
        len(RANKING_METRICS),
    ):
        raise ValueError(f"Unexpected paired ranking array shape: {delta.shape}")
    observed = cross_granularity_statistic(delta)

    bootstrap_values = np.empty((bootstrap_draws, len(RANKING_METRICS)))
    bootstrap_chunk = 1_000
    for start in range(0, bootstrap_draws, bootstrap_chunk):
        size = min(bootstrap_chunk, bootstrap_draws - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        bootstrap_values[start : start + size] = cross_granularity_statistic(
            delta[indices]
        )

    exceed = np.zeros(len(RANKING_METRICS), dtype=int)
    permutation_chunk = 2_000
    for start in range(0, permutation_draws, permutation_chunk):
        size = min(permutation_chunk, permutation_draws - start)
        signs = rng.choice(
            np.array([-1.0, 1.0]), size=(size, len(delta), 1, 1)
        )
        permuted = np.abs(cross_granularity_statistic(signs * delta))
        exceed += np.count_nonzero(
            permuted >= np.abs(observed)[None, :] - 1e-15,
            axis=0,
        )

    output: dict[str, Any] = {"n": len(delta)}
    for metric_index, metric in enumerate(RANKING_METRICS):
        output[metric] = {
            "delta": float(observed[metric_index]),
            "bootstrap_ci_95": [
                float(np.percentile(bootstrap_values[:, metric_index], 2.5)),
                float(np.percentile(bootstrap_values[:, metric_index], 97.5)),
            ],
            "sign_flip_p": float(
                (exceed[metric_index] + 1) / (permutation_draws + 1)
            ),
        }
    return output


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


def build_statistics(args: argparse.Namespace) -> dict[str, Any]:
    archive = args.archive.expanduser().resolve()
    truth_path = args.truth.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    baseline_workbench = args.baseline_workbench.expanduser().resolve()
    dalic_workbench = args.dalic_workbench.expanduser().resolve()
    instance_ids = load_instance_ids(manifest_path)
    evaluations = load_evaluations(
        archive,
        truth_path,
        instance_ids,
        baseline_workbench,
        dalic_workbench,
    )
    reference_evaluations = next(
        model_evaluations
        for method_evaluations in evaluations.values()
        for model_evaluations in method_evaluations.values()
    )
    mean3_eligible_instances = sum(
        has_complete_ground_truth(reference_evaluations[instance_id])
        for instance_id in instance_ids
    )

    payload: dict[str, Any] = {
        "method": {
            "manifest": str(manifest_path),
            "instances": len(instance_ids),
            "mean3_eligible_instances": mean3_eligible_instances,
            "empty_ground_truth": (
                "excluded separately from standalone granularity metrics"
            ),
            "mean3_eligibility": (
                "only instances with nonempty file, class, and function ground truth"
            ),
            "all_instances": "invalid, missing, or failed output scores zero",
            "common_success": "separate pairwise intersections for Original vs Baseline and DALIC vs Baseline",
            "average_precision_denominator": "full ground-truth size",
            "no_relevant_prediction": "AP=0 and RR=0",
            "mean3": (
                "macro mean of each eligible instance's unweighted file, class, "
                "and function AP or reciprocal-rank scores"
            ),
            "ranking_inference": "DALIC minus Baseline paired by instance",
            "bootstrap": (
                f"{args.bootstrap_draws} paired instance resamples; percentile 95% CI"
            ),
            "sign_flip": (
                f"two-sided Monte Carlo test with {args.permutation_draws} permutations; "
                "one shared sign per instance across granularities and metrics"
            ),
            "holm_families": (
                "separate all-instance and common-success families, each containing "
                "six models times MAP and MRR (12 tests)"
            ),
            "seed": args.seed,
        },
        "models": {},
    }

    raw_p_values: dict[str, dict[str, float]] = {
        scope: {} for scope in INFERENCE_SCOPES
    }

    for model in MODELS:
        available = [method for method in METHODS if model in evaluations[method]]
        if not available:
            continue
        model_output: dict[str, Any] = {
            "available_methods": available,
            "mean3_eligible_instances": sum(
                has_complete_ground_truth(
                    evaluations[available[0]][model][instance_id]
                )
                for instance_id in instance_ids
            ),
            "valid_outputs": {
                method: sum(
                    not evaluations[method][model][instance_id].is_failure
                    for instance_id in instance_ids
                )
                for method in available
            },
            "all_instances": {},
            "pairwise_common_success": {},
            "dalic_vs_baseline_inference": {},
        }
        for method in available:
            model_output["all_instances"][method] = summarize_method(
                archive=archive,
                method=method,
                model=model,
                instance_ids=instance_ids,
                evaluations=evaluations[method][model],
                failure_zero=True,
            )

        for comparison_name, first_method, second_method in PAIRWISE_COMPARISONS:
            if first_method not in available or second_method not in available:
                continue
            common_ids = [
                instance_id
                for instance_id in instance_ids
                if not evaluations[first_method][model][instance_id].is_failure
                and not evaluations[second_method][model][instance_id].is_failure
            ]
            comparison = {
                "methods": [first_method, second_method],
                "common_success_instances": len(common_ids),
                "mean3_instances": sum(
                    has_complete_ground_truth(
                        evaluations[first_method][model][instance_id]
                    )
                    for instance_id in common_ids
                ),
                "nonempty_ground_truth": {
                    granularity: sum(
                        bool(
                            evaluations[first_method][model][instance_id].ground_truth[
                                "total"
                            ][granularity]
                        )
                        for instance_id in common_ids
                    )
                    for granularity in GRANULARITIES
                },
                "results": {},
            }
            for method in (first_method, second_method):
                comparison["results"][method] = summarize_method(
                    archive=archive,
                    method=method,
                    model=model,
                    instance_ids=common_ids,
                    evaluations=evaluations[method][model],
                    failure_zero=False,
                )
            model_output["pairwise_common_success"][comparison_name] = comparison

        if "baseline" in available and "dalic" in available:
            baseline_evaluations = evaluations["baseline"][model]
            dalic_evaluations = evaluations["dalic"][model]
            common_ids = [
                instance_id
                for instance_id in instance_ids
                if not baseline_evaluations[instance_id].is_failure
                and not dalic_evaluations[instance_id].is_failure
                and has_complete_ground_truth(baseline_evaluations[instance_id])
            ]
            mean3_ids = [
                instance_id
                for instance_id in instance_ids
                if has_complete_ground_truth(baseline_evaluations[instance_id])
            ]
            for scope_index, (scope, ids, failure_zero) in enumerate(
                (
                    ("all_instances", mean3_ids, True),
                    ("common_success", common_ids, False),
                )
            ):
                baseline_scores = rank_score_matrix(
                    archive=archive,
                    method="baseline",
                    model=model,
                    instance_ids=ids,
                    evaluations=baseline_evaluations,
                    failure_zero=failure_zero,
                )
                dalic_scores = rank_score_matrix(
                    archive=archive,
                    method="dalic",
                    model=model,
                    instance_ids=ids,
                    evaluations=dalic_evaluations,
                    failure_zero=failure_zero,
                )
                if not np.array_equal(
                    np.isnan(baseline_scores), np.isnan(dalic_scores)
                ):
                    raise RuntimeError(
                        f"Ground-truth masks differ for DALIC and Baseline: {model}/{scope}"
                    )
                rng = np.random.default_rng(
                    args.seed + 10 * MODELS.index(model) + scope_index
                )
                inference = paired_ranking_inference(
                    dalic_scores - baseline_scores,
                    rng=rng,
                    bootstrap_draws=args.bootstrap_draws,
                    permutation_draws=args.permutation_draws,
                )
                model_output["dalic_vs_baseline_inference"][scope] = inference
                for metric in RANKING_METRICS:
                    raw_p_values[scope][f"{model}:{metric}"] = inference[metric][
                        "sign_flip_p"
                    ]
        payload["models"][model] = model_output

    for scope in INFERENCE_SCOPES:
        adjusted = holm_adjust(raw_p_values[scope])
        for key, adjusted_p in adjusted.items():
            model, metric = key.rsplit(":", 1)
            payload["models"][model]["dalic_vs_baseline_inference"][scope][metric][
                "holm_p_12_tests"
            ] = adjusted_p
    return payload


def format_metric(metric: dict[str, float]) -> str:
    return f"{metric['map']:.4f} / {metric['mrr']:.4f}"


def format_inference_metric(metric: dict[str, Any]) -> str:
    lower, upper = metric["bootstrap_ci_95"]
    return (
        f"{metric['delta']:+.4f} [{lower:+.4f}, {upper:+.4f}]; "
        f"p={metric['holm_p_12_tests']:.4g}"
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Agent final-output MAP/MRR",
        "",
        "Each metric cell is `MAP / MRR`. Standalone granularity metrics exclude empty ground truth at that granularity. Avg3 includes only instances with nonempty file, class, and function ground truth.",
        "",
        f"## All instances; Avg3 N={payload['method']['mean3_eligible_instances']}; invalid or failed outputs score zero",
        "",
        "| Model | Setup (valid/total) | File | Class | Function | Avg3 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    total = payload["method"]["instances"]
    for model, model_output in payload["models"].items():
        for method in model_output["available_methods"]:
            result = model_output["all_instances"][method]
            valid = model_output["valid_outputs"][method]
            lines.append(
                "| "
                + " | ".join(
                    (
                        model,
                        f"{DISPLAY_NAMES[method]} ({valid}/{total})",
                        format_metric(result["files"]),
                        format_metric(result["classes"]),
                        format_metric(result["functions"]),
                        format_metric(result["mean3"]),
                    )
                )
                + " |"
            )

    lines.extend(("", "## Pairwise common success", ""))
    for comparison_name, _, _ in PAIRWISE_COMPARISONS:
        title = COMPARISON_DISPLAY_NAMES[comparison_name]
        lines.extend(
            (
                f"### {title}",
                "",
                "| Model | Setup | Common N | Avg3 N | Nonempty N (F/C/Fn) | File | Class | Function | Avg3 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for model, model_output in payload["models"].items():
            comparison = model_output["pairwise_common_success"].get(comparison_name)
            if comparison is None:
                continue
            nonempty = comparison["nonempty_ground_truth"]
            nonempty_denominator = (
                f"{nonempty['files']}/{nonempty['classes']}/{nonempty['functions']}"
            )
            for method in comparison["methods"]:
                result = comparison["results"][method]
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            model,
                            DISPLAY_NAMES[method],
                            str(comparison["common_success_instances"]),
                            str(comparison["mean3_instances"]),
                            nonempty_denominator,
                            format_metric(result["files"]),
                            format_metric(result["classes"]),
                            format_metric(result["functions"]),
                            format_metric(result["mean3"]),
                        )
                    )
                    + " |"
                )
        lines.append("")
    lines.extend(
        (
            "## DALIC vs Baseline paired inference",
            "",
            "Each cell is `delta [95% paired-bootstrap CI]; Holm-adjusted p`. "
            "Holm correction is applied over 12 tests within each scope.",
            "",
            "| Model | Scope | N | MAP | MRR |",
            "|---|---|---:|---:|---:|",
        )
    )
    for model, model_output in payload["models"].items():
        for scope in INFERENCE_SCOPES:
            inference = model_output["dalic_vs_baseline_inference"].get(scope)
            if inference is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    (
                        model,
                        scope.replace("_", " "),
                        str(inference["n"]),
                        format_inference_metric(inference["map"]),
                        format_inference_metric(inference["mrr"]),
                    )
                )
                + " |"
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    payload = build_statistics(args)
    if args.format == "json":
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    else:
        rendered = render_markdown(payload)

    if args.output is None:
        print(rendered, end="")
        return
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
