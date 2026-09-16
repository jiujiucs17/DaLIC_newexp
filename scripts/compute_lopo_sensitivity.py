#!/usr/bin/env python3
"""Compute project-level leave-one-out sensitivity from archived agent outputs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import compute_requested_statistical_tests as stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset/results_archive/agent_results/lopo_sensitivity_results.json"
SUMMARY = ROOT / (
    "paper_writing/DaLIC-Paper/DaLIC_ feature recovery for python combining "
    "software reconnaissance and data dependencies/exp_metric_results/"
    "experiment_metrics_summary_readable.md"
)
BEGIN = "<!-- BEGIN GENERATED LOPO SENSITIVITY -->"
END = "<!-- END GENERATED LOPO SENSITIVITY -->"
GRANULARITIES = tuple(stats.G)
MODELS = tuple(stats.MODELS)
MODEL_NAMES = stats.NAMES


def project_of(instance_id: str) -> str:
    return instance_id.rsplit("-", 1)[0]


def precision_recall_f1(ranked: list[str], truth: set[str], valid: bool):
    if not valid:
        return 0.0, 0.0, 0.0
    predicted = set(ranked)
    if not predicted and not truth:
        return 1.0, 1.0, 1.0
    true_positive = len(predicted & truth)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def average_precision(ranked: list[str], truth: set[str], valid: bool) -> float:
    assert truth
    if not valid:
        return 0.0
    seen: set[str] = set()
    total = 0.0
    for rank, item in enumerate(ranked, 1):
        if item in truth and item not in seen:
            seen.add(item)
            total += len(seen) / rank
    return total / len(truth)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty cohort")
    return sum(values) / len(values)


def score_model(model: str, ids, truth, agents):
    f1_delta: dict[str, float] = {}
    map_delta: dict[str, float] = {}
    complete_ids = set(stats.rank_ids(ids, truth))

    for instance_id in ids:
        method_scores = {}
        for method in ("baseline", "dalic"):
            row = agents[method][model][instance_id]
            method_scores[method] = mean([
                precision_recall_f1(
                    row["ranked"][granularity],
                    truth[instance_id][granularity],
                    row["valid"],
                )[2]
                for granularity in GRANULARITIES
            ])
        f1_delta[instance_id] = method_scores["dalic"] - method_scores["baseline"]

        if instance_id in complete_ids:
            method_ap = {}
            for method in ("baseline", "dalic"):
                row = agents[method][model][instance_id]
                method_ap[method] = mean([
                    average_precision(
                        row["ranked"][granularity],
                        truth[instance_id][granularity],
                        row["valid"],
                    )
                    for granularity in GRANULARITIES
                ])
            map_delta[instance_id] = method_ap["dalic"] - method_ap["baseline"]

    return f1_delta, map_delta


def build():
    ids, truth, agents, _ = stats.load()
    projects = sorted({project_of(instance_id) for instance_id in ids})
    if len(projects) != 11:
        raise ValueError(f"Expected 11 projects, found {len(projects)}")

    project_counts = {
        project: {
            "instances": sum(project_of(instance_id) == project for instance_id in ids),
            "avg3_map_instances": sum(
                project_of(instance_id) == project
                and all(truth[instance_id][granularity] for granularity in GRANULARITIES)
                for instance_id in ids
            ),
        }
        for project in projects
    }
    result = {
        "metadata": {
            "effect_unit": "proportion; multiply by 100 for percentage points",
            "membership_instances": len(ids),
            "avg3_map_instances": len(stats.rank_ids(ids, truth)),
            "projects": project_counts,
            "invalid_output": "zero Avg3 F1 and zero Avg3 AP",
            "project_key": "instance_id with the final hyphen-delimited issue number removed",
            "cohort": str(stats.MANIFEST.relative_to(ROOT)),
            "cohort_sha256": hashlib.sha256(stats.MANIFEST.read_bytes()).hexdigest(),
            "truth": str(stats.TRUTH.relative_to(ROOT)),
            "truth_sha256": hashlib.sha256(stats.TRUTH.read_bytes()).hexdigest(),
            "agent_archive": str(stats.ARCH.relative_to(ROOT)),
            "models": list(MODELS),
        },
        "models": {},
    }

    for model in MODELS:
        f1_delta, map_delta = score_model(model, ids, truth, agents)
        lopo = {}
        for omitted_project in projects:
            retained = [i for i in ids if project_of(i) != omitted_project]
            retained_map = [i for i in retained if i in map_delta]
            lopo[omitted_project] = {
                "membership_instances": len(retained),
                "avg3_map_instances": len(retained_map),
                "avg3_f1_difference": mean([f1_delta[i] for i in retained]),
                "avg3_map_difference": mean([map_delta[i] for i in retained_map]),
            }

        min_f1_project = min(projects, key=lambda p: lopo[p]["avg3_f1_difference"])
        max_f1_project = max(projects, key=lambda p: lopo[p]["avg3_f1_difference"])
        min_map_project = min(projects, key=lambda p: lopo[p]["avg3_map_difference"])
        max_map_project = max(projects, key=lambda p: lopo[p]["avg3_map_difference"])
        result["models"][model] = {
            "full": {
                "membership_instances": len(ids),
                "avg3_map_instances": len(map_delta),
                "avg3_f1_difference": mean(list(f1_delta.values())),
                "avg3_map_difference": mean(list(map_delta.values())),
            },
            "lopo": lopo,
            "range": {
                "min_avg3_f1": {
                    "omitted_project": min_f1_project,
                    "difference": lopo[min_f1_project]["avg3_f1_difference"],
                },
                "max_avg3_f1": {
                    "omitted_project": max_f1_project,
                    "difference": lopo[max_f1_project]["avg3_f1_difference"],
                },
                "min_avg3_map": {
                    "omitted_project": min_map_project,
                    "difference": lopo[min_map_project]["avg3_map_difference"],
                },
                "max_avg3_map": {
                    "omitted_project": max_map_project,
                    "difference": lopo[max_map_project]["avg3_map_difference"],
                },
            },
        }
    return result


def pp(value: float) -> str:
    return f"{100 * value:+.2f}"


def render(data) -> str:
    projects = list(data["metadata"]["projects"])
    lines = [
        BEGIN,
        "## 9. Leave-One-Project-Out sensitivity analysis",
        "",
        "This analysis checks whether the DALIC minus Baseline All Instances effect is driven by a single project. Each of the 11 projects is omitted in turn. Invalid agent output contributes zero F1 and zero AP. No additional significance test or multiplicity correction is applied.",
        "",
        "Avg3 F1 uses all retained instances. Avg3 MAP uses retained instances whose file, class, and function ground truth are all nonempty. All differences below are percentage points and use the direction DALIC minus Baseline.",
        "",
        "### 9.1 Full-data effects and LOPO ranges",
        "",
        "| Model | Full ΔAvg3 F1 | Minimum LOPO ΔF1 (omitted project) | Maximum LOPO ΔF1 (omitted project) | Full ΔAvg3 MAP | Minimum LOPO ΔMAP (omitted project) | Maximum LOPO ΔMAP (omitted project) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = data["models"][model]
        bounds = row["range"]
        lines.append(
            f"| {MODEL_NAMES[model]} | {pp(row['full']['avg3_f1_difference'])} | "
            f"{pp(bounds['min_avg3_f1']['difference'])} (`{bounds['min_avg3_f1']['omitted_project']}`) | "
            f"{pp(bounds['max_avg3_f1']['difference'])} (`{bounds['max_avg3_f1']['omitted_project']}`) | "
            f"{pp(row['full']['avg3_map_difference'])} | "
            f"{pp(bounds['min_avg3_map']['difference'])} (`{bounds['min_avg3_map']['omitted_project']}`) | "
            f"{pp(bounds['max_avg3_map']['difference'])} (`{bounds['max_avg3_map']['omitted_project']}`) |"
        )

    lines += [
        "",
        "### 9.2 Project cohort sizes",
        "",
        "| Project | Omitted instances | Omitted Avg3 MAP instances | Retained F1 N | Retained MAP N |",
        "|---|---:|---:|---:|---:|",
    ]
    for project in projects:
        counts = data["metadata"]["projects"][project]
        lines.append(
            f"| `{project}` | {counts['instances']} | {counts['avg3_map_instances']} | "
            f"{data['metadata']['membership_instances'] - counts['instances']} | "
            f"{data['metadata']['avg3_map_instances'] - counts['avg3_map_instances']} |"
        )

    lines += [
        "",
        "### 9.3 Individual LOPO Avg3 F1 differences",
        "",
        "| Omitted project | Retained N | " + " | ".join(MODEL_NAMES[m] for m in MODELS) + " |",
        "|---|---:|" + "---:|" * len(MODELS),
    ]
    for project in projects:
        first = data["models"][MODELS[0]]["lopo"][project]
        values = [pp(data["models"][model]["lopo"][project]["avg3_f1_difference"]) for model in MODELS]
        lines.append(f"| `{project}` | {first['membership_instances']} | " + " | ".join(values) + " |")

    lines += [
        "",
        "### 9.4 Individual LOPO Avg3 MAP differences",
        "",
        "| Omitted project | Retained MAP N | " + " | ".join(MODEL_NAMES[m] for m in MODELS) + " |",
        "|---|---:|" + "---:|" * len(MODELS),
    ]
    for project in projects:
        first = data["models"][MODELS[0]]["lopo"][project]
        values = [pp(data["models"][model]["lopo"][project]["avg3_map_difference"]) for model in MODELS]
        lines.append(f"| `{project}` | {first['avg3_map_instances']} | " + " | ".join(values) + " |")

    lines += [
        "",
        "### 9.5 Replication metadata",
        "",
        f"- Machine-readable output: `{OUT.relative_to(ROOT)}`.",
        f"- Membership cohort: {data['metadata']['membership_instances']} instances.",
        f"- Full Avg3 MAP cohort: {data['metadata']['avg3_map_instances']} instances.",
        f"- Cohort SHA-256: `{data['metadata']['cohort_sha256']}`.",
        f"- Ground-truth SHA-256: `{data['metadata']['truth_sha256']}`.",
        "- The JSON artifact stores all 11 project-removal results for every model.",
        "",
        END,
    ]
    return "\n".join(lines) + "\n"


def main():
    data = build()
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    generated = render(data)
    text = SUMMARY.read_text()
    if BEGIN in text:
        start = text.index(BEGIN)
        stop = text.index(END, start) + len(END)
        text = text[:start].rstrip() + "\n\n" + generated.rstrip() + text[stop:]
    else:
        text = text.rstrip() + "\n\n" + generated
    SUMMARY.write_text(text.rstrip() + "\n")
    print(f"Wrote {OUT}")
    print(f"Updated {SUMMARY}")


if __name__ == "__main__":
    main()
