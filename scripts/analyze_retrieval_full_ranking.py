#!/usr/bin/env python3
"""Compute full-list MAP and MRR for repository retrieval methods.

By default, this script reproduces the comparison subset used in the paper:
instances from trace_completed_instance.toml for which Qwen 4B DALIC produced
a valid output. Ground-truth-empty instances are excluded separately at file,
class, and function granularity.

LSI is expensive. Independent shards can be run with --shard-count and
--shard-index. Each result includes AP/RR sums and counts, so shard results can
be combined without averaging shard-level means.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate as evaluator
from scripts.trad_methods import bm25, lsi, vsm


GRANULARITIES = evaluator.GRANULARITIES
Ranker = Callable[[Path], dict[str, list[str]]]
RANKERS: dict[str, Ranker] = {
    "bm25": bm25.direct_bm25_candidates,
    "vsm": vsm.direct_vsm_candidates,
    "lsi": lsi.direct_lsi_candidates,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(RANKERS),
        default=list(RANKERS),
        help="Repository retrieval methods to evaluate (default: all three).",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=ROOT / "dataset/swebench_feature_with_truth.jsonl",
    )
    parser.add_argument(
        "--selected-instances",
        type=Path,
        default=ROOT / "dataset/trace_completed_instance.toml",
        help="TOML manifest defining the benchmark instances.",
    )
    parser.add_argument(
        "--workbench",
        type=Path,
        default=ROOT / "dataset/agent_workbench/dalic",
        help="Workbench containing the repositories ranked by each method.",
    )
    parser.add_argument(
        "--dalic-results",
        type=Path,
        default=(
            ROOT
            / "dataset/results_archive/agent_results/dalic/Qwen_Qwen3.5-4B"
        ),
        help="DALIC result folder used to select valid comparison instances.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        help="Restrict evaluation to a valid instance; may be repeated.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of deterministic interleaved shards (default: 1).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to run (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; otherwise JSON is printed to stdout.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path, kind in (
        (args.truth, "file"),
        (args.selected_instances, "file"),
        (args.workbench, "directory"),
        (args.dalic_results, "directory"),
    ):
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not exists:
            raise ValueError(f"Expected {kind} does not exist: {path}")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")


def select_valid_instances(
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, dict[str, dict[str, set[str]]]]]:
    all_truth = evaluator.load_ground_truth_map(args.truth)
    selected = evaluator.load_selected_instance_ids(args.selected_instances)
    missing = selected - all_truth.keys()
    if missing:
        raise ValueError(f"Ground truth is missing for {len(missing)} instances")
    truth = {instance_id: all_truth[instance_id] for instance_id in selected}

    evaluations, _ = evaluator.load_instance_evaluations(
        args.dalic_results,
        truth,
        args.workbench,
    )
    valid = sorted(
        evaluation.instance_id
        for evaluation in evaluations
        if not evaluation.is_failure
    )

    if args.instance_ids:
        requested = set(args.instance_ids)
        invalid = requested - set(valid)
        if invalid:
            raise ValueError(
                "Requested instances are not valid DALIC outputs: "
                + ", ".join(sorted(invalid))
            )
        valid = [instance_id for instance_id in valid if instance_id in requested]

    return valid, truth


def average_precision_and_reciprocal_rank(
    ranking: list[str],
    ground_truth: set[str],
) -> tuple[float, float]:
    """Return AP and RR; callers must exclude empty ground truth."""
    if not ground_truth:
        raise ValueError("AP and RR are undefined for empty ground truth")

    relevant_seen: set[str] = set()
    precision_sum = 0.0
    first_relevant_rank: int | None = None
    for rank, candidate in enumerate(ranking, start=1):
        if candidate not in ground_truth or candidate in relevant_seen:
            continue
        relevant_seen.add(candidate)
        precision_sum += len(relevant_seen) / rank
        if first_relevant_rank is None:
            first_relevant_rank = rank

    ap = precision_sum / len(ground_truth)
    rr = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
    return ap, rr


def evaluate_method(
    method: str,
    ranker: Ranker,
    instance_ids: list[str],
    truth: dict[str, dict[str, dict[str, set[str]]]],
    workbench: Path,
) -> dict[str, dict[str, float | int]]:
    totals = {
        granularity: {"ap_sum": 0.0, "rr_sum": 0.0, "n": 0}
        for granularity in GRANULARITIES
    }

    progress = tqdm(
        instance_ids,
        desc=method,
        unit="instance",
        dynamic_ncols=True,
        file=sys.stderr,
    )
    for instance_id in progress:
        rankings = ranker(workbench / instance_id)
        for granularity in GRANULARITIES:
            relevant = truth[instance_id]["total"][granularity]
            if not relevant:
                continue
            ap, rr = average_precision_and_reciprocal_rank(
                rankings[granularity],
                relevant,
            )
            totals[granularity]["ap_sum"] += ap
            totals[granularity]["rr_sum"] += rr
            totals[granularity]["n"] += 1
        progress.set_postfix_str(instance_id, refresh=False)

    output: dict[str, dict[str, float | int]] = {}
    for granularity, values in totals.items():
        count = int(values["n"])
        output[granularity] = {
            "n": count,
            "ap_sum": float(values["ap_sum"]),
            "rr_sum": float(values["rr_sum"]),
            "map": float(values["ap_sum"]) / count if count else 0.0,
            "mrr": float(values["rr_sum"]) / count if count else 0.0,
        }
    return output


def main() -> int:
    args = parse_args()
    validate_args(args)
    valid_ids, truth = select_valid_instances(args)
    shard_ids = valid_ids[args.shard_index :: args.shard_count]
    if not shard_ids:
        raise ValueError("Selected shard contains no instances")

    result: dict[str, Any] = {
        "selection": {
            "valid_dalic_instances": len(valid_ids),
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "shard_instances": len(shard_ids),
            "empty_ground_truth_policy": "exclude separately by granularity",
            "average_precision_denominator": "full ground-truth size",
        },
        "methods": {},
    }
    for method in dict.fromkeys(args.methods):
        started = time.monotonic()
        metrics = evaluate_method(
            method,
            RANKERS[method],
            shard_ids,
            truth,
            args.workbench,
        )
        result["methods"][method] = {
            "elapsed_seconds": time.monotonic() - started,
            "metrics": metrics,
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
