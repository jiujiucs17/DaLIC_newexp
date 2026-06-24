#!/usr/bin/env python3
"""Direct LSI feature localization over DALIC workbench repositories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

try:
    from . import vsm as base
except ImportError:
    import vsm as base


N_COMPONENT = 200


def parse_args() -> argparse.Namespace:
    root = base.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth",
        type=Path,
        default=root / "dataset/swebench_feature_with_truth.jsonl",
        help="Ground-truth JSONL file.",
    )
    parser.add_argument(
        "--workbench",
        type=Path,
        default=root / "dataset/agent_workbench/dalic",
        help="DALIC workbench containing per-instance repositories.",
    )
    parser.add_argument(
        "--instances",
        "--selected-instances",
        dest="instances",
        type=Path,
        default=None,
        help="Optional TOML file containing instance_ids.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "dataset/trad_methods_results",
        help="Directory where lsi_nc<N_COMPONENT>.md is written.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        help="Evaluate only this instance; may be supplied more than once.",
    )
    return parser.parse_args()


def lsi_rank(query: str, documents: dict[str, str]) -> list[tuple[str, float]]:
    if not documents:
        return []

    names = sorted(documents)
    vectorizer = TfidfVectorizer(
        tokenizer=base.tokens,
        token_pattern=None,
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
    )
    document_term_matrix = vectorizer.fit_transform(documents[name] for name in names)
    n_components = min(
        N_COMPONENT,
        document_term_matrix.shape[0] - 1,
        document_term_matrix.shape[1] - 1,
    )
    if n_components < 1:
        return [(name, 0.0) for name in names]

    svd = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        n_iter=7,
        random_state=42,
    )
    lsa = make_pipeline(
        svd,
        Normalizer(copy=False),
    )
    document_vectors = lsa.fit_transform(document_term_matrix)
    query_term_vector = vectorizer.transform([query])
    query_vector = lsa.transform(query_term_vector)
    scores = linear_kernel(query_vector, document_vectors).ravel()

    ranked = [(name, float(scores[index])) for index, name in enumerate(names)]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def direct_lsi_candidates(instance_dir: Path) -> base.EntityMap:
    checkout = base.find_checkout(instance_dir)
    query = (instance_dir / "problem_statement.txt").read_text(encoding="utf-8")
    documents: dict[str, dict[str, str]] = {
        "files": {},
        "classes": {},
        "functions": {},
    }

    for source_file in sorted(checkout.rglob("*.py")):
        if not source_file.is_file():
            continue
        relative = source_file.relative_to(checkout).as_posix()
        if base.is_test_path(relative):
            continue
        index = base.load_source_index(source_file)
        if index is None:
            continue

        documents["files"][relative] = f"{relative} {index.text}"
        for scope in index.classes:
            name = f"{relative}::{scope.qualified_name}"
            documents["classes"][name] = f"{name} {index.source_for(scope)}"
        for scope in index.functions:
            name = f"{relative}::{scope.qualified_name}"
            documents["functions"][name] = f"{name} {index.source_for(scope)}"

    return {
        granularity: [name for name, _score in lsi_rank(query, documents[granularity])]
        for granularity in base.GRANULARITIES
    }


def render_markdown(
    *,
    truth_path: Path,
    workbench: Path,
    instance_count: int,
    aggregate: dict[str, dict[str, dict[str, Any]]],
) -> str:
    lines = [
        "# DALIC Direct LSI Ground-Truth Metrics",
        f"- Requested n_components: {N_COMPONENT}",
        (
            "- Actual n_components: "
            "`min(N_COMPONENT, X.shape[0] - 1, X.shape[1] - 1)` per ranking"
        ),
        (
            "- Vectorizer: `TfidfVectorizer(tokenizer=tokens, "
            'token_pattern=None, lowercase=False, sublinear_tf=True, norm="l2")`'
        ),
        (
            "- LSI: `TruncatedSVD(algorithm=\"randomized\", n_iter=7, "
            'random_state=42)` followed by `Normalizer(copy=False)`'
        ),
        "- Scoring: `linear_kernel(query_vector, document_vectors)`",
        f"- Workbench: `{workbench}`",
        f"- Ground truth: `{truth_path}`",
        f"- Evaluated instances: {instance_count}",
        "",
        "## Micro Metrics",
        "",
        "| Cutoff | Granularity | Candidates | Truth | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cutoff in base.cutoff_labels():
        for granularity in base.EVAL_GRANULARITIES:
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
    for cutoff in base.cutoff_labels():
        for granularity in base.EVAL_GRANULARITIES:
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
            "- `@k` is applied independently to direct LSI file, class, and function rankings.",
            "- `all` combines tagged file/class/function candidates at the same cutoff.",
            "- Ground truth is the union of `added` and `edited` entities.",
            "- Files match by repository-relative path; classes/functions match by exact `<file_path>::<qualified_name>`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    base.normalize_args(args)
    instance_ids = base.selected_instance_ids(args)
    truth_rows = base.load_truth_rows(args.truth, set(instance_ids))

    print(f"N_COMPONENT={N_COMPONENT}")
    print(
        'Vectorizer=TfidfVectorizer(token_pattern=None, lowercase=False, sublinear_tf=True, norm="l2")'
    )
    print('LSI=TruncatedSVD(algorithm="randomized", n_iter=7, random_state=42)+Normalizer')
    print("Scoring=linear_kernel")

    truth_by_instance: dict[str, base.TruthMap] = {}
    candidates_by_instance: dict[str, base.EntityMap] = {}
    for index, instance_id in enumerate(instance_ids, 1):
        truth_by_instance[instance_id] = base.truth_entities(truth_rows[instance_id])
        candidates_by_instance[instance_id] = direct_lsi_candidates(
            args.workbench / instance_id
        )
        full_score = base.score(
            base.candidates_at(candidates_by_instance[instance_id], "all", None),
            base.truth_at(truth_by_instance[instance_id], "all"),
        )
        print(
            f"[{index:03d}/{len(instance_ids):03d}] {instance_id}: "
            f"P/R/F1@Full={full_score.precision:.2%}/"
            f"{full_score.recall:.2%}/{full_score.f1:.2%}",
            flush=True,
        )

    aggregate = base.aggregate_metrics(
        instance_ids, truth_by_instance, candidates_by_instance
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"lsi_nc{N_COMPONENT}.md"
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
