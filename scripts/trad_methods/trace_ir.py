#!/usr/bin/env python3
"""Trace-filtered VSM/LSI feature-localization evaluation.

This keeps the trace-observed candidate set from ``dynamic_evidence/`` and
re-ranks those candidates with VSM or LSI using the full repository corpus as
the vectorizer/SVD background.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

try:
    from . import trace_bm25 as trace_eval
    from . import vsm as base
except ImportError:
    import trace_bm25 as trace_eval
    import vsm as base


DEFAULT_LSI_COMPONENTS = 500
RANKERS = ("vsm", "lsi")


def parse_args() -> argparse.Namespace:
    root = base.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranker",
        choices=(*RANKERS, "all"),
        default="all",
        help="Trace candidate re-ranker to evaluate.",
    )
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
        help=(
            "Optional TOML file containing instance_ids. By default all "
            "instances under --workbench are evaluated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "dataset/trad_methods_results",
        help="Directory where trace_vsm.md/trace_lsi_nc<N>.md are written.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        help="Evaluate only this instance; may be supplied more than once.",
    )
    parser.add_argument(
        "--lsi-components",
        type=int,
        default=DEFAULT_LSI_COMPONENTS,
        help="Requested LSI component count.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> None:
    args.truth = args.truth.expanduser().resolve()
    args.workbench = args.workbench.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.instances is not None:
        args.instances = args.instances.expanduser().resolve()
    if args.lsi_components < 2:
        raise ValueError("--lsi-components must be at least 2")


def repository_documents(instance_dir: Path) -> dict[str, dict[str, str]]:
    checkout = base.find_checkout(instance_dir)
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

    return documents


def rank_vsm_with_background(
    query: str,
    documents: dict[str, str],
    background_documents: list[str],
) -> list[tuple[str, float]]:
    if not documents:
        return []

    names = sorted(documents)
    corpus = [documents[name] for name in names] + background_documents
    vectorizer = TfidfVectorizer(
        tokenizer=base.tokens,
        token_pattern=None,
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
    )
    document_vectors = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    scores = linear_kernel(query_vector, document_vectors[: len(names)]).ravel()

    ranked = [(name, float(scores[index])) for index, name in enumerate(names)]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def rank_lsi_with_background(
    query: str,
    documents: dict[str, str],
    background_documents: list[str],
    n_components: int,
) -> list[tuple[str, float]]:
    if not documents:
        return []

    names = sorted(documents)
    corpus = [documents[name] for name in names] + background_documents
    vectorizer = TfidfVectorizer(
        tokenizer=base.tokens,
        token_pattern=None,
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
    )
    document_term_matrix = vectorizer.fit_transform(corpus)
    query_term_vector = vectorizer.transform([query])
    n_docs, n_terms = document_term_matrix.shape
    max_rank = min(n_docs, n_terms)

    if max_rank <= 2:
        scores = linear_kernel(
            query_term_vector,
            document_term_matrix[: len(names)],
        ).ravel()
    else:
        actual_components = max(2, min(n_components, max_rank - 1))
        svd = TruncatedSVD(
            n_components=actual_components,
            algorithm="randomized",
            n_iter=7,
            random_state=42,
        )
        lsa = make_pipeline(svd, Normalizer(copy=False))
        document_vectors = lsa.fit_transform(document_term_matrix)
        query_vector = lsa.transform(query_term_vector)
        scores = linear_kernel(query_vector, document_vectors[: len(names)]).ravel()

    ranked = [(name, float(scores[index])) for index, name in enumerate(names)]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def rerank_trace_candidates(
    instance_dir: Path,
    ranker: str,
    lsi_components: int,
) -> trace_eval.EntityMap:
    trace_candidates = trace_eval.load_candidates(instance_dir)
    all_documents = repository_documents(instance_dir)
    query = (instance_dir / "problem_statement.txt").read_text(encoding="utf-8")
    reranked = trace_eval.empty_entity_map()

    for granularity in trace_eval.GRANULARITIES:
        candidate_names = trace_candidates[granularity]
        candidate_set = set(candidate_names)
        documents = {
            name: all_documents[granularity].get(name, name)
            for name in candidate_names
        }
        background_documents = [
            document
            for name, document in all_documents[granularity].items()
            if name not in candidate_set
        ]
        if ranker == "vsm":
            ranked = rank_vsm_with_background(query, documents, background_documents)
        elif ranker == "lsi":
            ranked = rank_lsi_with_background(
                query,
                documents,
                background_documents,
                lsi_components,
            )
        else:
            raise ValueError(f"Unsupported ranker: {ranker}")
        reranked[granularity] = [name for name, _score in ranked]

    return reranked


def render_markdown(
    *,
    ranker: str,
    lsi_components: int,
    truth_path: Path,
    workbench: Path,
    instance_count: int,
    aggregate: dict[str, dict[str, dict[str, Any]]],
) -> str:
    title = {
        "vsm": "# DALIC Trace-Filtered VSM Ground-Truth Metrics",
        "lsi": "# DALIC Trace-Filtered LSI Ground-Truth Metrics",
    }[ranker]
    lines = [
        title,
        "",
        "- Candidate set: trace-observed entities from `dynamic_evidence/`.",
        "- Ranking corpus: candidate entities plus same-granularity non-test repository entities as background.",
        (
            "- VSM: `TfidfVectorizer(tokenizer=tokens, token_pattern=None, "
            'lowercase=False, sublinear_tf=True, norm="l2")` + cosine.'
        ),
    ]
    if ranker == "lsi":
        lines.extend(
            [
                f"- Requested LSI n_components: {lsi_components}",
                (
                    "- Actual LSI n_components: "
                    "`max(2, min(requested, min(X.shape) - 1))`; "
                    "uses TF-IDF cosine when `min(X.shape) <= 2`."
                ),
            ]
        )
    lines.extend(
        [
            f"- Workbench: `{workbench}`",
            f"- Ground truth: `{truth_path}`",
            f"- Evaluated instances: {instance_count}",
            "",
            "## Micro Metrics",
            "",
            "| Cutoff | Granularity | Candidates | Truth | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cutoff in trace_eval.cutoff_labels():
        for granularity in trace_eval.EVAL_GRANULARITIES:
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
    for cutoff in trace_eval.cutoff_labels():
        for granularity in trace_eval.EVAL_GRANULARITIES:
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
            "- `@k` is applied independently to file, class, and function rankings.",
            "- `all` combines tagged file/class/function candidates at the same cutoff.",
            "- Ground truth is the union of `added` and `edited` entities.",
            "- Files match by repository-relative path; classes/functions match by exact `<file_path>::<qualified_name>`.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_ranker(
    args: argparse.Namespace,
    ranker: str,
    instance_ids: list[str],
    truth_by_instance: dict[str, trace_eval.TruthMap],
) -> Path:
    candidates_by_instance: dict[str, trace_eval.EntityMap] = {}
    progress = tqdm(instance_ids, desc=f"trace-{ranker}", unit="instance")
    for instance_id in progress:
        candidates_by_instance[instance_id] = rerank_trace_candidates(
            args.workbench / instance_id,
            ranker,
            args.lsi_components,
        )

    aggregate = trace_eval.aggregate_metrics(
        instance_ids,
        truth_by_instance,
        candidates_by_instance,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = "trace_vsm.md" if ranker == "vsm" else f"trace_lsi_nc{args.lsi_components}.md"
    output_path = args.output_dir / filename
    output_path.write_text(
        render_markdown(
            ranker=ranker,
            lsi_components=args.lsi_components,
            truth_path=args.truth,
            workbench=args.workbench,
            instance_count=len(instance_ids),
            aggregate=aggregate,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    args = parse_args()
    normalize_args(args)
    instance_ids = trace_eval.selected_instance_ids(args)
    truth_rows = trace_eval.load_truth_rows(args.truth, set(instance_ids))
    truth_by_instance = {
        instance_id: trace_eval.truth_entities(truth_rows[instance_id])
        for instance_id in instance_ids
    }

    rankers = RANKERS if args.ranker == "all" else (args.ranker,)
    for ranker in rankers:
        output_path = evaluate_ranker(args, ranker, instance_ids, truth_by_instance)
        print(f"Wrote summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
