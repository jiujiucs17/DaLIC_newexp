#!/usr/bin/env python3
"""Direct BM25 feature localization over DALIC workbench repositories."""

from __future__ import annotations

import argparse
import ast
import json
import re
import tokenize
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s


CHANGE_TYPES = ("added", "edited")
GRANULARITIES = ("files", "classes", "functions")
EVAL_GRANULARITIES = (*GRANULARITIES, "all")
CUTOFFS: tuple[int | None, ...] = (3, 5, 10, 20, None)
TOKEN_RE = re.compile(r"(?u)\w+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
K1 = 1.5
B = 0.75

EntityMap = dict[str, list[str]]
TruthMap = dict[str, set[str]]


@dataclass(frozen=True)
class Scope:
    qualified_name: str
    start_line: int
    end_line: int


@dataclass
class SourceIndex:
    text: str
    lines: list[str]
    classes: list[Scope]
    functions: list[Scope]

    def source_for(self, scope: Scope) -> str:
        return "".join(self.lines[scope.start_line - 1 : scope.end_line])


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


class ScopeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []
        self.classes: list[Scope] = []
        self.functions: list[Scope] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.append(node.name)
        self.classes.append(self.make_scope(".".join(self.names), node))
        self.generic_visit(node)
        self.names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_function(node)

    def visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.names.append(node.name)
        self.functions.append(self.make_scope(".".join(self.names), node))
        self.generic_visit(node)
        self.names.pop()

    def make_scope(self, qualified_name: str, node: ast.AST) -> Scope:
        return Scope(
            qualified_name=qualified_name,
            start_line=int(node.lineno),
            end_line=int(getattr(node, "end_lineno", node.lineno)),
        )


def repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


def parse_args() -> argparse.Namespace:
    root = repo_root()
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
        help="Directory where bm25.md is written.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        help="Evaluate only this instance; may be supplied more than once.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> None:
    args.truth = args.truth.expanduser().resolve()
    args.workbench = args.workbench.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.instances is not None:
        args.instances = args.instances.expanduser().resolve()


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = parts[-1].lower() if parts else ""
    return (
        "tests" in {part.lower() for part in parts}
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def find_checkout(instance_dir: Path) -> Path:
    repo_dir = instance_dir / "repo"
    roots = sorted(path for path in repo_dir.iterdir() if path.is_dir())
    if len(roots) != 1:
        raise ValueError(f"Expected one checkout under {repo_dir}, found {len(roots)}")
    return roots[0]


def load_source_index(path: Path) -> SourceIndex | None:
    try:
        with tokenize.open(path) as handle:
            text = handle.read()
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[warning] skipped {path}: {exc}", flush=True)
        return None

    lines = text.splitlines(keepends=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            module = ast.parse(text, filename=str(path))
    except SyntaxError:
        return SourceIndex(text=text, lines=lines, classes=[], functions=[])

    visitor = ScopeVisitor()
    visitor.visit(module)
    return SourceIndex(
        text=text,
        lines=lines,
        classes=visitor.classes,
        functions=visitor.functions,
    )


def tokens(text: str) -> list[str]:
    result: list[str] = []
    for raw in TOKEN_RE.findall(text):
        expanded = [raw.casefold()]
        for snake_part in raw.split("_"):
            for part in CAMEL_RE.split(snake_part):
                normalized = part.casefold()
                if normalized and normalized not in expanded:
                    expanded.append(normalized)
        result.extend(expanded)
    return result


def bm25_rank(query: str, documents: dict[str, str]) -> list[tuple[str, float]]:
    if not documents:
        return []
    names = sorted(documents)
    query_tokens = list(dict.fromkeys(tokens(query)))
    if not query_tokens:
        return [(name, 0.0) for name in names]

    retriever = bm25s.BM25(method="lucene", k1=K1, b=B)
    retriever.index([tokens(documents[name]) for name in names], show_progress=False)
    scores = retriever.get_scores(query_tokens)
    ranked = [(name, float(scores[index])) for index, name in enumerate(names)]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def direct_bm25_candidates(instance_dir: Path) -> EntityMap:
    checkout = find_checkout(instance_dir)
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
        if is_test_path(relative):
            continue
        index = load_source_index(source_file)
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
        granularity: [name for name, _score in bm25_rank(query, documents[granularity])]
        for granularity in GRANULARITIES
    }


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
        and (path / "problem_statement.txt").is_file()
        and (path / "repo").is_dir()
    ]
    if not instance_ids:
        raise ValueError(f"No instance repositories found under {workbench}")
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
        "# DALIC Direct BM25 Ground-Truth Metrics",
        f"- K1: {K1}, B: {B}",
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
            "- `@k` is applied independently to direct BM25 file, class, and function rankings.",
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

    print(f"K1={K1}, B={B}")

    truth_by_instance: dict[str, TruthMap] = {}
    candidates_by_instance: dict[str, EntityMap] = {}
    for index, instance_id in enumerate(instance_ids, 1):
        truth_by_instance[instance_id] = truth_entities(truth_rows[instance_id])
        candidates_by_instance[instance_id] = direct_bm25_candidates(
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
    summary_path = args.output_dir / f"bm25_K1{K1}_B{B}.md"
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
