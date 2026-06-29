#!/usr/bin/env python3
# Options:
#   --agent_results PATH
#       default: dataset/results_archive/agent_results/dalic/Qwen_Qwen3.5-9B
#       output: BM25_variable_cutoff.txt under this folder

"""BM25 feature localization using per-instance cutoffs from agent results."""

from __future__ import annotations

import argparse
import ast
import json
import re
import tokenize
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import bm25s
from tqdm import tqdm


DEFAULT_AGENT_RESULTS = Path(
    "dataset/results_archive/agent_results/dalic/Qwen_Qwen3.5-9B"
)
DEFAULT_INPUT_PATH = Path("dataset/swebench_feature_with_truth.jsonl")
DEFAULT_SELECTED_INSTANCES_PATH = Path("dataset/trace_completed_instance.toml")
OUTPUT_NAME = "BM25_variable_cutoff.txt"
GRANULARITIES = ("files", "classes", "functions")
CHANGE_TYPES = ("added", "edited", "total")
TOKEN_RE = re.compile(r"(?u)\w+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
K1 = 1.5
B = 0.75

EntityMap = dict[str, list[str]]


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
class InstanceEvaluation:
    instance_id: str
    predictions: dict[str, set[str]]
    ground_truth: dict[str, dict[str, set[str]]]
    is_failure: bool
    failure_reason: str | None = None
    cutoff_lengths: dict[str, int] | None = None


@dataclass
class MetricCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class MacroMetricTotals:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    instance_count: int = 0


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


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


def parse_args() -> argparse.Namespace:
    root = get_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent_results",
        type=Path,
        default=root / DEFAULT_AGENT_RESULTS,
        help="Agent result folder whose successful instances and result lengths are used.",
    )
    return parser.parse_args()


def normalize_agent_results_path(path: Path, root: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def infer_workbench_path(agent_results: Path, root: Path) -> Path:
    parts = agent_results.parts
    for index, part in enumerate(parts):
        if part == "agent_results" and index + 1 < len(parts):
            return (root / "dataset" / "agent_workbench" / parts[index + 1]).resolve()
    return (root / "dataset" / "agent_workbench" / "dalic").resolve()


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


def find_instance_repository_root(workbench_dir: Path, instance_id: str) -> Path:
    repository_parent = workbench_dir / instance_id / "repo"
    if not repository_parent.is_dir():
        raise ValueError(f"Repository parent directory is missing: {repository_parent}")

    repository_roots = sorted(
        path for path in repository_parent.iterdir() if path.is_dir()
    )
    if len(repository_roots) != 1:
        raise ValueError(
            f"Expected exactly one repository root under {repository_parent}, "
            f"found {len(repository_roots)}"
        )
    return repository_roots[0]


def validate_repo_relative_path(path_text: str, repository_root: Path) -> str | None:
    if not path_text:
        return "path is empty"
    if path_text.startswith("/"):
        return "path must be repo-relative, not absolute"
    if path_text.startswith("repo/"):
        return "path must be repo-relative and must not start with 'repo/'"
    if "\\" in path_text:
        return "path must use forward slashes"

    path_parts = PurePosixPath(path_text).parts
    if any(part in (".", "..") for part in path_parts):
        return "path must not contain '.' or '..' segments"

    candidate_path = repository_root.joinpath(*path_parts)
    if (
        path_parts
        and path_parts[0] == repository_root.name
        and not candidate_path.is_file()
    ):
        stripped_path = repository_root.joinpath(*path_parts[1:])
        if stripped_path.is_file():
            return (
                f"path contains redundant checkout directory prefix "
                f"'{repository_root.name}/'; use "
                f"'{PurePosixPath(*path_parts[1:])}'"
            )
    return None


def parse_results_file(
    results_path: Path, repository_root: Path
) -> tuple[dict[str, set[str]] | None, str | None]:
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "results.txt is missing"
    except json.JSONDecodeError as exc:
        return None, f"results.txt is not valid JSON: {exc.msg}"

    if not isinstance(payload, dict):
        return None, "top-level JSON value must be an object"
    if set(payload) != {"results"}:
        return None, "top-level JSON object must contain exactly one key: 'results'"

    results = payload["results"]
    if not isinstance(results, dict):
        return None, "'results' must be a JSON object"
    if set(results) != set(GRANULARITIES):
        return None, "'results' must contain exactly these keys: files, classes, functions"

    parsed: dict[str, set[str]] = {}
    for field_name in GRANULARITIES:
        entries = results[field_name]
        if not isinstance(entries, list):
            return None, f"'{field_name}' must be a list"
        if any(not isinstance(entry, str) for entry in entries):
            return None, f"'{field_name}' must contain only strings"
        if len(entries) != len(set(entries)):
            return None, f"'{field_name}' contains duplicate entries"
        parsed[field_name] = set(entries)

    for file_path in parsed["files"]:
        path_error = validate_repo_relative_path(file_path, repository_root)
        if path_error:
            return None, f"invalid file entry '{file_path}': {path_error}"

    for field_name in ("classes", "functions"):
        for entry in parsed[field_name]:
            if entry.count("::") != 1:
                return (
                    None,
                    f"invalid {field_name[:-1]} entry '{entry}': "
                    "expected '<file_path>::<qualified_name>'",
                )
            file_path, qualified_name = entry.split("::", 1)
            path_error = validate_repo_relative_path(file_path, repository_root)
            if path_error:
                return None, f"invalid {field_name[:-1]} entry '{entry}': {path_error}"
            if not qualified_name:
                return (
                    None,
                    f"invalid {field_name[:-1]} entry '{entry}': qualified name is empty",
                )

    return parsed, None


def normalize_ground_truth(raw_ground_truth: dict) -> dict[str, dict[str, set[str]]]:
    normalized: dict[str, dict[str, set[str]]] = {
        "added": {granularity: set() for granularity in GRANULARITIES},
        "edited": {granularity: set() for granularity in GRANULARITIES},
        "total": {granularity: set() for granularity in GRANULARITIES},
    }

    for change_type in ("added", "edited"):
        change_payload = raw_ground_truth.get(change_type, {})
        if not isinstance(change_payload, dict):
            continue
        for granularity in GRANULARITIES:
            values = change_payload.get(granularity, [])
            if isinstance(values, list):
                normalized[change_type][granularity] = set(str(value) for value in values)

    for granularity in GRANULARITIES:
        normalized["total"][granularity] = (
            normalized["added"][granularity] | normalized["edited"][granularity]
        )

    return normalized


def load_ground_truth_map(
    truth_path: Path,
) -> dict[str, dict[str, dict[str, set[str]]]]:
    ground_truth_map: dict[str, dict[str, dict[str, set[str]]]] = {}
    with truth_path.open("r", encoding="utf-8") as truth_file:
        for line_number, raw_line in enumerate(truth_file, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if row.get("ground_truth_certainty") != "sure":
                continue
            instance_id = row.get("instance_id")
            if not instance_id:
                raise ValueError(
                    f"Missing instance_id in {truth_path} at line {line_number}."
                )
            ground_truth = row.get("ground_truth")
            if not isinstance(ground_truth, dict):
                raise ValueError(
                    f"Missing or invalid ground_truth for instance {instance_id} "
                    f"in {truth_path} at line {line_number}."
                )
            ground_truth_map[str(instance_id)] = normalize_ground_truth(ground_truth)
    return ground_truth_map


def load_selected_instance_ids(selected_instances_path: Path) -> set[str]:
    with selected_instances_path.open("rb") as selected_instances_file:
        payload = tomllib.load(selected_instances_file)
    return set(payload["instance_ids"])


def filter_selected_ground_truth(
    ground_truth_map: dict[str, dict[str, dict[str, set[str]]]],
    selected_instance_ids: set[str],
) -> dict[str, dict[str, dict[str, set[str]]]]:
    return {
        instance_id: ground_truth
        for instance_id, ground_truth in ground_truth_map.items()
        if instance_id in selected_instance_ids
    }


def iter_instance_dirs(result_folder: Path) -> list[Path]:
    return sorted(path for path in result_folder.iterdir() if path.is_dir())


def load_instance_evaluations(
    result_folder: Path,
    ground_truth_map: dict[str, dict[str, dict[str, set[str]]]],
    workbench_dir: Path,
) -> tuple[list[InstanceEvaluation], list[str]]:
    evaluations: list[InstanceEvaluation] = []
    result_instance_dirs = iter_instance_dirs(result_folder)
    skipped_instance_ids = [
        instance_dir.name
        for instance_dir in result_instance_dirs
        if instance_dir.name not in ground_truth_map
    ]

    for instance_id, ground_truth in sorted(ground_truth_map.items()):
        instance_dir = result_folder / instance_id
        if not instance_dir.is_dir():
            evaluations.append(
                InstanceEvaluation(
                    instance_id=instance_id,
                    predictions={granularity: set() for granularity in GRANULARITIES},
                    ground_truth=ground_truth,
                    is_failure=True,
                    failure_reason="instance result directory is missing",
                )
            )
            continue

        repository_root = find_instance_repository_root(workbench_dir, instance_id)
        predictions, failure_reason = parse_results_file(
            instance_dir / "results.txt", repository_root
        )
        if predictions is None:
            predictions = {granularity: set() for granularity in GRANULARITIES}
            evaluations.append(
                InstanceEvaluation(
                    instance_id=instance_id,
                    predictions=predictions,
                    ground_truth=ground_truth,
                    is_failure=True,
                    failure_reason=failure_reason,
                )
            )
            continue

        evaluations.append(
            InstanceEvaluation(
                instance_id=instance_id,
                predictions=predictions,
                ground_truth=ground_truth,
                is_failure=False,
            )
        )

    return evaluations, skipped_instance_ids


def result_lengths(evaluation: InstanceEvaluation) -> dict[str, int]:
    return {
        granularity: len(evaluation.predictions[granularity])
        for granularity in GRANULARITIES
    }


def variable_cutoff_bm25_predictions(
    agent_evaluation: InstanceEvaluation,
    workbench_dir: Path,
) -> dict[str, set[str]]:
    candidates = direct_bm25_candidates(workbench_dir / agent_evaluation.instance_id)
    lengths = result_lengths(agent_evaluation)
    return {
        granularity: set(candidates[granularity][: lengths[granularity]])
        for granularity in GRANULARITIES
    }


def build_counts(
    evaluations: list[InstanceEvaluation],
    *,
    ignore_failures: bool,
) -> dict[str, dict[str, MetricCounts]]:
    counts = {
        change_type: {granularity: MetricCounts() for granularity in GRANULARITIES}
        for change_type in CHANGE_TYPES
    }

    for evaluation in evaluations:
        if ignore_failures and evaluation.is_failure:
            continue

        for change_type in CHANGE_TYPES:
            for granularity in GRANULARITIES:
                predicted = evaluation.predictions[granularity]
                truth = evaluation.ground_truth[change_type][granularity]
                count = counts[change_type][granularity]
                count.tp += len(predicted & truth)
                count.fp += len(predicted - truth)
                count.fn += len(truth - predicted)

    return counts


def precision_recall_f1(counts: MetricCounts) -> tuple[float, float, float]:
    precision = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def instance_precision_recall_f1(
    predicted: set[str],
    truth: set[str],
) -> tuple[float, float, float]:
    if not predicted and not truth:
        return 1.0, 1.0, 1.0

    counts = MetricCounts(
        tp=len(predicted & truth),
        fp=len(predicted - truth),
        fn=len(truth - predicted),
    )
    return precision_recall_f1(counts)


def build_macro_metrics(
    evaluations: list[InstanceEvaluation],
    *,
    ignore_failures: bool,
) -> dict[str, dict[str, MacroMetricTotals]]:
    totals = {
        change_type: {granularity: MacroMetricTotals() for granularity in GRANULARITIES}
        for change_type in CHANGE_TYPES
    }

    for evaluation in evaluations:
        if ignore_failures and evaluation.is_failure:
            continue

        for change_type in CHANGE_TYPES:
            for granularity in GRANULARITIES:
                precision, recall, f1 = instance_precision_recall_f1(
                    evaluation.predictions[granularity],
                    evaluation.ground_truth[change_type][granularity],
                )
                metric = totals[change_type][granularity]
                metric.precision += precision
                metric.recall += recall
                metric.f1 += f1
                metric.instance_count += 1

    return totals


def format_metric_line(granularity: str, counts: MetricCounts) -> str:
    precision, recall, f1 = precision_recall_f1(counts)
    return (
        f"{granularity}: "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"(tp={counts.tp} fp={counts.fp} fn={counts.fn})"
    )


def format_macro_metric_line(granularity: str, totals: MacroMetricTotals) -> str:
    instance_count = totals.instance_count
    precision = totals.precision / instance_count if instance_count else 0.0
    recall = totals.recall / instance_count if instance_count else 0.0
    f1 = totals.f1 / instance_count if instance_count else 0.0
    return (
        f"{granularity}: "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"(instances={instance_count})"
    )


def instance_metric_counts(
    evaluation: InstanceEvaluation,
    change_type: str,
    granularity: str,
) -> MetricCounts:
    predicted = evaluation.predictions[granularity]
    truth = evaluation.ground_truth[change_type][granularity]
    return MetricCounts(
        tp=len(predicted & truth),
        fp=len(predicted - truth),
        fn=len(truth - predicted),
    )


def format_instance_lengths(evaluation: InstanceEvaluation) -> str:
    lengths = evaluation.cutoff_lengths or result_lengths(evaluation)
    return "lengths: " + " ".join(
        f"{granularity}={lengths[granularity]}" for granularity in GRANULARITIES
    )


def append_per_instance_results(
    sections: list[str],
    evaluations: list[InstanceEvaluation],
) -> None:
    successful_evaluations = [
        evaluation for evaluation in evaluations if not evaluation.is_failure
    ]
    if not successful_evaluations:
        return

    sections.append("per_instance_results:")
    for evaluation in successful_evaluations:
        sections.append(f"{evaluation.instance_id}:")
        sections.append(f"  {format_instance_lengths(evaluation)}")
        for change_type in CHANGE_TYPES:
            sections.append(f"  {change_type}:")
            for granularity in GRANULARITIES:
                counts = instance_metric_counts(evaluation, change_type, granularity)
                sections.append(f"    {format_metric_line(granularity, counts)}")
        sections.append("")


def render_evaluation_text(
    result_folder: Path,
    evaluations: list[InstanceEvaluation],
    skipped_instance_ids: list[str],
) -> str:
    failed_instance_ids = [
        evaluation.instance_id for evaluation in evaluations if evaluation.is_failure
    ]
    successful_instance_ids = [
        evaluation.instance_id for evaluation in evaluations if not evaluation.is_failure
    ]
    missing_result_instance_ids = [
        evaluation.instance_id
        for evaluation in evaluations
        if evaluation.failure_reason == "instance result directory is missing"
    ]

    sections: list[str] = [
        f"result_folder: {result_folder}",
        f"sure_ground_truth_instances: {len(evaluations)}",
        f"total_instance_dirs: {len(evaluations) - len(missing_result_instance_ids) + len(skipped_instance_ids)}",
        f"matched_instances: {len(evaluations) - len(missing_result_instance_ids)}",
        f"missing_result_instance_dirs: {len(missing_result_instance_ids)}",
        f"successful_instances: {len(successful_instance_ids)}",
        f"failed_instances: {len(failed_instance_ids)}",
        f"skipped_instances_without_ground_truth: {len(skipped_instance_ids)}",
        "",
    ]

    if failed_instance_ids:
        sections.append("failed_instance_reasons:")
        for evaluation in evaluations:
            if evaluation.is_failure:
                sections.append(f"{evaluation.instance_id}: {evaluation.failure_reason}")
        sections.append("")

    if skipped_instance_ids:
        sections.append("skipped_instances_without_ground_truth:")
        sections.extend(skipped_instance_ids)
        sections.append("")

    for ignore_failures in (True, False):
        mode_name = (
            "ignore_failed_instances" if ignore_failures else "include_failed_instances"
        )
        included_count = sum(
            1
            for evaluation in evaluations
            if not (ignore_failures and evaluation.is_failure)
        )
        sections.append(f"[{mode_name}]")
        sections.append(f"included_instances: {included_count}")
        sections.append("")

        sections.append("micro:")
        counts = build_counts(evaluations, ignore_failures=ignore_failures)
        for change_type in CHANGE_TYPES:
            sections.append(f"  {change_type}:")
            for granularity in GRANULARITIES:
                sections.append(
                    f"    {format_metric_line(granularity, counts[change_type][granularity])}"
                )
            sections.append("")

        sections.append("macro:")
        macro_metrics = build_macro_metrics(
            evaluations,
            ignore_failures=ignore_failures,
        )
        for change_type in CHANGE_TYPES:
            sections.append(f"  {change_type}:")
            for granularity in GRANULARITIES:
                sections.append(
                    f"    {format_macro_metric_line(granularity, macro_metrics[change_type][granularity])}"
                )
            sections.append("")

    append_per_instance_results(sections, evaluations)
    return "\n".join(sections).rstrip() + "\n"


def main() -> int:
    root = get_repo_root().resolve()
    args = parse_args()
    agent_results = normalize_agent_results_path(args.agent_results, root)
    if not agent_results.exists():
        raise ValueError(f"Agent results folder does not exist: {agent_results}")
    if not agent_results.is_dir():
        raise ValueError(f"Agent results path is not a directory: {agent_results}")

    truth_path = (root / DEFAULT_INPUT_PATH).resolve()
    selected_instances_path = (root / DEFAULT_SELECTED_INSTANCES_PATH).resolve()
    workbench_dir = infer_workbench_path(agent_results, root)
    if not workbench_dir.exists():
        raise ValueError(f"Workbench path does not exist: {workbench_dir}")
    if not workbench_dir.is_dir():
        raise ValueError(f"Workbench path is not a directory: {workbench_dir}")

    ground_truth_map = load_ground_truth_map(truth_path)
    selected_instance_ids = load_selected_instance_ids(selected_instances_path)
    ground_truth_map = filter_selected_ground_truth(
        ground_truth_map,
        selected_instance_ids,
    )
    agent_evaluations, _skipped_instance_ids = load_instance_evaluations(
        agent_results,
        ground_truth_map,
        workbench_dir,
    )
    successful_agent_evaluations = [
        evaluation for evaluation in agent_evaluations if not evaluation.is_failure
    ]

    print(f"K1={K1}, B={B}")
    print(f"agent_results={agent_results}")
    print(f"workbench={workbench_dir}")
    print(f"successful_instances={len(successful_agent_evaluations)}")

    bm25_evaluations: list[InstanceEvaluation] = []
    progress = tqdm(
        successful_agent_evaluations,
        desc="bm25-variable-cutoff",
        unit="instance",
    )
    for agent_evaluation in progress:
        cutoff_lengths = result_lengths(agent_evaluation)
        predictions = variable_cutoff_bm25_predictions(agent_evaluation, workbench_dir)
        bm25_evaluations.append(
            InstanceEvaluation(
                instance_id=agent_evaluation.instance_id,
                predictions=predictions,
                ground_truth=agent_evaluation.ground_truth,
                is_failure=False,
                cutoff_lengths=cutoff_lengths,
            )
        )
        counts = MetricCounts()
        for granularity in GRANULARITIES:
            predicted = predictions[granularity]
            truth = agent_evaluation.ground_truth["total"][granularity]
            counts.tp += len(predicted & truth)
            counts.fp += len(predicted - truth)
            counts.fn += len(truth - predicted)
        precision, recall, f1 = precision_recall_f1(counts)
        progress.set_postfix_str(
            f"{agent_evaluation.instance_id}: "
            f"P/R/F1@full={precision:.2%}/{recall:.2%}/{f1:.2%}"
        )

    output_path = agent_results / OUTPUT_NAME
    output_path.write_text(
        render_evaluation_text(agent_results, bm25_evaluations, []),
        encoding="utf-8",
    )
    print(f"Wrote summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
