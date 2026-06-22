#!/usr/bin/env python3
"""Create patched agent workbenches with BM25-ranked dynamic evidence."""

from __future__ import annotations

import argparse
import ast
import json
import mmap
import re
import shutil
import tokenize
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
from tqdm import tqdm

from prepare_agent_workbench import (
    InstanceRecord,
    clean_repo_artifacts,
    create_repo_snapshot,
    load_sure_instances,
    repo_name_from_instance_id,
)


CALL_MARKER = b'"type": "CALL"'
LINE_MARKER = b'"type": "LINE"'
FILE_FIELD = b'"file_path"'
FUNC_FIELD = b'"func"'
LINE_FIELD = b'"line_no"'
DAUGHTER_FIELD = b'"daughter_stack"'
TRACE_WINDOW = 16384
TOKEN_RE = re.compile(r"(?u)\w+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


@dataclass(frozen=True)
class Scope:
    kind: str
    qualified_name: str
    start_line: int
    end_line: int
    class_names: tuple[str, ...]


@dataclass
class SourceIndex:
    text: str
    lines: list[str]
    classes: list[Scope]
    functions: list[Scope]

    def source_for(self, scope: Scope) -> str:
        return "".join(self.lines[scope.start_line - 1 : scope.end_line])


@dataclass
class Evidence:
    tests: dict[str, set[str]]
    scopes: dict[str, Scope]


def repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-completed-instances",
        "--selected-instances",
        dest="trace_completed_instances",
        type=Path,
        default=root / "dataset/trace_completed_instance.toml",
        help=(
            "TOML file containing the instance_ids to process. "
            "Default: dataset/trace_completed_instance.toml"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "dataset/swebench_feature_with_truth.jsonl",
    )
    parser.add_argument(
        "--repo-cache-dir",
        type=Path,
        default=root / "dataset/agent_workbench/dalic/.repo_mirrors",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=root / "dataset/vartracer_results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "dataset/agent_workbench/dalic",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        help="Only create this selected instance; may be supplied more than once.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test-suite Python entities in candidate lists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output instance directories.",
    )
    args = parser.parse_args()
    for name in (
        "trace_completed_instances",
        "input",
        "repo_cache_dir",
        "trace_dir",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def load_instance_ids(path: Path, requested: list[str] | None) -> list[str]:
    with path.open("rb") as handle:
        instance_ids = tomllib.load(handle).get("instance_ids")
    if not isinstance(instance_ids, list) or not all(isinstance(value, str) for value in instance_ids):
        raise ValueError(f"{path} must contain a string array named instance_ids")
    if requested:
        requested_set = set(requested)
        missing = requested_set - set(instance_ids)
        if missing:
            raise ValueError(f"Instances are not trace-completed: {', '.join(sorted(missing))}")
        instance_ids = [value for value in instance_ids if value in requested_set]
    return instance_ids


def load_instances(input_path: Path, instance_ids: list[str]) -> list[InstanceRecord]:
    by_id = {
        instance.instance_id: instance
        for instance in load_sure_instances(input_path)
    }
    missing = set(instance_ids) - by_id.keys()
    if missing:
        raise ValueError(f"Input dataset is missing: {', '.join(sorted(missing))}")
    return [by_id[instance_id] for instance_id in instance_ids]


def find_checkout(instance_dir: Path) -> Path:
    roots = sorted(path for path in (instance_dir / "repo").iterdir() if path.is_dir())
    if len(roots) != 1:
        raise ValueError(f"Expected one checkout under {instance_dir / 'repo'}, found {len(roots)}")
    return roots[0]


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = parts[-1].lower() if parts else ""
    return "tests" in {part.lower() for part in parts} or name.startswith("test_") or name.endswith("_test.py")


def project_path(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.replace("\\", "/")
    marker = "/testbed/"
    if marker not in normalized:
        return None
    relative = normalized.split(marker, 1)[1].lstrip("/")
    return relative if relative.endswith(".py") else None


def read_json_string_field(
    mapped: mmap.mmap, marker: bytes, start: int, end: int
) -> str | None:
    field = mapped.find(marker, start, end)
    if field < 0:
        return None
    colon = mapped.find(b":", field + len(marker), end)
    quote = mapped.find(b'"', colon + 1, end) if colon >= 0 else -1
    if quote < 0:
        return None
    position = quote + 1
    while position < end:
        byte = mapped[position]
        if byte == 92:
            position += 2
            continue
        if byte == 34:
            try:
                return json.loads(mapped[quote : position + 1].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
        position += 1
    return None


class ScopeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []
        self.class_names: list[str] = []
        self.classes: list[Scope] = []
        self.functions: list[Scope] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.append(node.name)
        qualified = ".".join(self.names)
        self.class_names.append(qualified)
        self.classes.append(self.make_scope("class", qualified, node))
        self.generic_visit(node)
        self.class_names.pop()
        self.names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_function(node)

    def visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.names.append(node.name)
        self.functions.append(self.make_scope("function", ".".join(self.names), node))
        self.generic_visit(node)
        self.names.pop()

    def make_scope(self, kind: str, qualified: str, node: ast.AST) -> Scope:
        return Scope(
            kind=kind,
            qualified_name=qualified,
            start_line=int(node.lineno),
            end_line=int(getattr(node, "end_lineno", node.lineno)),
            class_names=tuple(self.class_names),
        )


def load_source_index(path: Path) -> SourceIndex | None:
    try:
        with tokenize.open(path) as handle:
            text = handle.read()
        module = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        location = f":{exc.lineno}" if exc.lineno is not None else ""
        tqdm.write(
            f"[warning] AST parse failed for {path}{location}: "
            f"{exc.msg or type(exc).__name__}"
        )
        return None
    except (OSError, UnicodeError) as exc:
        tqdm.write(f"[warning] source read failed for {path}: {exc}")
        return None
    visitor = ScopeVisitor()
    visitor.visit(module)
    return SourceIndex(
        text=text,
        lines=text.splitlines(keepends=True),
        classes=visitor.classes,
        functions=visitor.functions,
    )


def innermost_scope(scopes: list[Scope], line: int, leaf_name: str | None = None) -> Scope | None:
    matches = [scope for scope in scopes if scope.start_line <= line <= scope.end_line]
    named = [scope for scope in matches if scope.qualified_name.rsplit(".", 1)[-1] == leaf_name]
    if named:
        matches = named
    if not matches:
        return None
    return min(matches, key=lambda scope: (scope.end_line - scope.start_line, -scope.start_line))


def trace_paths(instance_trace_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for test_dir in sorted(instance_trace_dir.glob("tc_*")):
        for filename in ("exec_trace.json", "exec_stack.json"):
            candidate = test_dir / filename
            if candidate.is_file() and candidate.stat().st_size:
                paths.append(candidate)
                break
    return paths


def test_name(trace_path: Path) -> str:
    metadata_path = trace_path.parent / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return trace_path.parent.name
    return str(metadata.get("test_case_name") or metadata.get("pytest_nodeid") or trace_path.parent.name)


def add_frame(
    *,
    checkout: Path,
    source_path: str | None,
    func_name: str | None,
    line_text: str | None,
    case_name: str,
    include_tests: bool,
    source_cache: dict[str, SourceIndex | None],
    files: Evidence,
    classes: Evidence,
    functions: Evidence,
) -> None:
    if source_path is None or (not include_tests and is_test_path(source_path)):
        return
    source_file = checkout / source_path
    if not source_file.is_file():
        return
    files.tests.setdefault(source_path, set()).add(case_name)
    if source_path not in source_cache:
        source_cache[source_path] = load_source_index(source_file)
    index = source_cache[source_path]
    if index is None or line_text is None:
        return
    try:
        line = int(line_text)
    except ValueError:
        return

    function = innermost_scope(index.functions, line, func_name)
    if function is not None:
        function_id = f"{source_path}::{function.qualified_name}"
        functions.tests.setdefault(function_id, set()).add(case_name)
        functions.scopes[function_id] = function
        class_names = function.class_names
    else:
        owner = innermost_scope(index.classes, line)
        class_names = owner.class_names if owner is not None else ()

    class_by_name = {scope.qualified_name: scope for scope in index.classes}
    for class_name in class_names:
        class_id = f"{source_path}::{class_name}"
        classes.tests.setdefault(class_id, set()).add(case_name)
        classes.scopes[class_id] = class_by_name[class_name]


def scan_trace(
    trace_path: Path,
    checkout: Path,
    include_tests: bool,
    source_cache: dict[str, SourceIndex | None],
    files: Evidence,
    classes: Evidence,
    functions: Evidence,
) -> None:
    case_name = test_name(trace_path)
    size = trace_path.stat().st_size
    with trace_path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        root_line = mapped.find(LINE_MARKER)
        if root_line >= 0:
            end = min(size, root_line + TRACE_WINDOW)
            add_frame(
                checkout=checkout,
                source_path=project_path(read_json_string_field(mapped, FILE_FIELD, root_line, end)),
                func_name=read_json_string_field(mapped, FUNC_FIELD, root_line, end),
                line_text=read_json_string_field(mapped, LINE_FIELD, root_line, end),
                case_name=case_name,
                include_tests=include_tests,
                source_cache=source_cache,
                files=files,
                classes=classes,
                functions=functions,
            )

        position = 0
        while True:
            call_at = mapped.find(CALL_MARKER, position)
            if call_at < 0:
                break
            end = min(size, call_at + TRACE_WINDOW)
            daughter = mapped.find(DAUGHTER_FIELD, call_at, end)
            line_at = mapped.find(LINE_MARKER, daughter, end) if daughter >= 0 else -1
            source_path = project_path(
                read_json_string_field(mapped, FILE_FIELD, call_at, end)
            )
            func_name = read_json_string_field(mapped, FUNC_FIELD, call_at, end)
            line_text = None
            if line_at >= 0:
                line_source = project_path(
                    read_json_string_field(mapped, FILE_FIELD, line_at, end)
                )
                line_func = read_json_string_field(mapped, FUNC_FIELD, line_at, end)
                if line_source == source_path and line_func == func_name:
                    line_text = read_json_string_field(mapped, LINE_FIELD, line_at, end)
            add_frame(
                checkout=checkout,
                source_path=source_path,
                func_name=func_name,
                line_text=line_text,
                case_name=case_name,
                include_tests=include_tests,
                source_cache=source_cache,
                files=files,
                classes=classes,
                functions=functions,
            )
            position = call_at + len(CALL_MARKER)


def tokens(text: str) -> list[str]:
    result: list[str] = []
    for raw in TOKEN_RE.findall(text):
        expanded: list[str] = [raw.casefold()]
        for snake_part in raw.split("_"):
            for part in CAMEL_RE.split(snake_part):
                normalized = part.casefold()
                if normalized and normalized not in expanded:
                    expanded.append(normalized)
        result.extend(expanded)
    return result


def bm25_scores(
    query: str,
    documents: list[str],
    background_documents: list[str] | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not documents:
        return []
    document_tokens = [tokens(document) for document in documents]
    if background_documents:
        document_tokens.extend(tokens(document) for document in background_documents)
    query_tokens = list(dict.fromkeys(tokens(query)))
    if not query_tokens:
        return [0.0] * len(documents)
    retriever = bm25s.BM25(method="lucene", k1=k1, b=b)
    retriever.index(document_tokens, show_progress=False)
    return [float(score) for score in retriever.get_scores(query_tokens)[: len(documents)]]


def ranked_records(
    evidence: Evidence,
    problem_statement: str,
    document_for: Any,
    background_documents: list[str],
    total_tests: int,
    extra_for: Any,
) -> list[dict[str, Any]]:
    names = sorted(evidence.tests)
    scores = bm25_scores(
        problem_statement,
        [f"{name} {document_for(name)}" for name in names],
        background_documents,
    )
    records = []
    for name, score in zip(names, scores):
        cases = sorted(evidence.tests[name])
        records.append(
            {
                "qualified_name": name,
                "_sort_score": score,
                "bm25_score": round(score, 8),
                "test_case_count": len(cases),
                "total_test_case_count": total_tests,
                "test_cases": cases,
                **extra_for(name),
            }
        )
    records.sort(
        key=lambda record: (-record["_sort_score"], record["qualified_name"])
    )
    for record in records:
        del record["_sort_score"]
    return records


def write_dynamic_evidence(
    instance_trace_dir: Path,
    instance_dir: Path,
    include_tests: bool,
) -> None:
    checkout = find_checkout(instance_dir)
    problem_statement = (instance_dir / "problem_statement.txt").read_text(encoding="utf-8")
    files = Evidence(defaultdict(set), {})
    classes = Evidence(defaultdict(set), {})
    functions = Evidence(defaultdict(set), {})
    source_cache: dict[str, SourceIndex | None] = {}
    paths = trace_paths(instance_trace_dir)
    if not paths:
        raise FileNotFoundError(f"No exec trace found under {instance_trace_dir}")
    for path in paths:
        scan_trace(
            path,
            checkout,
            include_tests,
            source_cache,
            files,
            classes,
            functions,
        )

    class_functions: dict[str, list[str]] = defaultdict(list)
    file_classes: dict[str, list[str]] = defaultdict(list)
    file_functions: dict[str, list[str]] = defaultdict(list)
    function_classes: dict[str, list[str]] = {}
    for class_id in classes.tests:
        file_classes[class_id.split("::", 1)[0]].append(class_id)
    for function_id, scope in functions.scopes.items():
        file_path = function_id.split("::", 1)[0]
        file_functions[file_path].append(function_id)
        owners = [f"{file_path}::{name}" for name in scope.class_names if f"{file_path}::{name}" in classes.tests]
        function_classes[function_id] = owners
        for owner in owners:
            class_functions[owner].append(function_id)

    def source_text(path: str) -> str:
        index = source_cache.get(path)
        return index.text if index is not None else ""

    def scope_text(name: str, evidence: Evidence) -> str:
        path = name.split("::", 1)[0]
        index = source_cache.get(path)
        scope = evidence.scopes.get(name)
        return index.source_for(scope) if index is not None and scope is not None else ""

    repo_documents: dict[str, dict[str, str]] = {
        "files": {},
        "classes": {},
        "functions": {},
    }
    for source_file in sorted(checkout.rglob("*.py")):
        relative = source_file.relative_to(checkout).as_posix()
        if not source_file.is_file() or (not include_tests and is_test_path(relative)):
            continue
        if relative not in source_cache:
            source_cache[relative] = load_source_index(source_file)
        index = source_cache[relative]
        if index is None:
            continue
        repo_documents["files"][relative] = f"{relative} {index.text}"
        for scope in index.classes:
            name = f"{relative}::{scope.qualified_name}"
            repo_documents["classes"][name] = f"{name} {index.source_for(scope)}"
        for scope in index.functions:
            name = f"{relative}::{scope.qualified_name}"
            repo_documents["functions"][name] = f"{name} {index.source_for(scope)}"

    def background(kind: str, evidence: Evidence) -> list[str]:
        return [
            document
            for name, document in repo_documents[kind].items()
            if name not in evidence.tests
        ]

    total_tests = len(paths)
    file_records = ranked_records(
        files,
        problem_statement,
        source_text,
        background("files", files),
        total_tests,
        lambda name: {
            "classes": sorted(file_classes[name]),
            "functions": sorted(file_functions[name]),
        },
    )
    class_records = ranked_records(
        classes,
        problem_statement,
        lambda name: scope_text(name, classes),
        background("classes", classes),
        total_tests,
        lambda name: {
            "file": name.split("::", 1)[0],
            "functions": sorted(class_functions[name]),
        },
    )
    function_records = ranked_records(
        functions,
        problem_statement,
        lambda name: scope_text(name, functions),
        background("functions", functions),
        total_tests,
        lambda name: {
            "file": name.split("::", 1)[0],
            "classes": function_classes.get(name, []),
        },
    )

    class_rank = {
        record["qualified_name"]: rank for rank, record in enumerate(class_records)
    }
    function_rank = {
        record["qualified_name"]: rank for rank, record in enumerate(function_records)
    }
    for record in file_records:
        record["classes"].sort(key=class_rank.__getitem__)
        record["functions"].sort(key=function_rank.__getitem__)
    for record in class_records:
        record["functions"].sort(key=function_rank.__getitem__)
    for record in function_records:
        record["classes"].sort(key=class_rank.__getitem__)

    output_dir = instance_dir / "dynamic_evidence"
    output_dir.mkdir()
    for filename, records in (
        ("file.json", file_records),
        ("class.json", class_records),
        ("function.json", function_records),
    ):
        (output_dir / filename).write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def create_instance(
    instance: InstanceRecord,
    repo_cache_dir: Path,
    trace_dir: Path,
    output_dir: Path,
    include_tests: bool,
    overwrite: bool,
) -> str:
    instance_id = instance.instance_id
    destination = output_dir / instance_id
    if destination.exists():
        if not overwrite:
            return "skipped"
        shutil.rmtree(destination)

    temporary = output_dir / f".{instance_id}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        checkout = temporary / "repo" / repo_name_from_instance_id(instance_id)
        checkout.parent.mkdir(parents=True)
        create_repo_snapshot(instance, checkout, repo_cache_dir)
        clean_repo_artifacts(checkout)
        (temporary / "problem_statement.txt").write_text(
            instance.problem_statement,
            encoding="utf-8",
        )
        write_dynamic_evidence(trace_dir / instance_id, temporary, include_tests)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "created"


def main() -> int:
    args = parse_args()
    instance_ids = load_instance_ids(args.trace_completed_instances, args.instance_id)
    instances = load_instances(args.input, instance_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0
    failures: list[tuple[str, str]] = []
    for instance in tqdm(instances, desc="Creating candidate lists", unit="instance"):
        instance_id = instance.instance_id
        try:
            status = create_instance(
                instance,
                args.repo_cache_dir,
                args.trace_dir,
                args.output_dir,
                args.include_tests,
                args.overwrite,
            )
        except Exception as exc:
            failures.append((instance_id, str(exc)))
            tqdm.write(f"[failed] {instance_id}: {exc}")
            continue
        if status == "created":
            created += 1
        else:
            skipped += 1
    print(f"Created {created}; skipped {skipped}; failed {len(failures)}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
