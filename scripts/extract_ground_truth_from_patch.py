#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError


PYTHON_LIKE_SUFFIXES = {".py", ".pyi", ".pyx"}
DOWNLOAD_TIMEOUT_SECONDS = 30
DEF_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:async\s+def|def)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)
CLASS_RE = re.compile(r"^(?P<indent>[ \t]*)class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()

    parser = argparse.ArgumentParser(
        description="Extract added/edited module/class/function ground truth from patch fields."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "dataset" / "swebench_feature.jsonl",
        help="Input jsonl file. Default: dataset/swebench_feature.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "dataset" / "swebench_feature_with_truth.jsonl",
        help="Output jsonl file. Default: dataset/swebench_feature_with_truth.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / "dataset" / ".gt_source_cache",
        help="Directory for caching downloaded base files.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()

    if not args.input.exists():
        raise ValueError(f"Input file does not exist: {args.input}")
    if args.input.suffix != ".jsonl":
        raise ValueError("--input must be a .jsonl file.")
    if args.output.suffix != ".jsonl":
        raise ValueError("--output must be a .jsonl file.")
    if args.input == args.output:
        raise ValueError("--input and --output must be different files.")


def indent_width(text: str) -> int:
    return len(text.expandtabs(4))


def parse_symbol(line: str) -> tuple[str | None, str | None, int | None]:
    def_match = DEF_RE.match(line)
    if def_match:
        return "def", def_match.group("name"), indent_width(def_match.group("indent"))

    class_match = CLASS_RE.match(line)
    if class_match:
        return "class", class_match.group("name"), indent_width(class_match.group("indent"))

    return None, None, None


def normalize_patch_path(raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    if raw_path == "/dev/null":
        return None
    if raw_path.startswith(("a/", "b/")):
        return raw_path[2:]
    return raw_path


def parse_patch_files(
    patch_text: str,
    certainty_state: dict[str, set[str]],
) -> list[dict[str, Any]]:
    if not patch_text.strip():
        return []

    try:
        patch_set = PatchSet(patch_text.splitlines(True))
    except UnidiffParseError:
        mark_not_sure(certainty_state, "unsupported_patch_format")
        return []

    parsed_files: list[dict[str, Any]] = []
    for patched_file in patch_set:
        file_path = (
            normalize_patch_path(patched_file.target_file)
            or normalize_patch_path(patched_file.source_file)
            or normalize_patch_path(getattr(patched_file, "path", None))
        )
        if file_path is None:
            mark_not_sure(certainty_state, "unsupported_patch_format")
            continue

        base_lookup_path = None
        if not patched_file.is_added_file:
            base_lookup_path = normalize_patch_path(patched_file.source_file) or file_path

        parsed_files.append(
            {
                "file_path": file_path,
                "base_lookup_path": base_lookup_path,
                "patched_file": patched_file,
            }
        )

    if not parsed_files:
        mark_not_sure(certainty_state, "unsupported_patch_format")
    return parsed_files


def is_meaningful_change(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    return True


def cache_path_for_file(cache_dir: Path, repo: str, base_commit: str, file_path: str) -> Path:
    return cache_dir / repo.replace("/", "__") / base_commit / file_path


def write_text_atomically(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None

    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=target_path.parent, delete=False) as f:
            temp_path = Path(f.name)
            f.write(content)
        temp_path.replace(target_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def build_download_request(url: str) -> Request:
    headers = {"User-Agent": "DaLIC-ground-truth-extractor/2.0"}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return Request(url, headers=headers)


def load_base_file_text(
    repo: str,
    base_commit: str,
    file_path: str,
    history_state: dict[str, Any],
) -> str | None:
    cache_key = (repo, base_commit, file_path)
    memory_cache = history_state["memory_cache"]
    if cache_key in memory_cache:
        history_state["stats"]["memory_cache_hits"] += 1
        return memory_cache[cache_key]

    cache_dir: Path = history_state["cache_dir"]
    cached_file_path = cache_path_for_file(cache_dir, repo, base_commit, file_path)
    if cached_file_path.exists():
        history_state["stats"]["disk_cache_hits"] += 1
        text = cached_file_path.read_text(encoding="utf-8")
        memory_cache[cache_key] = text
        return text

    raw_url = f"https://raw.githubusercontent.com/{repo}/{base_commit}/{file_path}"
    request = build_download_request(raw_url)

    try:
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        if exc.code == 404:
            history_state["stats"]["missing_remote_files"] += 1
            memory_cache[cache_key] = None
            return None
        history_state["stats"]["download_failures"] += 1
        memory_cache[cache_key] = None
        return None
    except (URLError, OSError):
        history_state["stats"]["download_failures"] += 1
        memory_cache[cache_key] = None
        return None

    write_text_atomically(cached_file_path, text)
    history_state["stats"]["downloaded_files"] += 1
    memory_cache[cache_key] = text
    return text


def format_scope_id(file_path: str, qualname_parts: tuple[str, ...] | list[str] | None = None) -> str:
    if not qualname_parts:
        return file_path
    return f"{file_path}::" + ".".join(qualname_parts)


def build_ground_truth_container() -> dict[str, dict[str, set[str]]]:
    return {
        "added": {"modules": set(), "classes": set(), "functions": set()},
        "edited": {"modules": set(), "classes": set(), "functions": set()},
    }


def finalize_ground_truth(ground_truth: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, list[str]]]:
    return {
        change_type: {
            level: sorted(values)
            for level, values in levels.items()
        }
        for change_type, levels in ground_truth.items()
    }


def build_certainty_state() -> dict[str, set[str]]:
    return {"reasons": set()}


def mark_not_sure(certainty_state: dict[str, set[str]], reason: str) -> None:
    certainty_state["reasons"].add(reason)


def finalize_certainty_state(certainty_state: dict[str, set[str]]) -> tuple[str, list[str]]:
    reasons = sorted(certainty_state["reasons"])
    if reasons:
        return "not_sure", reasons
    return "sure", []


def make_symbol_record(
    qualname_parts: tuple[str, ...],
    class_prefixes: tuple[tuple[str, ...], ...],
    start: int,
    end: int,
) -> dict[str, Any]:
    return {
        "qualname_parts": qualname_parts,
        "class_prefixes": class_prefixes,
        "start": int(start),
        "end": int(end),
        "leaf_name": qualname_parts[-1],
    }


class ScopeIndexCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope_stack: list[dict[str, str]] = []
        self.class_symbols: list[dict[str, Any]] = []
        self.function_symbols: list[dict[str, Any]] = []

    def _current_scope_names(self) -> list[str]:
        return [scope["name"] for scope in self.scope_stack]

    def _current_class_prefixes(self) -> list[tuple[str, ...]]:
        prefixes: list[tuple[str, ...]] = []
        for index, scope in enumerate(self.scope_stack):
            if scope["kind"] != "class":
                continue
            prefixes.append(tuple(stack_scope["name"] for stack_scope in self.scope_stack[: index + 1]))
        return prefixes

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname_parts = tuple([*self._current_scope_names(), node.name])
        class_prefixes = tuple([*self._current_class_prefixes(), qualname_parts])
        end_lineno = getattr(node, "end_lineno", node.lineno)
        decorator_lines = [int(decorator.lineno) for decorator in node.decorator_list if hasattr(decorator, "lineno")]
        start_lineno = min([int(node.lineno), *decorator_lines]) if decorator_lines else int(node.lineno)
        self.class_symbols.append(
            make_symbol_record(
                qualname_parts=qualname_parts,
                class_prefixes=class_prefixes,
                start=start_lineno,
                end=int(end_lineno),
            )
        )

        self.scope_stack.append({"kind": "class", "name": node.name})
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname_parts = tuple([*self._current_scope_names(), node.name])
        class_prefixes = tuple(self._current_class_prefixes())
        end_lineno = getattr(node, "end_lineno", node.lineno)
        decorator_lines = [int(decorator.lineno) for decorator in node.decorator_list if hasattr(decorator, "lineno")]
        start_lineno = min([int(node.lineno), *decorator_lines]) if decorator_lines else int(node.lineno)
        self.function_symbols.append(
            make_symbol_record(
                qualname_parts=qualname_parts,
                class_prefixes=class_prefixes,
                start=start_lineno,
                end=int(end_lineno),
            )
        )

        self.scope_stack.append({"kind": "def", "name": node.name})
        self.generic_visit(node)
        self.scope_stack.pop()


def build_symbol_index_from_ast(source_text: str) -> dict[str, list[dict[str, Any]]]:
    module = ast.parse(source_text)
    collector = ScopeIndexCollector()
    collector.visit(module)
    return {
        "classes": collector.class_symbols,
        "functions": collector.function_symbols,
    }


def build_symbol_index_from_indentation(source_text: str) -> dict[str, list[dict[str, Any]]]:
    class_symbols: list[dict[str, Any]] = []
    function_symbols: list[dict[str, Any]] = []
    scope_stack: list[dict[str, Any]] = []
    lines = source_text.splitlines()
    pending_decorator_start: int | None = None

    def current_scope_names() -> list[str]:
        return [scope["name"] for scope in scope_stack]

    def current_class_prefixes() -> list[tuple[str, ...]]:
        prefixes: list[tuple[str, ...]] = []
        for index, scope in enumerate(scope_stack):
            if scope["kind"] != "class":
                continue
            prefixes.append(tuple(stack_scope["name"] for stack_scope in scope_stack[: index + 1]))
        return prefixes

    def close_scopes(current_indent: int, current_line_number: int) -> None:
        while scope_stack and current_indent <= int(scope_stack[-1]["indent"]):
            finished_scope = scope_stack.pop()
            symbol_list = (
                class_symbols
                if finished_scope["kind"] == "class"
                else function_symbols
            )
            symbol_list[int(finished_scope["symbol_index"])]["end"] = current_line_number - 1

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        kind, name, indent = parse_symbol(line)
        current_indent = indent_width(line[: len(line) - len(line.lstrip(" \t"))])

        if stripped:
            close_scopes(current_indent, line_number)

        if stripped.startswith("@"):
            if pending_decorator_start is None:
                pending_decorator_start = line_number
            continue

        if kind == "class" and name and indent is not None:
            start_line = pending_decorator_start if pending_decorator_start is not None else line_number
            qualname_parts = tuple([*current_scope_names(), name])
            class_prefixes = tuple([*current_class_prefixes(), qualname_parts])
            class_symbols.append(
                make_symbol_record(
                    qualname_parts=qualname_parts,
                    class_prefixes=class_prefixes,
                    start=start_line,
                    end=len(lines),
                )
            )
            scope_stack.append(
                {
                    "kind": "class",
                    "name": name,
                    "indent": indent,
                    "symbol_index": len(class_symbols) - 1,
                }
            )
            pending_decorator_start = None
            continue

        if kind == "def" and name and indent is not None:
            start_line = pending_decorator_start if pending_decorator_start is not None else line_number
            qualname_parts = tuple([*current_scope_names(), name])
            class_prefixes = tuple(current_class_prefixes())
            function_symbols.append(
                make_symbol_record(
                    qualname_parts=qualname_parts,
                    class_prefixes=class_prefixes,
                    start=start_line,
                    end=len(lines),
                )
            )
            scope_stack.append(
                {
                    "kind": "def",
                    "name": name,
                    "indent": indent,
                    "symbol_index": len(function_symbols) - 1,
                }
            )
            pending_decorator_start = None
            continue

        if stripped:
            pending_decorator_start = None

    close_scopes(-1, len(lines) + 1)
    return {
        "classes": class_symbols,
        "functions": function_symbols,
    }


def build_symbol_index(file_path: str, source_text: str) -> tuple[dict[str, list[dict[str, Any]]], str]:
    if not source_text.strip():
        return {"classes": [], "functions": []}, "empty"

    suffix = Path(file_path).suffix
    if suffix == ".py":
        try:
            return build_symbol_index_from_ast(source_text), "ast"
        except SyntaxError:
            return build_symbol_index_from_indentation(source_text), "indentation_syntax_fallback"

    return build_symbol_index_from_indentation(source_text), "indentation_non_py"


def symbol_depth(symbol: dict[str, Any]) -> int:
    return len(symbol["qualname_parts"])


def find_best_symbol(
    symbols: list[dict[str, Any]],
    line_number: int | None,
    leaf_name: str | None = None,
) -> dict[str, Any] | None:
    if line_number is None:
        return None

    matched_symbols = [
        symbol
        for symbol in symbols
        if int(symbol["start"]) <= line_number <= int(symbol["end"])
    ]
    if not matched_symbols:
        return None

    if leaf_name is not None:
        exact_name_matches = [
            symbol
            for symbol in matched_symbols
            if str(symbol["leaf_name"]) == leaf_name
        ]
        if exact_name_matches:
            matched_symbols = exact_name_matches

    return max(
        matched_symbols,
        key=lambda symbol: (
            symbol_depth(symbol),
            -(int(symbol["end"]) - int(symbol["start"])),
        ),
    )


def apply_file_diff_to_source(base_text: str, patched_file: Any) -> str:
    base_lines = base_text.splitlines()
    patched_lines: list[str] = []
    base_index = 0

    for hunk in patched_file:
        old_start = int(hunk.source_start or 0)
        old_start_index = max(old_start - 1, 0)
        if old_start_index < base_index or old_start_index > len(base_lines):
            raise ValueError(f"Hunk start is out of order or out of range: {hunk}")

        patched_lines.extend(base_lines[base_index:old_start_index])
        current_base_index = old_start_index

        for hunk_line in hunk:
            line = hunk_line.value.rstrip("\n")

            if hunk_line.is_context:
                if current_base_index >= len(base_lines):
                    raise ValueError("Context line exceeds base file length.")
                base_line = base_lines[current_base_index]
                if base_line != line:
                    raise ValueError("Context line does not match base file.")
                patched_lines.append(base_line)
                current_base_index += 1
                continue

            if hunk_line.is_removed:
                if current_base_index >= len(base_lines):
                    raise ValueError("Removed line exceeds base file length.")
                base_line = base_lines[current_base_index]
                if base_line != line:
                    raise ValueError("Removed line does not match base file.")
                current_base_index += 1
                continue

            if hunk_line.is_added:
                patched_lines.append(line)
                continue

        base_index = current_base_index

    patched_lines.extend(base_lines[base_index:])
    return "\n".join(patched_lines)


def update_class_stack(class_stack: list[tuple[int, str]], indent: int, line: str) -> None:
    if line.strip():
        while class_stack and indent <= class_stack[-1][0]:
            class_stack.pop()


def build_visible_scope_parts(class_stack: list[tuple[int, str]]) -> list[str]:
    return [class_name for _, class_name in class_stack]


def build_fallback_class_event(file_path: str, class_stack: list[tuple[int, str]], name: str) -> dict[str, Any]:
    visible_parts = [*build_visible_scope_parts(class_stack), name]
    class_ids = [
        format_scope_id(file_path, tuple(visible_parts[:index]))
        for index in range(1, len(visible_parts) + 1)
    ]
    return {
        "scope_id": class_ids[-1],
        "class_ancestor_ids": class_ids,
    }


def build_fallback_function_event(file_path: str, class_stack: list[tuple[int, str]], name: str) -> dict[str, Any]:
    visible_class_parts = build_visible_scope_parts(class_stack)
    return {
        "scope_id": format_scope_id(file_path, tuple([*visible_class_parts, name])),
        "class_ancestor_ids": [
            format_scope_id(file_path, tuple(visible_class_parts[:index]))
            for index in range(1, len(visible_class_parts) + 1)
        ],
    }


def make_direct_event(
    name: str,
    old_lineno: int | None,
    new_lineno: int | None,
    fallback_scope_id: str,
    fallback_class_ancestor_ids: list[str],
    needs_resolution: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "old_lineno": old_lineno,
        "new_lineno": new_lineno,
        "fallback_scope_id": fallback_scope_id,
        "fallback_class_ancestor_ids": fallback_class_ancestor_ids,
        "needs_resolution": needs_resolution,
    }


def append_unique_event(events: list[dict[str, Any]], seen_keys: set[tuple[Any, ...]], event: dict[str, Any]) -> None:
    key = (
        event.get("name"),
        event.get("old_lineno"),
        event.get("new_lineno"),
        event.get("fallback_scope_id"),
        tuple(event.get("fallback_class_ancestor_ids", [])),
        event.get("needs_resolution"),
    )
    if key in seen_keys:
        return
    seen_keys.add(key)
    events.append(event)


def count_resolution_required_events(events: dict[str, Any]) -> int:
    direct_event_groups = (
        events["added_classes"],
        events["edited_classes"],
        events["added_functions"],
        events["edited_functions"],
    )
    direct_event_count = sum(
        1
        for event_group in direct_event_groups
        for event in event_group
        if event["needs_resolution"]
    )
    return len(events["generic_changes"]) + direct_event_count


def collect_hunk_events(
    file_path: str,
    hunk: Any,
    event_state: dict[str, Any],
) -> None:
    class_stack: list[tuple[int, str]] = []
    recent_removed_defs: dict[tuple[int, str], dict[str, Any]] = {}
    recent_removed_classes: dict[tuple[int, str], dict[str, Any]] = {}
    inside_added_structure = False

    for hunk_line in hunk:
        if hunk_line.is_context:
            marker = " "
        elif hunk_line.is_added:
            marker = "+"
        elif hunk_line.is_removed:
            marker = "-"
        else:
            continue
        line = hunk_line.value.rstrip("\n")
        old_lineno = getattr(hunk_line, "source_line_no", None)
        new_lineno = getattr(hunk_line, "target_line_no", None)

        if marker != "+":
            inside_added_structure = False
        if marker == " ":
            recent_removed_defs.clear()
            recent_removed_classes.clear()

        kind, name, indent = parse_symbol(line)

        if indent is not None:
            update_class_stack(class_stack, indent, line)

        if kind == "class" and name and indent is not None:
            fallback = build_fallback_class_event(file_path, class_stack, name)
            needs_resolution = indent > 0
            event = make_direct_event(
                name=name,
                old_lineno=old_lineno,
                new_lineno=new_lineno,
                fallback_scope_id=fallback["scope_id"],
                fallback_class_ancestor_ids=fallback["class_ancestor_ids"],
                needs_resolution=needs_resolution,
            )

            if marker == "-":
                recent_removed_classes[(indent, name)] = event
            elif marker == "+":
                matched_removed = recent_removed_classes.pop((indent, name), None)
                if matched_removed is not None:
                    append_unique_event(
                        event_state["edited_classes"],
                        event_state["edited_classes_seen"],
                        make_direct_event(
                            name=name,
                            old_lineno=matched_removed["old_lineno"],
                            new_lineno=new_lineno,
                            fallback_scope_id=event["fallback_scope_id"],
                            fallback_class_ancestor_ids=event["fallback_class_ancestor_ids"],
                            needs_resolution=matched_removed["needs_resolution"] or needs_resolution,
                        ),
                    )
                else:
                    append_unique_event(
                        event_state["added_classes"],
                        event_state["added_classes_seen"],
                        event,
                    )
                inside_added_structure = True

            if marker != "-":
                class_stack.append((indent, name))
            continue

        if kind == "def" and name and indent is not None:
            fallback = build_fallback_function_event(file_path, class_stack, name)
            needs_resolution = indent > 0
            event = make_direct_event(
                name=name,
                old_lineno=old_lineno,
                new_lineno=new_lineno,
                fallback_scope_id=fallback["scope_id"],
                fallback_class_ancestor_ids=fallback["class_ancestor_ids"],
                needs_resolution=needs_resolution,
            )

            if marker == "-":
                recent_removed_defs[(indent, name)] = event
            elif marker == "+":
                matched_removed = recent_removed_defs.pop((indent, name), None)
                if matched_removed is not None:
                    append_unique_event(
                        event_state["edited_functions"],
                        event_state["edited_functions_seen"],
                        make_direct_event(
                            name=name,
                            old_lineno=matched_removed["old_lineno"],
                            new_lineno=new_lineno,
                            fallback_scope_id=event["fallback_scope_id"],
                            fallback_class_ancestor_ids=event["fallback_class_ancestor_ids"],
                            needs_resolution=matched_removed["needs_resolution"] or needs_resolution,
                        ),
                    )
                else:
                    append_unique_event(
                        event_state["added_functions"],
                        event_state["added_functions_seen"],
                        event,
                    )
                inside_added_structure = True
            continue

        if marker in {"+", "-"} and is_meaningful_change(line):
            if marker == "+" and inside_added_structure:
                continue
            event_state["generic_changes"].append(
                {
                    "old_lineno": old_lineno,
                    "new_lineno": new_lineno,
                }
            )


def collect_file_events(file_path: str, patched_file: Any) -> dict[str, Any]:
    event_state: dict[str, Any] = {
        "added_classes": [],
        "edited_classes": [],
        "added_functions": [],
        "edited_functions": [],
        "generic_changes": [],
        "added_classes_seen": set(),
        "edited_classes_seen": set(),
        "added_functions_seen": set(),
        "edited_functions_seen": set(),
    }

    for hunk in patched_file:
        collect_hunk_events(
            file_path=file_path,
            hunk=hunk,
            event_state=event_state,
        )

    return event_state


def class_ids_from_symbol(file_path: str, symbol: dict[str, Any]) -> list[str]:
    return [
        format_scope_id(file_path, prefix)
        for prefix in symbol["class_prefixes"]
    ]


def scope_id_from_symbol(file_path: str, symbol: dict[str, Any]) -> str:
    return format_scope_id(file_path, symbol["qualname_parts"])


def build_symbol_map(
    file_path: str,
    symbol_index: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        symbol_type: {
            scope_id_from_symbol(file_path, symbol): symbol
            for symbol in symbols
        }
        for symbol_type, symbols in symbol_index.items()
    }


def resolve_direct_event(
    file_path: str,
    level: str,
    event: dict[str, Any],
    base_index: dict[str, list[dict[str, Any]]] | None,
    patched_index: dict[str, list[dict[str, Any]]] | None,
    history_state: dict[str, Any],
    preference: str,
    lookup_requested: bool,
) -> tuple[str | None, list[str]]:
    if not event["needs_resolution"]:
        return event["fallback_scope_id"], list(event["fallback_class_ancestor_ids"])

    symbol_lists = {
        "class": "classes",
        "function": "functions",
    }
    symbol_list_name = symbol_lists[level]
    resolved_symbol: dict[str, Any] | None = None
    lookup_attempted = base_index is not None or patched_index is not None

    if preference in {"base", "base_then_patched"} and base_index is not None:
        resolved_symbol = find_best_symbol(
            base_index[symbol_list_name],
            event["old_lineno"],
            leaf_name=event["name"],
        )

    if (
        resolved_symbol is None
        and preference in {"patched", "base_then_patched", "patched_then_base"}
        and patched_index is not None
    ):
        resolved_symbol = find_best_symbol(
            patched_index[symbol_list_name],
            event["new_lineno"],
            leaf_name=event["name"],
        )

    if (
        resolved_symbol is None
        and preference == "patched_then_base"
        and base_index is not None
    ):
        resolved_symbol = find_best_symbol(
            base_index[symbol_list_name],
            event["old_lineno"],
            leaf_name=event["name"],
        )

    if resolved_symbol is not None:
        history_state["stats"]["resolved_scope_candidates"] += 1
        return scope_id_from_symbol(file_path, resolved_symbol), class_ids_from_symbol(file_path, resolved_symbol)

    if lookup_requested and lookup_attempted:
        history_state["stats"]["unresolved_scope_candidates"] += 1
        return None, []
    if lookup_requested:
        return None, []
    return event["fallback_scope_id"], list(event["fallback_class_ancestor_ids"])


def resolve_generic_change(
    file_path: str,
    event: dict[str, Any],
    base_index: dict[str, list[dict[str, Any]]] | None,
    patched_index: dict[str, list[dict[str, Any]]] | None,
    patched_symbol_map: dict[str, dict[str, dict[str, Any]]] | None,
    history_state: dict[str, Any],
) -> tuple[str, str | None, list[str]]:
    function_symbol: dict[str, Any] | None = None
    class_symbol: dict[str, Any] | None = None
    lookup_attempted = base_index is not None or patched_index is not None

    if patched_index is not None:
        function_symbol = find_best_symbol(
            patched_index["functions"],
            event["new_lineno"],
        )
    if function_symbol is not None:
        history_state["stats"]["resolved_scope_candidates"] += 1
        return "function", scope_id_from_symbol(file_path, function_symbol), class_ids_from_symbol(file_path, function_symbol)

    if base_index is not None and patched_symbol_map is not None:
        base_function_symbol = find_best_symbol(
            base_index["functions"],
            event["old_lineno"],
        )
        if base_function_symbol is not None:
            base_function_id = scope_id_from_symbol(file_path, base_function_symbol)
            patched_function_symbol = patched_symbol_map["functions"].get(base_function_id)
            if patched_function_symbol is not None:
                history_state["stats"]["resolved_scope_candidates"] += 1
                return (
                    "function",
                    base_function_id,
                    class_ids_from_symbol(file_path, patched_function_symbol),
                )

    if patched_index is not None:
        class_symbol = find_best_symbol(
            patched_index["classes"],
            event["new_lineno"],
        )
    if class_symbol is not None:
        history_state["stats"]["resolved_scope_candidates"] += 1
        return "class", scope_id_from_symbol(file_path, class_symbol), class_ids_from_symbol(file_path, class_symbol)

    if base_index is not None and patched_symbol_map is not None:
        base_class_symbol = find_best_symbol(
            base_index["classes"],
            event["old_lineno"],
        )
        if base_class_symbol is not None:
            base_class_id = scope_id_from_symbol(file_path, base_class_symbol)
            patched_class_symbol = patched_symbol_map["classes"].get(base_class_id)
            if patched_class_symbol is not None:
                history_state["stats"]["resolved_scope_candidates"] += 1
                return (
                    "class",
                    base_class_id,
                    class_ids_from_symbol(file_path, patched_class_symbol),
                )

    if lookup_attempted:
        history_state["stats"]["unresolved_scope_candidates"] += 1
    return "module", None, []


def apply_class_closure(
    ground_truth: dict[str, dict[str, set[str]]],
    added_class_ids: set[str],
    added_class_ancestor_map: dict[str, list[str]],
    edited_class_ancestor_map: dict[str, list[str]],
    added_function_class_map: dict[str, list[str]],
    edited_function_class_map: dict[str, list[str]],
) -> None:
    for class_id, class_ancestor_ids in added_class_ancestor_map.items():
        for ancestor_class_id in class_ancestor_ids[:-1]:
            if ancestor_class_id in added_class_ids:
                continue
            ground_truth["edited"]["classes"].add(ancestor_class_id)

    for class_ancestor_ids in edited_class_ancestor_map.values():
        for ancestor_class_id in class_ancestor_ids:
            if ancestor_class_id in added_class_ids:
                continue
            ground_truth["edited"]["classes"].add(ancestor_class_id)

    for class_ancestor_ids in added_function_class_map.values():
        for ancestor_class_id in class_ancestor_ids:
            if ancestor_class_id in added_class_ids:
                continue
            ground_truth["edited"]["classes"].add(ancestor_class_id)

    for class_ancestor_ids in edited_function_class_map.values():
        for ancestor_class_id in class_ancestor_ids:
            if ancestor_class_id in added_class_ids:
                continue
            ground_truth["edited"]["classes"].add(ancestor_class_id)


def maybe_build_source_indexes(
    repo: str,
    base_commit: str,
    file_path: str,
    base_lookup_path: str | None,
    patched_file: Any,
    events: dict[str, Any],
    history_state: dict[str, Any],
    certainty_state: dict[str, set[str]],
) -> tuple[
    dict[str, list[dict[str, Any]]] | None,
    dict[str, list[dict[str, Any]]] | None,
    bool,
]:
    pending_lookup_count = count_resolution_required_events(events)
    needs_source_lookup = pending_lookup_count > 0

    if not needs_source_lookup:
        return None, None, False

    base_text: str | None = None
    if base_lookup_path is not None:
        history_state["source_lookup_files"].add((repo, base_commit, base_lookup_path))
        base_text = load_base_file_text(repo, base_commit, base_lookup_path, history_state)

    if base_text is None and not patched_file.is_added_file:
        history_state["stats"]["unresolved_scope_candidates"] += pending_lookup_count
        mark_not_sure(certainty_state, "source_lookup_failed")
        return None, None, True

    if base_text is None:
        base_text = ""

    try:
        patched_text = apply_file_diff_to_source(base_text, patched_file)
    except ValueError:
        history_state["stats"]["unresolved_scope_candidates"] += pending_lookup_count
        mark_not_sure(certainty_state, "patch_context_mismatch")
        return None, None, True

    base_index, base_mode = build_symbol_index(file_path, base_text)
    patched_index, patched_mode = build_symbol_index(file_path, patched_text)
    if base_mode.startswith("indentation") or patched_mode.startswith("indentation"):
        mark_not_sure(certainty_state, "non_ast_scope_analysis_used")

    return base_index, patched_index, True


def process_file_diff(
    repo: str,
    base_commit: str,
    file_diff: dict[str, Any],
    history_state: dict[str, Any],
    certainty_state: dict[str, set[str]],
) -> dict[str, dict[str, set[str]]]:
    ground_truth = build_ground_truth_container()
    file_path = str(file_diff["file_path"])
    base_lookup_path = file_diff["base_lookup_path"]
    patched_file = file_diff["patched_file"]
    file_has_hunks = len(patched_file) > 0

    if not file_has_hunks:
        if patched_file.is_removed_file:
            return ground_truth
        if patched_file.is_added_file:
            ground_truth["added"]["modules"].add(file_path)
            return ground_truth
        mark_not_sure(certainty_state, "unsupported_diff_form")
        return ground_truth

    if patched_file.is_removed_file:
        return ground_truth

    file_is_new = patched_file.is_added_file
    if file_is_new:
        ground_truth["added"]["modules"].add(file_path)
    else:
        ground_truth["edited"]["modules"].add(file_path)

    events = collect_file_events(file_path, patched_file)
    base_index, patched_index, lookup_requested = maybe_build_source_indexes(
        repo=repo,
        base_commit=base_commit,
        file_path=file_path,
        base_lookup_path=base_lookup_path,
        patched_file=patched_file,
        events=events,
        history_state=history_state,
        certainty_state=certainty_state,
    )
    patched_symbol_map = build_symbol_map(file_path, patched_index) if patched_index is not None else None

    added_class_ancestor_map: dict[str, list[str]] = {}
    edited_class_ancestor_map: dict[str, list[str]] = {}
    added_function_class_map: dict[str, list[str]] = {}
    edited_function_class_map: dict[str, list[str]] = {}

    for event in events["added_classes"]:
        class_id, class_ancestor_ids = resolve_direct_event(
            file_path=file_path,
            level="class",
            event=event,
            base_index=base_index,
            patched_index=patched_index,
            history_state=history_state,
            preference="patched",
            lookup_requested=lookup_requested,
        )
        if class_id is not None:
            ground_truth["added"]["classes"].add(class_id)
            added_class_ancestor_map[class_id] = class_ancestor_ids
        elif event["needs_resolution"]:
            mark_not_sure(certainty_state, "scope_resolution_failed")

    for event in events["edited_classes"]:
        class_id, class_ancestor_ids = resolve_direct_event(
            file_path=file_path,
            level="class",
            event=event,
            base_index=base_index,
            patched_index=patched_index,
            history_state=history_state,
            preference="base_then_patched",
            lookup_requested=lookup_requested,
        )
        if class_id is not None:
            if class_id not in ground_truth["added"]["classes"]:
                ground_truth["edited"]["classes"].add(class_id)
            edited_class_ancestor_map[class_id] = class_ancestor_ids
        elif event["needs_resolution"]:
            mark_not_sure(certainty_state, "scope_resolution_failed")

    for event in events["added_functions"]:
        function_id, class_ancestor_ids = resolve_direct_event(
            file_path=file_path,
            level="function",
            event=event,
            base_index=base_index,
            patched_index=patched_index,
            history_state=history_state,
            preference="patched",
            lookup_requested=lookup_requested,
        )
        if function_id is not None:
            ground_truth["added"]["functions"].add(function_id)
            added_function_class_map[function_id] = class_ancestor_ids
        elif event["needs_resolution"]:
            mark_not_sure(certainty_state, "scope_resolution_failed")

    for event in events["edited_functions"]:
        function_id, class_ancestor_ids = resolve_direct_event(
            file_path=file_path,
            level="function",
            event=event,
            base_index=base_index,
            patched_index=patched_index,
            history_state=history_state,
            preference="base_then_patched",
            lookup_requested=lookup_requested,
        )
        if function_id is not None:
            if function_id not in ground_truth["added"]["functions"]:
                ground_truth["edited"]["functions"].add(function_id)
            edited_function_class_map[function_id] = class_ancestor_ids
        elif event["needs_resolution"]:
            mark_not_sure(certainty_state, "scope_resolution_failed")

    for event in events["generic_changes"]:
        resolved_level, scope_id, class_ancestor_ids = resolve_generic_change(
            file_path=file_path,
            event=event,
            base_index=base_index,
            patched_index=patched_index,
            patched_symbol_map=patched_symbol_map,
            history_state=history_state,
        )

        if resolved_level == "function" and scope_id is not None:
            if scope_id in ground_truth["added"]["functions"]:
                continue
            ground_truth["edited"]["functions"].add(scope_id)
            edited_function_class_map.setdefault(scope_id, class_ancestor_ids)
            continue

        if resolved_level == "class" and scope_id is not None:
            if scope_id in ground_truth["added"]["classes"]:
                continue
            ground_truth["edited"]["classes"].add(scope_id)
            edited_class_ancestor_map.setdefault(scope_id, class_ancestor_ids)
            continue

    apply_class_closure(
        ground_truth=ground_truth,
        added_class_ids=ground_truth["added"]["classes"],
        added_class_ancestor_map=added_class_ancestor_map,
        edited_class_ancestor_map=edited_class_ancestor_map,
        added_function_class_map=added_function_class_map,
        edited_function_class_map=edited_function_class_map,
    )

    return ground_truth


def merge_ground_truth(
    target: dict[str, dict[str, set[str]]],
    file_ground_truth: dict[str, dict[str, set[str]]],
) -> None:
    for change_type in ("added", "edited"):
        for level in ("modules", "classes", "functions"):
            target[change_type][level].update(file_ground_truth[change_type][level])


def extract_ground_truth_from_patch(
    repo: str,
    base_commit: str,
    patch_text: str,
    history_state: dict[str, Any],
) -> tuple[dict[str, dict[str, list[str]]], str, list[str]]:
    ground_truth = build_ground_truth_container()
    certainty_state = build_certainty_state()
    parsed_files = parse_patch_files(patch_text, certainty_state)

    for file_diff in parsed_files:
        file_path = str(file_diff["file_path"])
        if Path(file_path).suffix not in PYTHON_LIKE_SUFFIXES:
            continue

        file_ground_truth = process_file_diff(
            repo=repo,
            base_commit=base_commit,
            file_diff=file_diff,
            history_state=history_state,
            certainty_state=certainty_state,
        )
        merge_ground_truth(ground_truth, file_ground_truth)

    certainty_label, certainty_reasons = finalize_certainty_state(certainty_state)
    return finalize_ground_truth(ground_truth), certainty_label, certainty_reasons


def build_history_state(cache_dir: Path) -> dict[str, Any]:
    return {
        "cache_dir": cache_dir,
        "memory_cache": {},
        "source_lookup_files": set(),
        "stats": {
            "memory_cache_hits": 0,
            "disk_cache_hits": 0,
            "downloaded_files": 0,
            "download_failures": 0,
            "missing_remote_files": 0,
            "resolved_scope_candidates": 0,
            "unresolved_scope_candidates": 0,
        },
    }


def print_history_summary(history_state: dict[str, Any]) -> None:
    stats = history_state["stats"]
    print(
        "History resolution summary:"
        f" files={len(history_state['source_lookup_files'])},"
        f" resolved_scope_candidates={stats['resolved_scope_candidates']},"
        f" unresolved_scope_candidates={stats['unresolved_scope_candidates']},"
        f" memory_cache_hits={stats['memory_cache_hits']},"
        f" disk_cache_hits={stats['disk_cache_hits']},"
        f" downloaded_files={stats['downloaded_files']},"
        f" missing_remote_files={stats['missing_remote_files']},"
        f" download_failures={stats['download_failures']}"
    )


def print_certainty_summary(certainty_summary: dict[str, int]) -> None:
    total = certainty_summary["sure"] + certainty_summary["not_sure"]
    sure_ratio = (certainty_summary["sure"] / total) if total else 0.0
    not_sure_ratio = (certainty_summary["not_sure"] / total) if total else 0.0
    print(
        "Ground-truth certainty summary:"
        f" sure={certainty_summary['sure']},"
        f" not_sure={certainty_summary['not_sure']},"
        f" sure_ratio={sure_ratio:.4f},"
        f" not_sure_ratio={not_sure_ratio:.4f}"
    )


def count_jsonl_rows(input_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as input_file:
        return sum(1 for _ in input_file)


def enrich_jsonl(input_path: Path, output_path: Path, cache_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    history_state = build_history_state(cache_dir)
    certainty_summary = {"sure": 0, "not_sure": 0}
    total_rows = count_jsonl_rows(input_path)

    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            with NamedTemporaryFile(
                "w", encoding="utf-8", dir=output_path.parent, delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)

                progress_bar = tqdm(
                    input_file,
                    total=total_rows,
                    desc="Processing rows",
                    unit="row",
                )
                for line in progress_bar:
                    row = json.loads(line)
                    patch_text = str(row.get("patch") or "")
                    repo = str(row.get("repo") or "")
                    base_commit = str(row.get("base_commit") or "")

                    ground_truth, certainty_label, certainty_reasons = extract_ground_truth_from_patch(
                        repo=repo,
                        base_commit=base_commit,
                        patch_text=patch_text,
                        history_state=history_state,
                    )
                    row["ground_truth"] = ground_truth
                    row["ground_truth_certainty"] = certainty_label
                    row["ground_truth_certainty_reasons"] = certainty_reasons
                    certainty_summary[certainty_label] += 1
                    temp_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                progress_bar.close()

        if temp_path is None:
            raise RuntimeError("Temporary output file was not created.")
        temp_path.replace(output_path)
        print_history_summary(history_state)
        print_certainty_summary(certainty_summary)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def main() -> None:
    args = parse_args()
    enrich_jsonl(args.input, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
