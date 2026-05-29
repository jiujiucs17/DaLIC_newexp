#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError


DEFAULT_INPUT_FILE = "swebench_feature_with_truth.jsonl"
DEFAULT_OUTPUT_FILE = "swebench_feature_with_truth_sr_scripts.jsonl"
DEFAULT_CACHE_DIR = ".gt_source_cache"
DEFAULT_TEST_BUDGET = 8
DOWNLOAD_TIMEOUT_SECONDS = 30
DIRECT_TEST_BODY_FILE_CLASSES = {
    "py_test_body_only",
    "py_test_and_non_test_lines",
}
PARAMETRIZE_DECORATOR_NAMES = {
    "parametrize",
    "pytest.mark.parametrize",
    "mark.parametrize",
}
RAISES_CONTEXT_NAMES = {
    "pytest.raises",
    "raises",
    "self.assertRaises",
    "self.assertRaisesRegex",
    "assertRaises",
    "assertRaisesRegex",
    "pytest.warns",
    "warns",
    "does_not_raise",
}
ASSERTION_CALL_NAMES = {
    "pytest.fail",
    "fail",
    "pytest.skip",
    "skip",
    "pytest.xfail",
    "xfail",
    "self.fail",
    "self.skipTest",
    "raises",
    "self.assertRaises",
    "self.assertRaisesRegex",
    "assertRaises",
    "assertRaisesRegex",
}
ASSERTION_ATTR_PREFIXES = (
    "assert",
    "fail",
    "skip",
)
KEEP_EXPRESSION = object()
RUNTIME_FIXTURE_EXPR = {
    "monkeypatch": "_sr_monkeypatch()",
    "tmp_path": "_sr_tmp_path()",
    "tmp_path_factory": "_sr_tmp_path_factory()",
    "tmpdir": "_sr_tmpdir()",
    "tmpdir_factory": "_sr_tmpdir_factory()",
}
RELIABILITY_LEVEL_0 = "0. 未达到第1级：缺少直接 test-body 锚点或 A/B 为空"
RELIABILITY_LEVEL_1 = "1. 有直接 test-body 锚点，而且 A/B 都能生成非空"
RELIABILITY_LEVEL_2 = "2. 在第1级基础上，生成出来的 A/B 里没有 unresolved 占位 workflow"
RELIABILITY_LEVEL_3 = (
    "3. 满足 high 条件：A 的 test 直接命中 test-body，test_patch 不触及非 .py 测试资产，"
    "A/B 无 unresolved、无 fixture stub、且均非空"
)
RELIABILITY_LEVEL_4 = "4. patch 非常干净，只改直接 test body，不带任何额外 non-test 变动"
RELIABILITY_LEVELS = (
    RELIABILITY_LEVEL_0,
    RELIABILITY_LEVEL_1,
    RELIABILITY_LEVEL_2,
    RELIABILITY_LEVEL_3,
    RELIABILITY_LEVEL_4,
)


@dataclass
class PatchedPythonFile:
    file_path: str
    base_lookup_path: str | None
    patch_order: int
    patched_file: Any


@dataclass
class TestDefinition:
    file_path: str
    class_path: tuple[str, ...]
    function_name: str
    start_line: int
    end_line: int
    order_index: int
    node: ast.FunctionDef | ast.AsyncFunctionDef

    def key(self) -> tuple[str, tuple[str, ...], str]:
        return self.file_path, self.class_path, self.function_name

    def class_expr(self) -> str | None:
        if not self.class_path:
            return None
        return ".".join(self.class_path)

    def label(self, case_id: str | None = None) -> str:
        qualname = "::".join([*self.class_path, self.function_name])
        if case_id is None:
            return f"{self.file_path}::{qualname}"
        return f"{self.file_path}::{qualname}[{case_id}]"


@dataclass(frozen=True)
class ParamCase:
    values: tuple[str, ...]
    explicit_id: str | None


@dataclass
class DecoratorParamInfo:
    param_names: list[str]
    cases: list[ParamCase]


@dataclass
class ModuleSourceInfo:
    file_path: str
    source_text: str
    module: ast.Module
    bindings: dict[str, ast.AST]
    tests: list[TestDefinition]
    changed_intervals: tuple[tuple[int, int], ...]
    patch_order: int


@dataclass(frozen=True)
class ExpandedInvocation:
    raw_label: str
    file_path: str
    target_kind: str
    class_expr: str | None
    function_name: str
    keyword_arg_exprs: tuple[tuple[str, str], ...]
    note: str | None


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Generate software-reconnaissance A/B scripts from SWE-bench test_patch fields. "
            "A uses patch-modified tests, and B uses nearby unmodified tests from the same "
            "test file or module."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "dataset" / DEFAULT_INPUT_FILE,
        help=f"Input jsonl file. Default: dataset/{DEFAULT_INPUT_FILE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "dataset" / DEFAULT_OUTPUT_FILE,
        help=f"Output jsonl file. Default: dataset/{DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / "dataset" / DEFAULT_CACHE_DIR,
        help=f"Directory for cached source files. Default: dataset/{DEFAULT_CACHE_DIR}",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_TEST_BUDGET,
        help=f"Maximum number of workflows in each of script_A/script_B. Default: {DEFAULT_TEST_BUDGET}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows.",
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
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided.")
    if args.budget <= 0:
        raise ValueError("--budget must be positive.")


def get_required_field(row: dict[str, Any], field_name: str) -> Any:
    if field_name not in row:
        raise ValueError(f"Missing required field '{field_name}' in row: {row}")
    return row[field_name]


def cache_path_for_file(cache_dir: Path, repo: str, base_commit: str, file_path: str) -> Path:
    return cache_dir / repo.replace("/", "__") / base_commit / file_path


def write_text_atomically(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=target_path.parent, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
        temp_path.replace(target_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def build_download_request(url: str) -> Request:
    headers = {"User-Agent": "DaLIC-sr-script-generator/2.0"}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return Request(url, headers=headers)


def load_source_text(
    repo: str,
    base_commit: str,
    file_path: str,
    history_state: dict[str, Any],
) -> str | None:
    cache_key = (repo, base_commit, file_path)
    memory_cache: dict[tuple[str, str, str], str | None] = history_state["memory_cache"]
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
        else:
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


def normalize_patch_path(raw_path: str | None) -> str | None:
    if raw_path is None or raw_path == "/dev/null":
        return None
    if raw_path.startswith(("a/", "b/")):
        return raw_path[2:]
    return raw_path


def parse_patch_files(
    patch_text: str,
    history_state: dict[str, Any],
) -> list[tuple[str, str | None, int, Any]]:
    if not patch_text.strip():
        return []

    try:
        patch_set = PatchSet(patch_text.splitlines(True))
    except UnidiffParseError:
        history_state["stats"]["patch_parse_failures"] += 1
        return []

    parsed_files: list[tuple[str, str | None, int, Any]] = []
    for patch_order, patched_file in enumerate(patch_set):
        file_path = (
            normalize_patch_path(patched_file.target_file)
            or normalize_patch_path(patched_file.source_file)
            or normalize_patch_path(getattr(patched_file, "path", None))
        )
        if file_path is None:
            continue

        base_lookup_path = None
        if not patched_file.is_added_file:
            base_lookup_path = normalize_patch_path(patched_file.source_file) or file_path

        parsed_files.append((file_path, base_lookup_path, patch_order, patched_file))

    return parsed_files


def parse_test_patch(patch_text: str, history_state: dict[str, Any]) -> list[PatchedPythonFile]:
    parsed_files: list[PatchedPythonFile] = []
    for file_path, base_lookup_path, patch_order, patched_file in parse_patch_files(patch_text, history_state):
        if not file_path.endswith(".py"):
            continue

        parsed_files.append(
            PatchedPythonFile(
                file_path=file_path,
                base_lookup_path=base_lookup_path,
                patch_order=patch_order,
                patched_file=patched_file,
            )
        )

    return parsed_files


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


def merge_intervals(intervals: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not intervals:
        return ()

    sorted_intervals = sorted(intervals)
    merged: list[tuple[int, int]] = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
            continue
        merged.append((start, end))
    return tuple(merged)


def collect_changed_intervals(patched_file: Any) -> tuple[tuple[int, int], ...]:
    direct_intervals: list[tuple[int, int]] = []
    fallback_intervals: list[tuple[int, int]] = []

    for hunk in patched_file:
        target_start = max(int(hunk.target_start or 1), 1)
        target_length = int(hunk.target_length or 0)
        if target_length > 0:
            fallback_intervals.append((target_start, target_start + target_length - 1))
        else:
            fallback_intervals.append((target_start, target_start))

        for hunk_line in hunk:
            if hunk_line.is_added and hunk_line.target_line_no is not None:
                line_no = int(hunk_line.target_line_no)
                direct_intervals.append((line_no, line_no))

    merged_direct = merge_intervals(direct_intervals)
    if merged_direct:
        return merged_direct
    return merge_intervals(fallback_intervals)


def expand_intervals(intervals: tuple[tuple[int, int], ...]) -> set[int]:
    expanded: set[int] = set()
    for start, end in intervals:
        if end < start:
            continue
        expanded.update(range(int(start), int(end) + 1))
    return expanded


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def source_for_node(source_text: str, node: ast.AST) -> str:
    return ast.get_source_segment(source_text, node) or ast.unparse(node)


def normalize_param_id(text: str) -> str:
    return "".join(str(text).split()).strip("'\"")


def resolve_name_binding(expr: ast.AST, module_info: ModuleSourceInfo, seen: set[str] | None = None) -> ast.AST | None:
    if not isinstance(expr, ast.Name):
        return expr
    if seen is None:
        seen = set()
    if expr.id in seen:
        return None
    bound = module_info.bindings.get(expr.id)
    if bound is None:
        return expr
    seen.add(expr.id)
    if isinstance(bound, ast.Name):
        return resolve_name_binding(bound, module_info, seen)
    return bound


def parse_param_names(node: ast.AST, source_text: str) -> list[str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [part.strip() for part in node.value.split(",") if part.strip()]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                names.append(elt.value)
            elif isinstance(elt, ast.Name):
                names.append(elt.id)
            else:
                return None
        return names
    text = source_for_node(source_text, node)
    names = [part.strip() for part in text.strip("()").split(",") if part.strip()]
    return names or None


def expr_to_case_id(expr: ast.AST, source_text: str) -> str:
    if isinstance(expr, ast.Lambda):
        return "<lambda>"
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return source_for_node(source_text, expr).replace(" ", "")
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            return expr.value
        return repr(expr.value)
    if (
        isinstance(expr, ast.UnaryOp)
        and isinstance(expr.op, ast.USub)
        and isinstance(expr.operand, ast.Constant)
        and isinstance(expr.operand.value, (int, float))
    ):
        return f"-{expr.operand.value}"
    return source_for_node(source_text, expr).replace(" ", "").replace("\n", "")


def expand_string_sequence(expr: ast.AST, module_info: ModuleSourceInfo) -> list[str] | None:
    resolved = resolve_name_binding(expr, module_info)
    if resolved is None:
        return None
    if isinstance(resolved, (ast.List, ast.Tuple)):
        values: list[str] = []
        for elt in resolved.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                values.append(elt.value)
            else:
                return None
        return values
    return None


def expand_single_value_sequence(expr: ast.AST, module_info: ModuleSourceInfo) -> list[tuple[str, str]] | None:
    resolved = resolve_name_binding(expr, module_info)
    if resolved is None:
        return None

    if isinstance(resolved, (ast.List, ast.Tuple)):
        return [
            (source_for_node(module_info.source_text, elt), expr_to_case_id(elt, module_info.source_text))
            for elt in resolved.elts
        ]
    return None


def parse_pytest_param_case(
    expr: ast.Call,
    module_info: ModuleSourceInfo,
    expected_arity: int,
) -> ParamCase | None:
    func_name = dotted_name(expr.func)
    if func_name not in {"pytest.param", "param"}:
        return None

    explicit_id: str | None = None
    for keyword in expr.keywords:
        if keyword.arg == "id" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            explicit_id = keyword.value.value

    if expected_arity == 1:
        if len(expr.args) != 1:
            return None
        return ParamCase(values=(source_for_node(module_info.source_text, expr.args[0]),), explicit_id=explicit_id)

    if len(expr.args) == expected_arity:
        return ParamCase(
            values=tuple(source_for_node(module_info.source_text, arg) for arg in expr.args),
            explicit_id=explicit_id,
        )

    if len(expr.args) == 1 and isinstance(expr.args[0], (ast.Tuple, ast.List)) and len(expr.args[0].elts) == expected_arity:
        return ParamCase(
            values=tuple(source_for_node(module_info.source_text, arg) for arg in expr.args[0].elts),
            explicit_id=explicit_id,
        )

    return None


def expand_param_cases(expr: ast.AST, module_info: ModuleSourceInfo, expected_arity: int) -> list[ParamCase] | None:
    resolved = resolve_name_binding(expr, module_info)
    if resolved is None:
        return None

    if isinstance(resolved, ast.Call) and dotted_name(resolved.func) == "zip":
        parts = [expand_single_value_sequence(arg, module_info) for arg in resolved.args]
        if not parts or any(part is None for part in parts):
            return None
        if len(parts) != expected_arity:
            return None
        cases: list[ParamCase] = []
        for zipped_values in zip(*parts):
            values = tuple(item[0] for item in zipped_values)
            case_id = "-".join(item[1] for item in zipped_values)
            cases.append(ParamCase(values=values, explicit_id=case_id))
        return cases

    if isinstance(resolved, (ast.List, ast.Tuple)):
        cases: list[ParamCase] = []
        for elt in resolved.elts:
            if isinstance(elt, ast.Call):
                param_case = parse_pytest_param_case(elt, module_info, expected_arity)
                if param_case is None:
                    return None
                cases.append(param_case)
                continue

            if expected_arity == 1:
                cases.append(
                    ParamCase(
                        values=(source_for_node(module_info.source_text, elt),),
                        explicit_id=expr_to_case_id(elt, module_info.source_text),
                    )
                )
                continue

            if isinstance(elt, (ast.Tuple, ast.List)) and len(elt.elts) == expected_arity:
                case_values = tuple(source_for_node(module_info.source_text, child) for child in elt.elts)
                case_id = "-".join(expr_to_case_id(child, module_info.source_text) for child in elt.elts)
                cases.append(ParamCase(values=case_values, explicit_id=case_id))
                continue

            return None
        return cases

    return None


def extract_parametrize_info(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_info: ModuleSourceInfo,
) -> list[DecoratorParamInfo] | None:
    result: list[DecoratorParamInfo] = []
    for decorator in function_node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        dotted = dotted_name(decorator.func)
        if dotted not in PARAMETRIZE_DECORATOR_NAMES:
            continue
        if len(decorator.args) < 2:
            return None
        param_names = parse_param_names(decorator.args[0], module_info.source_text)
        if not param_names:
            return None
        cases = expand_param_cases(decorator.args[1], module_info, len(param_names))
        if cases is None:
            return None

        for keyword in decorator.keywords:
            if keyword.arg == "ids":
                ids = expand_string_sequence(keyword.value, module_info)
                if ids is not None and len(ids) == len(cases):
                    cases = [
                        ParamCase(values=case.values, explicit_id=explicit_id)
                        for case, explicit_id in zip(cases, ids)
                    ]
        result.append(DecoratorParamInfo(param_names=param_names, cases=cases))
    return result


def positional_param_names(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    params = list(function_node.args.posonlyargs) + list(function_node.args.args)
    return [param.arg for param in params]


def required_param_names(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    param_names = positional_param_names(function_node)
    defaults = list(function_node.args.defaults)
    required_count = len(param_names) - len(defaults)
    return set(param_names[:required_count])


def expand_keyword_arg_options(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_info: ModuleSourceInfo,
) -> tuple[list[tuple[tuple[tuple[str, str], ...], str | None]] | None, str | None]:
    decorators = extract_parametrize_info(function_node, module_info)
    if decorators is None:
        return None, "could not statically expand one or more @pytest.mark.parametrize decorators"

    combinations: list[tuple[dict[str, str], str | None]] = [({}, None)]
    for decorator in decorators:
        next_combinations: list[tuple[dict[str, str], str | None]] = []
        for current_bindings, current_id in combinations:
            for case in decorator.cases:
                merged = dict(current_bindings)
                merged.update(zip(decorator.param_names, case.values))
                case_id = case.explicit_id or "-".join(normalize_param_id(value) for value in case.values)
                if current_id:
                    combined_id = f"{current_id}-{case_id}"
                else:
                    combined_id = case_id
                next_combinations.append((merged, combined_id))
        combinations = next_combinations

    all_param_names = positional_param_names(function_node)
    if all_param_names and all_param_names[0] in {"self", "cls"}:
        call_param_names = all_param_names[1:]
        required_names = {name for name in required_param_names(function_node) if name != all_param_names[0]}
    else:
        call_param_names = all_param_names
        required_names = required_param_names(function_node)

    options: list[tuple[tuple[tuple[str, str], ...], str | None]] = []
    for bound_values, case_id in combinations:
        for name in call_param_names:
            if name in bound_values:
                continue
            runtime_expr = RUNTIME_FIXTURE_EXPR.get(name)
            if runtime_expr is not None:
                bound_values[name] = runtime_expr
        missing_required = sorted(name for name in required_names if name not in bound_values)
        if missing_required:
            return None, f"requires unresolved fixture or runtime parameter(s): {', '.join(missing_required)}"
        keyword_arg_exprs = tuple((name, bound_values[name]) for name in call_param_names if name in bound_values)
        options.append((keyword_arg_exprs, case_id))

    if not options:
        return [((), None)], None
    return options, None


def test_overlaps_intervals(test: TestDefinition, intervals: tuple[tuple[int, int], ...]) -> bool:
    for start, end in intervals:
        if not (test.end_line < start or end < test.start_line):
            return True
    return False


def line_distance_to_intervals(line_no: int, intervals: tuple[tuple[int, int], ...]) -> int:
    if not intervals:
        return 10**9
    distances: list[int] = []
    for start, end in intervals:
        if start <= line_no <= end:
            distances.append(0)
            continue
        distances.append(min(abs(line_no - start), abs(line_no - end)))
    return min(distances)


def collect_test_definitions(
    body: list[ast.stmt],
    file_path: str,
    class_path: tuple[str, ...],
    order_counter: list[int],
) -> list[TestDefinition]:
    collected: list[TestDefinition] = []
    for node in body:
        if isinstance(node, ast.ClassDef):
            collected.extend(
                collect_test_definitions(
                    body=node.body,
                    file_path=file_path,
                    class_path=(*class_path, node.name),
                    order_counter=order_counter,
                )
            )
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            collected.append(
                TestDefinition(
                    file_path=file_path,
                    class_path=class_path,
                    function_name=node.name,
                    start_line=int(node.lineno),
                    end_line=int(getattr(node, "end_lineno", node.lineno)),
                    order_index=order_counter[0],
                    node=node,
                )
            )
            order_counter[0] += 1
    return collected


def build_module_source_info(
    file_path: str,
    source_text: str,
    changed_intervals: tuple[tuple[int, int], ...],
    patch_order: int,
) -> ModuleSourceInfo:
    module = ast.parse(source_text, filename=file_path)
    bindings: dict[str, ast.AST] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            bindings[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            bindings[node.target.id] = node.value

    tests = collect_test_definitions(module.body, file_path, (), [0])
    return ModuleSourceInfo(
        file_path=file_path,
        source_text=source_text,
        module=module,
        bindings=bindings,
        tests=tests,
        changed_intervals=changed_intervals,
        patch_order=patch_order,
    )


def get_or_load_patched_module_info(
    repo: str,
    base_commit: str,
    file_diff: PatchedPythonFile,
    history_state: dict[str, Any],
    module_cache: dict[str, ModuleSourceInfo | None],
) -> ModuleSourceInfo | None:
    if file_diff.file_path in module_cache:
        return module_cache[file_diff.file_path]

    if file_diff.patched_file.is_removed_file:
        module_cache[file_diff.file_path] = None
        return None

    if file_diff.patched_file.is_added_file:
        base_text = ""
    else:
        if file_diff.base_lookup_path is None:
            module_cache[file_diff.file_path] = None
            return None
        base_text = load_source_text(repo, base_commit, file_diff.base_lookup_path, history_state)
        if base_text is None:
            module_cache[file_diff.file_path] = None
            return None

    try:
        patched_text = apply_file_diff_to_source(base_text, file_diff.patched_file)
    except ValueError:
        history_state["stats"]["patch_apply_failures"] += 1
        module_cache[file_diff.file_path] = None
        return None

    try:
        info = build_module_source_info(
            file_path=file_diff.file_path,
            source_text=patched_text,
            changed_intervals=collect_changed_intervals(file_diff.patched_file),
            patch_order=file_diff.patch_order,
        )
    except SyntaxError:
        history_state["stats"]["syntax_errors"] += 1
        module_cache[file_diff.file_path] = None
        return None

    module_cache[file_diff.file_path] = info
    return info


def classify_py_module_for_reliability(module_info: ModuleSourceInfo | None) -> str:
    if module_info is None:
        return "py_unloadable"
    if not module_info.tests:
        return "py_no_test_defs"

    changed_lines = expand_intervals(module_info.changed_intervals)
    test_lines: set[int] = set()
    for test in module_info.tests:
        test_lines.update(range(test.start_line, test.end_line + 1))

    has_overlap = any(test_overlaps_intervals(test, module_info.changed_intervals) for test in module_info.tests)
    if not has_overlap:
        return "py_test_module_non_test_only"
    if changed_lines and changed_lines.issubset(test_lines):
        return "py_test_body_only"
    return "py_test_and_non_test_lines"


class OracleStripper(ast.NodeTransformer):
    def visit_Assert(self, node: ast.Assert) -> None:
        return None

    def visit_Expr(self, node: ast.Expr) -> ast.stmt | list[ast.stmt] | None:
        value = self.visit(node.value)
        if value is None:
            return None
        if isinstance(value, ast.Call):
            replacement = self._rewrite_assertion_call(value)
            if replacement is KEEP_EXPRESSION:
                node.value = value
                return node
            return replacement
        node.value = value
        return node

    def visit_With(self, node: ast.With) -> ast.stmt | list[ast.stmt]:
        if self._is_oracle_with(node):
            return self._rewrite_oracle_with(node)
        node.items = [self.visit(item) for item in node.items]  # type: ignore[assignment]
        node.body = self._ensure_body(self.visit_statements(node.body))
        return node

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.stmt | list[ast.stmt]:
        if self._is_oracle_with(node):
            return self._rewrite_oracle_with(node)
        node.items = [self.visit(item) for item in node.items]  # type: ignore[assignment]
        node.body = self._ensure_body(self.visit_statements(node.body))
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        node.test = self.visit(node.test)
        node.body = self._ensure_body(self.visit_statements(node.body))
        node.orelse = self.visit_statements(node.orelse)
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        node.body = self._ensure_body(self.visit_statements(node.body))
        node.orelse = self.visit_statements(node.orelse)
        return node

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AsyncFor:
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        node.body = self._ensure_body(self.visit_statements(node.body))
        node.orelse = self.visit_statements(node.orelse)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        node.test = self.visit(node.test)
        node.body = self._ensure_body(self.visit_statements(node.body))
        node.orelse = self.visit_statements(node.orelse)
        return node

    def visit_Try(self, node: ast.Try) -> ast.Try:
        node.body = self._ensure_body(self.visit_statements(node.body))
        node.handlers = [self.visit(handler) for handler in node.handlers]  # type: ignore[assignment]
        node.orelse = self.visit_statements(node.orelse)
        node.finalbody = self.visit_statements(node.finalbody)
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        if node.type is not None:
            node.type = self.visit(node.type)
        node.body = self._ensure_body(self.visit_statements(node.body))
        return node

    def _ensure_body(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        if statements:
            return statements
        return [ast.Pass()]

    def _is_oracle_with(self, node: ast.With | ast.AsyncWith) -> bool:
        return any(self._is_oracle_context(item.context_expr) for item in node.items)

    def _is_oracle_context(self, expr: ast.expr) -> bool:
        if isinstance(expr, ast.Call):
            dotted = dotted_name(expr.func)
            return dotted in RAISES_CONTEXT_NAMES
        return False

    def _rewrite_oracle_with(self, node: ast.With | ast.AsyncWith) -> list[ast.stmt]:
        prelude: list[ast.stmt] = []
        non_oracle_items: list[ast.withitem] = []
        optional_var_names: list[str] = []

        for item in node.items:
            if self._is_oracle_context(item.context_expr):
                if item.optional_vars is not None:
                    optional_var_names.extend(self._collect_bound_names(item.optional_vars))
                continue
            non_oracle_items.append(item)

        for name in optional_var_names:
            prelude.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(None)))

        body = self._ensure_body(self.visit_statements(node.body))
        try_stmt = ast.Try(
            body=body,
            handlers=[ast.ExceptHandler(type=None, name=None, body=[ast.Pass()])],
            orelse=[],
            finalbody=[],
        )

        if non_oracle_items:
            if isinstance(node, ast.AsyncWith):
                wrapper: ast.stmt = ast.AsyncWith(items=non_oracle_items, body=[try_stmt], type_comment=None)
            else:
                wrapper = ast.With(items=non_oracle_items, body=[try_stmt], type_comment=None)
            return prelude + [wrapper]
        return prelude + [try_stmt]

    def _collect_bound_names(self, node: ast.AST) -> list[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            names: list[str] = []
            for child in node.elts:
                names.extend(self._collect_bound_names(child))
            return names
        return []

    def _rewrite_assertion_call(self, call: ast.Call) -> ast.stmt | list[ast.stmt] | None:
        dotted = dotted_name(call.func)
        base_name = dotted.split(".")[-1] if dotted else None

        if dotted in {"raises", "self.assertRaises", "self.assertRaisesRegex", "assertRaises", "assertRaisesRegex"}:
            if len(call.args) >= 2:
                invoked_call = ast.Call(func=call.args[1], args=call.args[2:], keywords=call.keywords)
                return ast.Try(
                    body=[ast.Expr(value=invoked_call)],
                    handlers=[ast.ExceptHandler(type=None, name=None, body=[ast.Pass()])],
                    orelse=[],
                    finalbody=[],
                )
            return None

        if dotted in ASSERTION_CALL_NAMES:
            return None

        if base_name is None:
            return KEEP_EXPRESSION

        if base_name.startswith(ASSERTION_ATTR_PREFIXES):
            return None

        return KEEP_EXPRESSION

    def visit_statements(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        rewritten: list[ast.stmt] = []
        for stmt in statements:
            new_stmt = self.visit(stmt)
            if new_stmt is None:
                continue
            if isinstance(new_stmt, list):
                rewritten.extend(new_stmt)
            else:
                rewritten.append(new_stmt)
        return rewritten


def rewrite_callable_body(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    drop_decorators: bool,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    rewritten_node = ast.fix_missing_locations(ast.parse(ast.unparse(function_node)).body[0])
    assert isinstance(rewritten_node, (ast.FunctionDef, ast.AsyncFunctionDef))
    if drop_decorators:
        rewritten_node.decorator_list = []
    rewritten_node.body = OracleStripper().visit_statements(rewritten_node.body)
    if not rewritten_node.body:
        rewritten_node.body = [ast.Pass()]
    return rewritten_node


def body_contains_tests(body: list[ast.stmt]) -> bool:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            return True
        if isinstance(node, ast.ClassDef) and body_contains_tests(node.body):
            return True
    return False


def class_has_selected_tests(
    class_path: tuple[str, ...],
    selected_targets: set[tuple[tuple[str, ...], str]],
) -> bool:
    for target_class_path, _ in selected_targets:
        if target_class_path[: len(class_path)] == class_path:
            return True
    return False


def prune_body(
    body: list[ast.stmt],
    class_path: tuple[str, ...],
    selected_targets: set[tuple[tuple[str, ...], str]],
) -> list[ast.stmt]:
    pruned: list[ast.stmt] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                if (class_path, node.name) in selected_targets:
                    pruned.append(rewrite_callable_body(node, drop_decorators=True))
                continue
            pruned.append(rewrite_callable_body(node, drop_decorators=False))
            continue

        if isinstance(node, ast.ClassDef):
            child_class_path = (*class_path, node.name)
            if class_has_selected_tests(child_class_path, selected_targets):
                rebuilt_class = ast.parse(ast.unparse(node)).body[0]
                assert isinstance(rebuilt_class, ast.ClassDef)
                rebuilt_class.body = prune_body(node.body, child_class_path, selected_targets) or [ast.Pass()]
                pruned.append(ast.fix_missing_locations(rebuilt_class))
                continue

            if body_contains_tests(node.body):
                continue

            pruned.append(node)
            continue

        pruned.append(node)

    return pruned


def build_module_source_text(
    module_info: ModuleSourceInfo,
    selected_targets: set[tuple[tuple[str, ...], str]],
) -> str:
    reduced_module = ast.Module(
        body=prune_body(module_info.module.body, (), selected_targets),
        type_ignores=[],
    )
    reduced_module = ast.fix_missing_locations(reduced_module)
    return ast.unparse(reduced_module) + "\n"


def module_name_from_file_path(file_path: str) -> str:
    if file_path.endswith("/__init__.py"):
        base = file_path[: -len("/__init__.py")]
    elif file_path.endswith(".py"):
        base = file_path[:-3]
    else:
        base = file_path
    module_name = base.replace("/", ".").replace("-", "_")
    return module_name or "__sr_module__"


def select_feature_test_definitions(module_infos: list[ModuleSourceInfo], budget: int) -> list[TestDefinition]:
    selected: list[TestDefinition] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()

    for module_info in sorted(module_infos, key=lambda info: info.patch_order):
        overlapping = [test for test in module_info.tests if test_overlaps_intervals(test, module_info.changed_intervals)]
        candidates = overlapping
        if not candidates and module_info.tests:
            candidates = sorted(
                module_info.tests,
                key=lambda test: (
                    line_distance_to_intervals(test.start_line, module_info.changed_intervals),
                    test.order_index,
                ),
            )[:1]

        for test in candidates:
            if test.key() in seen:
                continue
            selected.append(test)
            seen.add(test.key())
            if len(selected) >= budget:
                return selected

    return selected


def select_baseline_test_definitions(
    module_infos: list[ModuleSourceInfo],
    feature_tests: list[TestDefinition],
    budget: int,
) -> list[TestDefinition]:
    if not feature_tests:
        return []

    feature_by_file: dict[str, list[TestDefinition]] = defaultdict(list)
    feature_keys = {test.key() for test in feature_tests}
    for test in feature_tests:
        feature_by_file[test.file_path].append(test)

    ranked_candidates: list[tuple[int, int, int, int, TestDefinition]] = []
    for module_info in module_infos:
        feature_tests_in_file = feature_by_file.get(module_info.file_path)
        if not feature_tests_in_file:
            continue

        feature_scopes = {test.class_path for test in feature_tests_in_file}
        for test in module_info.tests:
            if test.key() in feature_keys:
                continue
            same_scope_rank = 0 if test.class_path in feature_scopes else 1
            distance = min(abs(test.start_line - feature_test.start_line) for feature_test in feature_tests_in_file)
            ranked_candidates.append(
                (
                    same_scope_rank,
                    module_info.patch_order,
                    distance,
                    test.order_index,
                    test,
                )
            )

    selected: list[TestDefinition] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for _, _, _, _, test in sorted(ranked_candidates):
        if test.key() in seen:
            continue
        selected.append(test)
        seen.add(test.key())
        if len(selected) >= budget:
            break
    return selected


def expand_test_definition(test: TestDefinition, module_info: ModuleSourceInfo) -> list[ExpandedInvocation]:
    options, note = expand_keyword_arg_options(test.node, module_info)
    if options is None:
        return [
            ExpandedInvocation(
                raw_label=test.label(),
                file_path=test.file_path,
                target_kind="method" if test.class_path else "function",
                class_expr=test.class_expr(),
                function_name=test.function_name,
                keyword_arg_exprs=(),
                note=note,
            )
        ]

    invocations: list[ExpandedInvocation] = []
    for keyword_arg_exprs, case_id in options:
        invocations.append(
            ExpandedInvocation(
                raw_label=test.label(case_id),
                file_path=test.file_path,
                target_kind="method" if test.class_path else "function",
                class_expr=test.class_expr(),
                function_name=test.function_name,
                keyword_arg_exprs=keyword_arg_exprs,
                note=None,
            )
        )
    return invocations


def build_invocations_from_tests(
    selected_tests: list[TestDefinition],
    module_infos: list[ModuleSourceInfo],
    budget: int,
) -> tuple[list[ExpandedInvocation], dict[str, set[tuple[tuple[str, ...], str]]]]:
    if budget <= 0 or not selected_tests:
        return [], {}

    module_by_path = {module_info.file_path: module_info for module_info in module_infos}
    queues: list[tuple[TestDefinition, list[ExpandedInvocation]]] = []
    for test in selected_tests:
        module_info = module_by_path.get(test.file_path)
        if module_info is None:
            continue
        queues.append((test, expand_test_definition(test, module_info)))

    selected_invocations: list[ExpandedInvocation] = []
    selected_targets: dict[str, set[tuple[tuple[str, ...], str]]] = defaultdict(set)

    while len(selected_invocations) < budget:
        progressed = False
        for test, queue in queues:
            if not queue:
                continue
            selected_invocations.append(queue.pop(0))
            selected_targets[test.file_path].add((test.class_path, test.function_name))
            progressed = True
            if len(selected_invocations) >= budget:
                break
        if not progressed:
            break

    return selected_invocations, dict(selected_targets)


def build_module_sources(
    module_infos: list[ModuleSourceInfo],
    selected_targets_by_file: dict[str, set[tuple[tuple[str, ...], str]]],
) -> list[tuple[str, str, str]]:
    module_sources: list[tuple[str, str, str]] = []
    for module_info in sorted(module_infos, key=lambda info: info.patch_order):
        selected_targets = selected_targets_by_file.get(module_info.file_path)
        if not selected_targets:
            continue
        module_sources.append(
            (
                module_info.file_path,
                module_name_from_file_path(module_info.file_path),
                build_module_source_text(module_info, selected_targets),
            )
        )
    return module_sources


def determine_reliability_level(
    *,
    file_classes: list[str],
    selected_feature_tests: list[TestDefinition],
    script_a_invocations: list[ExpandedInvocation],
    script_b_invocations: list[ExpandedInvocation],
    module_by_path: dict[str, ModuleSourceInfo],
) -> str:
    has_direct_test_body_anchor = any(file_class in DIRECT_TEST_BODY_FILE_CLASSES for file_class in file_classes)
    test_patch_touches_non_py_assets = any(file_class == "non_py" for file_class in file_classes)
    both_nonempty = bool(script_a_invocations) and bool(script_b_invocations)
    no_unresolved = all(invocation.note is None for invocation in [*script_a_invocations, *script_b_invocations])
    uses_fixture_stub = any(
        expr in RUNTIME_FIXTURE_EXPR.values()
        for invocation in [*script_a_invocations, *script_b_invocations]
        for _, expr in invocation.keyword_arg_exprs
    )
    selected_feature_tests_direct = bool(selected_feature_tests) and all(
        test_overlaps_intervals(test, module_by_path[test.file_path].changed_intervals)
        for test in selected_feature_tests
    )
    patch_is_direct_test_body_only = bool(file_classes) and all(file_class == "py_test_body_only" for file_class in file_classes)

    if (
        patch_is_direct_test_body_only
        and selected_feature_tests_direct
        and not test_patch_touches_non_py_assets
        and no_unresolved
        and not uses_fixture_stub
        and both_nonempty
    ):
        return RELIABILITY_LEVEL_4
    if (
        selected_feature_tests_direct
        and not test_patch_touches_non_py_assets
        and no_unresolved
        and not uses_fixture_stub
        and both_nonempty
    ):
        return RELIABILITY_LEVEL_3
    if has_direct_test_body_anchor and both_nonempty and no_unresolved:
        return RELIABILITY_LEVEL_2
    if has_direct_test_body_anchor and both_nonempty:
        return RELIABILITY_LEVEL_1
    return RELIABILITY_LEVEL_0


def render_script_text(
    instance_id: str,
    selected_invocations: list[ExpandedInvocation],
    module_sources: list[tuple[str, str, str]],
) -> str:
    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "import asyncio",
        "import inspect",
        "import pathlib",
        "import sys",
        "import tempfile",
        "import types",
        "",
        "# Auto-generated by scripts/generate_sr_scripts.py",
        f"# instance_id: {instance_id}",
        "",
        "def _sr_load_module(source: str, filename: str, module_name: str) -> dict[str, object]:",
        "    module = types.ModuleType(module_name)",
        "    module.__file__ = filename",
        "    module.__package__ = module_name.rpartition('.')[0]",
        "    sys.modules[module_name] = module",
        "    exec(compile(source, filename, 'exec'), module.__dict__)",
        "    return module.__dict__",
        "",
        "def _sr_eval(expr: str, namespace: dict[str, object]) -> object:",
        "    scope = dict(globals())",
        "    scope.update(namespace)",
        "    return eval(expr, scope)",
        "",
        "def _sr_eval_kwargs(items: tuple[tuple[str, str], ...], namespace: dict[str, object]) -> dict[str, object]:",
        "    return {name: _sr_eval(expr, namespace) for name, expr in items}",
        "",
        "def _sr_maybe_await(value: object) -> object:",
        "    if inspect.isawaitable(value):",
        "        return asyncio.run(value)",
        "    return value",
        "",
        "class _SrTmpDir:",
        "    def __init__(self, path: pathlib.Path) -> None:",
        "        self._path = path",
        "",
        "    def join(self, *parts: str) -> '_SrTmpDir':",
        "        return _SrTmpDir(self._path.joinpath(*parts))",
        "",
        "    def mkdir(self, *parts: str) -> '_SrTmpDir':",
        "        target = self._path.joinpath(*parts)",
        "        target.mkdir(parents=True, exist_ok=True)",
        "        return _SrTmpDir(target)",
        "",
        "    def open(self, *args: object, **kwargs: object):",
        "        return self._path.open(*args, **kwargs)",
        "",
        "    def write(self, data: object, mode: str = 'w') -> None:",
        "        if 'b' in mode:",
        "            payload = data if isinstance(data, (bytes, bytearray)) else str(data).encode()",
        "            self._path.write_bytes(payload)",
        "            return None",
        "        self._path.write_text(str(data), encoding='utf-8')",
        "        return None",
        "",
        "    def write_text(self, data: str, encoding: str = 'utf-8') -> None:",
        "        self._path.write_text(data, encoding=encoding)",
        "",
        "    def write_bytes(self, data: bytes) -> None:",
        "        self._path.write_bytes(data)",
        "",
        "    def exists(self) -> bool:",
        "        return self._path.exists()",
        "",
        "    @property",
        "    def basename(self) -> str:",
        "        return self._path.name",
        "",
        "    @property",
        "    def dirname(self) -> str:",
        "        return str(self._path.parent)",
        "",
        "    def __fspath__(self) -> str:",
        "        return str(self._path)",
        "",
        "    def __str__(self) -> str:",
        "        return str(self._path)",
        "",
        "    def __truediv__(self, part: str) -> '_SrTmpDir':",
        "        return _SrTmpDir(self._path / part)",
        "",
        "class _SrTempFactory:",
        "    def __init__(self, *, as_tmpdir: bool) -> None:",
        "        self._root = pathlib.Path(tempfile.mkdtemp())",
        "        self._as_tmpdir = as_tmpdir",
        "",
        "    def mktemp(self, basename: str) -> object:",
        "        target = self._root / basename",
        "        target.mkdir(parents=True, exist_ok=True)",
        "        if self._as_tmpdir:",
        "            return _SrTmpDir(target)",
        "        return target",
        "",
        "def _sr_tmp_path() -> pathlib.Path:",
        "    return pathlib.Path(tempfile.mkdtemp())",
        "",
        "def _sr_tmpdir() -> _SrTmpDir:",
        "    return _SrTmpDir(pathlib.Path(tempfile.mkdtemp()))",
        "",
        "def _sr_tmp_path_factory() -> _SrTempFactory:",
        "    return _SrTempFactory(as_tmpdir=False)",
        "",
        "def _sr_tmpdir_factory() -> _SrTempFactory:",
        "    return _SrTempFactory(as_tmpdir=True)",
        "",
        "def _sr_monkeypatch() -> object:",
        "    import pytest",
        "",
        "    return pytest.MonkeyPatch()",
        "",
        "def _sr_cleanup_kwargs(kwargs: dict[str, object]) -> None:",
        "    for value in kwargs.values():",
        "        undo = getattr(value, 'undo', None)",
        "        if callable(undo):",
        "            undo()",
        "",
        "def _sr_resolve_dotted(namespace: dict[str, object], dotted_name: str) -> object:",
        "    parts = dotted_name.split('.')",
        "    current = namespace[parts[0]]",
        "    for part in parts[1:]:",
        "        current = getattr(current, part)",
        "    return current",
        "",
        "def _sr_call_optional(target: object, name: str, *args: object) -> None:",
        "    value = getattr(target, name, None)",
        "    if not callable(value):",
        "        return None",
        "    try:",
        "        value(*args)",
        "    except TypeError:",
        "        value()",
        "    return None",
        "",
        "def _sr_run_function(",
        "    namespace: dict[str, object],",
        "    function_name: str,",
        "    keyword_items: tuple[tuple[str, str], ...],",
        ") -> object:",
        "    kwargs = _sr_eval_kwargs(keyword_items, namespace)",
        "    try:",
        "        return _sr_maybe_await(namespace[function_name](**kwargs))",
        "    finally:",
        "        _sr_cleanup_kwargs(kwargs)",
        "",
        "def _sr_run_method(",
        "    namespace: dict[str, object],",
        "    class_expr: str,",
        "    method_name: str,",
        "    keyword_items: tuple[tuple[str, str], ...],",
        ") -> object:",
        "    owner = _sr_resolve_dotted(namespace, class_expr)",
        "    instance = owner()",
        "    method = getattr(instance, method_name)",
        "    _sr_call_optional(instance, 'setup_method', method)",
        "    _sr_call_optional(instance, 'setUp')",
        "    kwargs = _sr_eval_kwargs(keyword_items, namespace)",
        "    try:",
        "        return _sr_maybe_await(method(**kwargs))",
        "    finally:",
        "        _sr_cleanup_kwargs(kwargs)",
        "        _sr_call_optional(instance, 'tearDown')",
        "        _sr_call_optional(instance, 'teardown_method', method)",
        "",
    ]

    module_var_names: dict[str, str] = {}
    for idx, (file_path, module_name, source_text) in enumerate(module_sources):
        source_var = f"_SR_MODULE_SOURCE_{idx}"
        namespace_var = f"_SR_MODULE_NS_{idx}"
        module_var_names[file_path] = namespace_var
        lines.append(f"{source_var} = {json.dumps(source_text, ensure_ascii=False)}")
        lines.append(
            f"{namespace_var} = _sr_load_module("
            f"{source_var}, {json.dumps(file_path)}, {json.dumps(module_name)})"
        )
        lines.append("")

    for idx, invocation in enumerate(selected_invocations):
        wrapper_name = f"workflow_{idx:04d}"
        lines.append(f"def {wrapper_name}() -> object:")
        lines.append(f"    # {invocation.raw_label}")
        if invocation.note is not None:
            lines.append(f"    # unresolved: {invocation.note}")
            lines.append("    return None")
            lines.append("")
            continue

        namespace_var = module_var_names.get(invocation.file_path)
        if namespace_var is None:
            lines.append("    # unresolved: missing generated module namespace")
            lines.append("    return None")
            lines.append("")
            continue

        keyword_items_literal = json.dumps(list(invocation.keyword_arg_exprs), ensure_ascii=False)
        if invocation.target_kind == "function":
            lines.append(
                f"    return _sr_run_function("
                f"{namespace_var}, {json.dumps(invocation.function_name)}, tuple({keyword_items_literal}))"
            )
        else:
            lines.append(
                f"    return _sr_run_method("
                f"{namespace_var}, {json.dumps(invocation.class_expr)}, "
                f"{json.dumps(invocation.function_name)}, tuple({keyword_items_literal}))"
            )
        lines.append("")

    lines.append("def main() -> None:")
    if selected_invocations:
        for idx in range(len(selected_invocations)):
            lines.append(f"    workflow_{idx:04d}()")
    else:
        lines.append("    return None")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    lines.append("")
    return "\n".join(lines)


def generate_sr_scripts_for_row(
    row: dict[str, Any],
    history_state: dict[str, Any],
    budget: int,
) -> tuple[str, str, dict[str, Any]]:
    repo = str(get_required_field(row, "repo"))
    base_commit = str(get_required_field(row, "base_commit"))
    patch_text = str(row.get("test_patch") or "")
    patch_records = parse_patch_files(patch_text, history_state)

    module_cache: dict[str, ModuleSourceInfo | None] = {}
    module_infos: list[ModuleSourceInfo] = []
    file_classes: list[str] = []
    for file_path, base_lookup_path, patch_order, patched_file in patch_records:
        if not file_path.endswith(".py"):
            file_classes.append("non_py")
            continue
        patch_file = PatchedPythonFile(
            file_path=file_path,
            base_lookup_path=base_lookup_path,
            patch_order=patch_order,
            patched_file=patched_file,
        )
        module_info = get_or_load_patched_module_info(
            repo=repo,
            base_commit=base_commit,
            file_diff=patch_file,
            history_state=history_state,
            module_cache=module_cache,
        )
        file_classes.append(classify_py_module_for_reliability(module_info))
        if module_info is None or not module_info.tests:
            continue
        module_infos.append(module_info)

    selected_feature_tests = select_feature_test_definitions(module_infos, budget)
    selected_baseline_tests = select_baseline_test_definitions(module_infos, selected_feature_tests, budget)

    script_a_invocations, script_a_targets = build_invocations_from_tests(selected_feature_tests, module_infos, budget)
    script_b_invocations, script_b_targets = build_invocations_from_tests(selected_baseline_tests, module_infos, budget)
    module_by_path = {module_info.file_path: module_info for module_info in module_infos}
    reliability_level = determine_reliability_level(
        file_classes=file_classes,
        selected_feature_tests=selected_feature_tests,
        script_a_invocations=script_a_invocations,
        script_b_invocations=script_b_invocations,
        module_by_path=module_by_path,
    )

    script_a = render_script_text(
        instance_id=str(get_required_field(row, "instance_id")),
        selected_invocations=script_a_invocations,
        module_sources=build_module_sources(module_infos, script_a_targets),
    )
    script_b = render_script_text(
        instance_id=str(get_required_field(row, "instance_id")),
        selected_invocations=script_b_invocations,
        module_sources=build_module_sources(module_infos, script_b_targets),
    )

    stats = {
        "script_a_workflows": len(script_a_invocations),
        "script_b_workflows": len(script_b_invocations),
        "script_a_unresolved": sum(1 for invocation in script_a_invocations if invocation.note is not None),
        "script_b_unresolved": sum(1 for invocation in script_b_invocations if invocation.note is not None),
        "script_a_selected_tests": len(selected_feature_tests),
        "script_b_selected_tests": len(selected_baseline_tests),
        "A_B_Reliability": reliability_level,
    }
    return script_a, script_b, stats


def export_with_sr_scripts(
    input_path: Path,
    output_path: Path,
    history_state: dict[str, Any],
    budget: int,
    limit: int | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for idx, line in enumerate(input_file):
            if limit is not None and idx >= limit:
                break
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Input file is empty: {input_path}")

    stats = {
        "rows_processed": 0,
        "rows_with_empty_A": 0,
        "rows_with_empty_B": 0,
        "script_a_workflows": 0,
        "script_b_workflows": 0,
        "script_a_unresolved": 0,
        "script_b_unresolved": 0,
        "A_B_Reliability_counts": {level: 0 for level in RELIABILITY_LEVELS},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            for row in tqdm(rows, total=len(rows), desc="Generating SR scripts", unit="row"):
                script_a, script_b, row_stats = generate_sr_scripts_for_row(row, history_state, budget)

                output_row = dict(row)
                output_row["script_A"] = script_a
                output_row["script_B"] = script_b
                output_row["A_B_Reliability"] = str(row_stats["A_B_Reliability"])

                temp_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                stats["rows_processed"] += 1
                stats["script_a_workflows"] += row_stats["script_a_workflows"]
                stats["script_b_workflows"] += row_stats["script_b_workflows"]
                stats["script_a_unresolved"] += row_stats["script_a_unresolved"]
                stats["script_b_unresolved"] += row_stats["script_b_unresolved"]
                stats["A_B_Reliability_counts"][str(row_stats["A_B_Reliability"])] += 1
                if row_stats["script_a_workflows"] == 0:
                    stats["rows_with_empty_A"] += 1
                if row_stats["script_b_workflows"] == 0:
                    stats["rows_with_empty_B"] += 1

        if temp_path is None:
            raise RuntimeError("Temporary output file was not created.")
        temp_path.replace(output_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise

    return stats


def build_history_state(cache_dir: Path) -> dict[str, Any]:
    return {
        "cache_dir": cache_dir,
        "memory_cache": {},
        "stats": {
            "memory_cache_hits": 0,
            "disk_cache_hits": 0,
            "downloaded_files": 0,
            "download_failures": 0,
            "missing_remote_files": 0,
            "patch_parse_failures": 0,
            "patch_apply_failures": 0,
            "syntax_errors": 0,
        },
    }


def main() -> None:
    args = parse_args()
    history_state = build_history_state(args.cache_dir)
    export_stats = export_with_sr_scripts(
        input_path=args.input,
        output_path=args.output,
        history_state=history_state,
        budget=args.budget,
        limit=args.limit,
    )
    cache_stats = history_state["stats"]
    print(
        json.dumps(
            {
                **export_stats,
                **cache_stats,
                "budget": args.budget,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
