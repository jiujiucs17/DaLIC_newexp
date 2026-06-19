#!/usr/bin/env python3
# Options:
#   --input PATH
#       default: dataset/swebench_feature_with_truth.jsonl (schedule only sure rows)
#   --output-dir PATH
#       default: dataset/vartracer_results
#   --cache-dir PATH
#       default: dataset/.gt_source_cache
#   --vartracer-wheel PATH
#       default: tools/vartracer/vartracer-0.2.15-py3-none-any.whl
#   --test [BOOL]
#       default: False (trace all sure instances; when enabled, randomly trace at most 5)
#   --seed INTEGER
#       default: None (fresh random sample in test mode; ignored outside test mode)
#   --run-id TEXT
#       default: trace_test_cases
#   --timeout INTEGER
#       default: 600 seconds normally; 300 seconds in test mode
#   --resume
#       default: False (rerun all; when enabled, skip finalized current-policy results)
#
# Relative paths are resolved from the repository root. Resume mode still runs
# static screening for every scheduled instance before deciding which traces to skip.

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import random
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError



UTF8 = "utf-8"
DOCKER_USER = "root"
DOCKER_WORKDIR = "/testbed"
OFFICIAL_NAMESPACE = "swebench"
DOWNLOAD_TIMEOUT_SECONDS = 30
DEFAULT_INPUT_FILE = "swebench_feature_with_truth.jsonl"
DEFAULT_OUTPUT_DIR = "vartracer_results"
DEFAULT_CACHE_DIR = ".gt_source_cache"
DEFAULT_RUN_ID = "trace_test_cases"
TRACE_ELIGIBLE_TOML_NAME = "trace_eligible_instance.toml"
TRACE_COMPLETED_TOML_NAME = "trace_completed_instance.toml"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_TEST_TIMEOUT_SECONDS = 300
TEST_SAMPLE_SIZE = 5
IMAGE_LIMIT_BYTES = 800 * 1000**3
MAX_TEST_CASES_PER_INSTANCE = 200
RECOVERY_POLICY = "direct_test_patch_changes_only_v2"
PARAMETRIZE_DECORATOR_NAMES = {
    "parametrize",
    "pytest.mark.parametrize",
    "mark.parametrize",
}
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

    def base_name(self) -> str:
        qualname = "::".join([*self.class_path, self.function_name])
        return f"{self.file_path}::{qualname}"

    def name(self, param_case_id: str | None = None) -> str:
        if param_case_id is None:
            return self.base_name()
        return f"{self.base_name()}[{param_case_id}]"


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
    test_class_names: set[str]
    base_tests: list[TestDefinition]
    base_test_class_names: set[str]
    changed_regions: tuple[ChangedRegion, ...]
    patch_order: int


@dataclass(frozen=True)
class ChangedRegion:
    kind: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class TestCase:
    instance_id: str
    test_case_id: str
    test_case_name: str
    pytest_nodeid: str
    file_path: str
    class_path: tuple[str, ...]
    function_name: str
    param_case_id: str | None


@dataclass
class ExtractionResult:
    test_cases: list[TestCase]
    module_sources: dict[str, str]
    changed_regions: int
    matched_changed_regions: int
    unmatched_changes: list[dict[str, Any]]

    @property
    def unmatched_changed_regions(self) -> int:
        return len(self.unmatched_changes)

    @property
    def fully_recovered(self) -> bool:
        return bool(self.test_cases) and not self.unmatched_changes


@dataclass
class ScreeningResult:
    extractions_by_instance_id: dict[str, ExtractionResult]
    eligible_instance_ids: list[str]
    category_counts: dict[str, int]


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else repo_root / path).resolve()


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace feature-positive test cases from SWE-bench test_patch fields "
            "with VarTracer inside official historical SWE-bench containers."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "dataset" / DEFAULT_INPUT_FILE,
        help=f"Input dataset jsonl. Default: dataset/{DEFAULT_INPUT_FILE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "dataset" / DEFAULT_OUTPUT_DIR,
        help=f"Trace output directory. Default: dataset/{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / "dataset" / DEFAULT_CACHE_DIR,
        help=f"Source cache directory. Default: dataset/{DEFAULT_CACHE_DIR}",
    )
    parser.add_argument(
        "--vartracer-wheel",
        type=Path,
        default=repo_root / "tools" / "vartracer" / "vartracer-0.2.15-py3-none-any.whl",
        help=(
            "Local VarTracer wheel copied into each container. "
            "Default: tools/vartracer/vartracer-0.2.15-py3-none-any.whl"
        ),
    )
    parser.add_argument(
        "--test",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="When true, randomly sample 5 sure instances instead of tracing all sure instances.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --test sampling. Omit for a fresh random sample each run.")
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=f"Run identifier passed to the SWE-bench container builder. Default: {DEFAULT_RUN_ID}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Positive per-command timeout in seconds. When omitted: "
            f"{DEFAULT_TIMEOUT_SECONDS} normally, {DEFAULT_TEST_TIMEOUT_SECONDS} with --test."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip instances whose summary.json is already finalized. "
            "Rerun instances with missing, unreadable, unfinished, or older-policy summaries. "
            "Static screening still runs before resume decisions."
        ),
    )
    args = parser.parse_args()

    args.input = resolve_path(args.input, repo_root)
    args.output_dir = resolve_path(args.output_dir, repo_root)
    args.cache_dir = resolve_path(args.cache_dir, repo_root)
    args.vartracer_wheel = resolve_path(args.vartracer_wheel, repo_root)

    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if not args.vartracer_wheel.is_file():
        parser.error(f"VarTracer wheel does not exist: {args.vartracer_wheel}")
    if not hasattr(args, "timeout"):
        args.timeout = DEFAULT_TEST_TIMEOUT_SECONDS if args.test else DEFAULT_TIMEOUT_SECONDS
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def get_required_field(row: dict[str, Any], field_name: str) -> Any:
    if field_name not in row:
        raise ValueError(f"Missing required field '{field_name}' in row: {row}")
    return row[field_name]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding=UTF8)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding=UTF8))


def write_text_atomically(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding=UTF8, dir=target_path.parent, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
        temp_path.replace(target_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def write_instance_ids_toml(path: Path, instance_ids: list[str]) -> None:
    lines = ["instance_ids = ["]
    lines.extend(f"  {json.dumps(instance_id)}," for instance_id in instance_ids)
    lines.append("]")
    write_text_atomically(path, "\n".join(lines) + "\n")


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trace_test_cases")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_file, mode="w", encoding=UTF8)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - instance=%(instance_id)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def instance_logger(logger: logging.Logger, instance_id: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, {"instance_id": instance_id})


def load_existing_resume_summary(
    *,
    index: int,
    total: int,
    instance_id: str,
    instance_dir: Path,
    logger: logging.LoggerAdapter,
) -> dict[str, Any] | None:
    summary_path = instance_dir / "summary.json"
    if not summary_path.is_file():
        return None

    try:
        summary = read_json(summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[Instance %d/%d] existing summary unreadable (%s); rerunning", index, total, exc)
        return None

    if not isinstance(summary, dict):
        logger.warning("[Instance %d/%d] existing summary is not a JSON object; rerunning", index, total)
        return None

    if str(summary.get("instance_id")) != instance_id:
        logger.warning(
            "[Instance %d/%d] existing summary instance_id mismatch (%s); rerunning",
            index,
            total,
            summary.get("instance_id"),
        )
        return None

    if summary.get("recovery_policy") != RECOVERY_POLICY:
        logger.info(
            "[Instance %d/%d] existing summary uses an older recovery policy; rerunning",
            index,
            total,
        )
        return None

    finalized = summary.get("finalized")
    if finalized is True:
        logger.info("[Instance %d/%d] resume skipped finalized instance", index, total)
        return summary
    if finalized is False:
        logger.info("[Instance %d/%d] resume found unfinished summary; rerunning", index, total)
        return None

    logger.info("[Instance %d/%d] resume found summary without finalized marker; rerunning", index, total)
    return None


def load_docker_dependencies() -> tuple[Any, Any, Any, Any]:
    import docker
    from swebench.harness.docker_build import build_container
    from swebench.harness.docker_utils import remove_image
    from swebench.harness.test_spec.test_spec import make_test_spec

    return docker, build_container, remove_image, make_test_spec


def build_download_request(url: str) -> Request:
    headers = {"User-Agent": "DaLIC-test-case-tracer/1.0"}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return Request(url, headers=headers)


def cache_path_for_file(cache_dir: Path, repo: str, base_commit: str, file_path: str) -> Path:
    return cache_dir / repo.replace("/", "__") / base_commit / file_path


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

    cached_file_path = cache_path_for_file(history_state["cache_dir"], repo, base_commit, file_path)
    if cached_file_path.exists():
        history_state["stats"]["disk_cache_hits"] += 1
        text = cached_file_path.read_text(encoding=UTF8)
        memory_cache[cache_key] = text
        return text

    raw_url = f"https://raw.githubusercontent.com/{repo}/{base_commit}/{file_path}"
    try:
        with urlopen(build_download_request(raw_url), timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            text = response.read().decode(UTF8, errors="replace")
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


def collect_changed_regions(patched_file: Any) -> tuple[ChangedRegion, ...]:
    added_lines: list[tuple[int, int]] = []
    removed_lines: list[tuple[int, int]] = []

    for hunk in patched_file:
        for hunk_line in hunk:
            if hunk_line.is_added and hunk_line.target_line_no is not None:
                line_no = int(hunk_line.target_line_no)
                added_lines.append((line_no, line_no))
            elif hunk_line.is_removed and hunk_line.source_line_no is not None:
                line_no = int(hunk_line.source_line_no)
                removed_lines.append((line_no, line_no))

    return tuple(
        [ChangedRegion("added", start, end) for start, end in merge_intervals(added_lines)]
        + [ChangedRegion("removed", start, end) for start, end in merge_intervals(removed_lines)]
    )


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
    return [part.strip() for part in text.strip("()").split(",") if part.strip()] or None


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
    if isinstance(resolved, (ast.List, ast.Tuple)):
        return [
            (source_for_node(module_info.source_text, elt), expr_to_case_id(elt, module_info.source_text))
            for elt in resolved.elts
        ]
    return None


def parse_pytest_param_case(expr: ast.Call, module_info: ModuleSourceInfo, expected_arity: int) -> ParamCase | None:
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
                    cases = [ParamCase(values=case.values, explicit_id=explicit_id) for case, explicit_id in zip(cases, ids)]
        result.append(DecoratorParamInfo(param_names=param_names, cases=cases))
    return result


def expanded_param_case_ids(function_node: ast.FunctionDef | ast.AsyncFunctionDef, module_info: ModuleSourceInfo) -> list[str | None]:
    decorators = extract_parametrize_info(function_node, module_info)
    if decorators is None:
        return [None]

    combinations: list[str | None] = [None]
    for decorator in decorators:
        next_combinations: list[str | None] = []
        for current_id in combinations:
            for case in decorator.cases:
                case_id = case.explicit_id or "-".join(normalize_param_id(value) for value in case.values)
                next_combinations.append(f"{current_id}-{case_id}" if current_id else case_id)
        combinations = next_combinations

    return combinations or [None]


def test_overlaps_intervals(test: TestDefinition, intervals: tuple[tuple[int, int], ...]) -> bool:
    for start, end in intervals:
        if not (test.end_line < start or end < test.start_line):
            return True
    return False


def django_test_label_from_parts(
    file_path: str,
    class_path: tuple[str, ...],
    function_name: str,
) -> str:
    module_path = file_path
    if module_path.startswith("tests/"):
        module_path = module_path[len("tests/") :]
    if module_path.endswith(".py"):
        module_path = module_path[:-3]
    module_name = module_path.replace("/", ".")
    return ".".join([module_name, *class_path, function_name])


def test_definition_is_candidate(test: TestDefinition, test_class_names: set[str]) -> bool:
    if not test.class_path:
        return True
    return any(class_name in test_class_names for class_name in test.class_path)


def collect_class_definitions(body: list[ast.stmt]) -> list[ast.ClassDef]:
    classes: list[ast.ClassDef] = []
    for node in body:
        if isinstance(node, ast.ClassDef):
            classes.append(node)
            classes.extend(collect_class_definitions(node.body))
    return classes


def collect_test_class_names(module: ast.Module) -> set[str]:
    class_nodes = collect_class_definitions(module.body)
    local_class_names = {node.name for node in class_nodes}
    base_names = {
        node.name: {
            base_name.rsplit(".", 1)[-1]
            for base in node.bases
            if (base_name := dotted_name(base)) is not None
        }
        for node in class_nodes
    }
    test_class_names = {
        node.name
        for node in class_nodes
        if node.name.startswith("Test")
        or node.name.endswith(("Test", "Tests", "TestCase"))
        or any(base.endswith(("Test", "Tests", "TestCase")) for base in base_names[node.name])
    }

    changed = True
    while changed:
        changed = False
        for class_name, bases in base_names.items():
            local_bases = bases & local_class_names
            if class_name in test_class_names:
                new_bases = local_bases - test_class_names
                if new_bases:
                    test_class_names.update(new_bases)
                    changed = True
            elif local_bases & test_class_names:
                test_class_names.add(class_name)
                changed = True

    return test_class_names


def collect_test_definitions(
    body: list[ast.stmt],
    file_path: str,
    class_path: tuple[str, ...],
    order_counter: list[int],
) -> list[TestDefinition]:
    collected: list[TestDefinition] = []
    for node in body:
        if isinstance(node, ast.ClassDef):
            collected.extend(collect_test_definitions(node.body, file_path, (*class_path, node.name), order_counter))
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            decorator_lines = [int(decorator.lineno) for decorator in node.decorator_list]
            collected.append(
                TestDefinition(
                    file_path=file_path,
                    class_path=class_path,
                    function_name=node.name,
                    start_line=min([int(node.lineno), *decorator_lines]),
                    end_line=int(getattr(node, "end_lineno", node.lineno)),
                    order_index=order_counter[0],
                    node=node,
                )
            )
            order_counter[0] += 1
    return collected


def build_module_source_info(
    file_path: str,
    base_source_text: str,
    source_text: str,
    changed_regions: tuple[ChangedRegion, ...],
    patch_order: int,
) -> ModuleSourceInfo:
    module = ast.parse(source_text, filename=file_path)
    base_tests: list[TestDefinition] = []
    base_test_class_names: set[str] = set()
    if any(region.kind == "removed" for region in changed_regions):
        base_module = ast.parse(base_source_text, filename=file_path)
        base_tests = collect_test_definitions(base_module.body, file_path, (), [0])
        base_test_class_names = collect_test_class_names(base_module)

    bindings: dict[str, ast.AST] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            bindings[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            bindings[node.target.id] = node.value

    return ModuleSourceInfo(
        file_path=file_path,
        source_text=source_text,
        module=module,
        bindings=bindings,
        tests=collect_test_definitions(module.body, file_path, (), [0]),
        test_class_names=collect_test_class_names(module),
        base_tests=base_tests,
        base_test_class_names=base_test_class_names,
        changed_regions=changed_regions,
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

    if file_diff.patched_file.is_added_file:
        base_text = ""
    else:
        if file_diff.base_lookup_path is None:
            module_cache[file_diff.file_path] = None
            return None
        base_text = load_source_text(
            repo=repo,
            base_commit=base_commit,
            file_path=file_diff.base_lookup_path,
            history_state=history_state,
        )
        if base_text is None:
            module_cache[file_diff.file_path] = None
            return None

    try:
        patched_text = apply_file_diff_to_source(base_text, file_diff.patched_file)
        module_info = build_module_source_info(
            file_path=file_diff.file_path,
            base_source_text=base_text,
            source_text=patched_text,
            changed_regions=collect_changed_regions(file_diff.patched_file),
            patch_order=file_diff.patch_order,
        )
    except (SyntaxError, ValueError):
        history_state["stats"]["test_patch_source_failures"] += 1
        module_cache[file_diff.file_path] = None
        return None

    module_cache[file_diff.file_path] = module_info
    return module_info


def select_feature_test_definitions(
    module_infos: list[ModuleSourceInfo],
) -> tuple[list[TestDefinition], int, list[dict[str, Any]]]:
    selected: list[TestDefinition] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    matched_changed_regions = 0
    unmatched_changes: list[dict[str, Any]] = []
    for module_info in sorted(module_infos, key=lambda info: info.patch_order):
        patched_tests_by_key = {test.key(): test for test in module_info.tests}
        for region in module_info.changed_regions:
            source_tests = module_info.tests if region.kind == "added" else module_info.base_tests
            source_test_class_names = (
                module_info.test_class_names
                if region.kind == "added"
                else module_info.base_test_class_names
            )
            overlapping_tests = [
                test
                for test in source_tests
                if test_overlaps_intervals(test, ((region.start_line, region.end_line),))
            ]
            candidate_tests = [
                test
                for test in overlapping_tests
                if test_definition_is_candidate(test, source_test_class_names)
            ]

            if not overlapping_tests:
                unmatched_changes.append(
                    {
                        "file_path": module_info.file_path,
                        "change_kind": region.kind,
                        "start_line": region.start_line,
                        "end_line": region.end_line,
                        "reason": "changed_region_does_not_overlap_test_definition",
                    }
                )
                continue

            if not candidate_tests:
                unmatched_changes.append(
                    {
                        "file_path": module_info.file_path,
                        "change_kind": region.kind,
                        "start_line": region.start_line,
                        "end_line": region.end_line,
                        "reason": "changed_region_only_overlaps_non_test_class_method",
                    }
                )
                continue

            if region.kind == "removed":
                missing_keys = [test.key() for test in candidate_tests if test.key() not in patched_tests_by_key]
                if missing_keys:
                    unmatched_changes.append(
                        {
                            "file_path": module_info.file_path,
                            "change_kind": region.kind,
                            "start_line": region.start_line,
                            "end_line": region.end_line,
                            "reason": "removed_region_test_not_present_after_patch",
                            "test_keys": ["::".join([key[0], *key[1], key[2]]) for key in missing_keys],
                        }
                    )
                    continue
                candidates = [patched_tests_by_key[test.key()] for test in candidate_tests]
            else:
                candidates = candidate_tests

            matched_changed_regions += 1
            for test in candidates:
                if test.key() in seen:
                    continue
                selected.append(test)
                seen.add(test.key())

    return selected, matched_changed_regions, unmatched_changes


def make_test_case_id(instance_id: str, test_case_name: str) -> str:
    digest = hashlib.sha1(f"{instance_id}::{test_case_name}".encode(UTF8)).hexdigest()[:8]
    return f"tc_{digest}"


def extract_feature_test_cases(row: dict[str, Any], history_state: dict[str, Any]) -> ExtractionResult:
    instance_id = str(get_required_field(row, "instance_id"))
    repo = str(get_required_field(row, "repo"))
    base_commit = str(get_required_field(row, "base_commit"))
    test_patch = str(row.get("test_patch") or "")

    module_cache: dict[str, ModuleSourceInfo | None] = {}
    module_infos: list[ModuleSourceInfo] = []
    changed_regions = 0
    unmatched_changes: list[dict[str, Any]] = []
    for file_path, base_lookup_path, patch_order, patched_file in parse_patch_files(test_patch, history_state):
        file_changed_regions = collect_changed_regions(patched_file)
        if not file_changed_regions:
            changed_regions += 1
            unmatched_changes.append(
                {
                    "file_path": file_path,
                    "change_kind": "metadata",
                    "start_line": None,
                    "end_line": None,
                    "reason": "file_change_has_no_changed_line_interval",
                }
            )
            continue

        changed_regions += len(file_changed_regions)
        if not file_path.endswith(".py"):
            unmatched_changes.extend(
                {
                    "file_path": file_path,
                    "change_kind": region.kind,
                    "start_line": region.start_line,
                    "end_line": region.end_line,
                    "reason": "changed_non_python_file",
                }
                for region in file_changed_regions
            )
            continue

        file_diff = PatchedPythonFile(file_path, base_lookup_path, patch_order, patched_file)
        module_info = get_or_load_patched_module_info(repo, base_commit, file_diff, history_state, module_cache)
        if module_info is None:
            reason = "removed_python_file" if patched_file.is_removed_file else "python_test_source_recovery_failed"
            unmatched_changes.extend(
                {
                    "file_path": file_path,
                    "change_kind": region.kind,
                    "start_line": region.start_line,
                    "end_line": region.end_line,
                    "reason": reason,
                }
                for region in file_changed_regions
            )
            continue
        module_infos.append(module_info)

    module_by_path = {module_info.file_path: module_info for module_info in module_infos}
    selected_tests, matched_changed_regions, module_unmatched_changes = select_feature_test_definitions(
        module_infos,
    )
    unmatched_changes.extend(module_unmatched_changes)

    test_cases: list[TestCase] = []
    seen_names: set[str] = set()
    for test in selected_tests:
        module_info = module_by_path[test.file_path]
        for param_case_id in expanded_param_case_ids(test.node, module_info):
            test_case_name = test.name(param_case_id)
            if test_case_name in seen_names:
                continue
            seen_names.add(test_case_name)
            test_cases.append(
                TestCase(
                    instance_id=instance_id,
                    test_case_id=make_test_case_id(instance_id, test_case_name),
                    test_case_name=test_case_name,
                    pytest_nodeid=test_case_name,
                    file_path=test.file_path,
                    class_path=test.class_path,
                    function_name=test.function_name,
                    param_case_id=param_case_id,
                )
            )

    return ExtractionResult(
        test_cases=test_cases,
        module_sources={module_info.file_path: module_info.source_text for module_info in module_infos},
        changed_regions=changed_regions,
        matched_changed_regions=matched_changed_regions,
        unmatched_changes=unmatched_changes,
    )


def test_case_to_jsonable(test_case: TestCase) -> dict[str, Any]:
    payload = asdict(test_case)
    payload["class_path"] = list(test_case.class_path)
    return payload


def write_test_cases_jsonl(instance_dir: Path, test_cases: list[TestCase]) -> Path:
    path = instance_dir / "test_cases.jsonl"
    instance_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=UTF8) as output_file:
        for test_case in test_cases:
            output_file.write(json.dumps(test_case_to_jsonable(test_case), ensure_ascii=False) + "\n")
    return path


def insertion_index_for_helper_import(module: ast.Module) -> int:
    index = 0
    if module.body and isinstance(module.body[0], ast.Expr) and isinstance(module.body[0].value, ast.Constant):
        if isinstance(module.body[0].value.value, str):
            index = 1
    while index < len(module.body):
        node = module.body[index]
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            index += 1
            continue
        break
    return index


def find_test_function(
    body: list[ast.stmt],
    class_path: tuple[str, ...],
    function_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    if class_path:
        class_name = class_path[0]
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return find_test_function(node.body, class_path[1:], function_name)
        raise ValueError(f"Could not find class {class_name}")

    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    raise ValueError(f"Could not find test function {function_name}")


def inject_vartracer_into_source(
    source_text: str,
    class_path: tuple[str, ...],
    function_name: str,
    output_dir: str,
) -> str:
    module = ast.parse(source_text)
    helper_import = ast.ImportFrom(
        module="_vartracer_trace_helper",
        names=[
            ast.alias(name="create_vartracer", asname="_vt_create"),
            ast.alias(name="export_vartracer", asname="_vt_export"),
        ],
        level=0,
    )
    module.body.insert(insertion_index_for_helper_import(module), helper_import)

    test_func = find_test_function(module.body, class_path, function_name)
    original_body = test_func.body
    setup_body = ast.parse("_vt = _vt_create()\n_vt.start()").body
    final_body = ast.parse(f"_vt.stop()\n_vt_export(_vt, {output_dir!r})").body
    test_func.body = [
        *setup_body,
        ast.Try(body=original_body, handlers=[], orelse=[], finalbody=final_body),
    ]

    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def build_trace_helper_source() -> str:
    return (
        "from pathlib import Path\n"
        "from VarTracer import VarTracer\n\n"
        "def create_vartracer():\n"
        f"    return VarTracer(only_project_root={DOCKER_WORKDIR!r}, clean_stdlib=True)\n\n"
        "def export_vartracer(vt, output_dir):\n"
        "    output_path = Path(output_dir)\n"
        "    output_path.mkdir(parents=True, exist_ok=True)\n"
        "    vt.exec_stack_json(output_path=str(output_path), output_name='exec_stack.json')\n"
        "    vt.dep_tree_json(output_path=str(output_path), output_name='dep_tree.json')\n"
    )


def write_instance_inputs(row: dict[str, Any], instance_dir: Path) -> dict[str, Path]:
    instance_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "row": instance_dir / "row.json",
        "patch": instance_dir / "patch.diff",
        "test_patch": instance_dir / "test_patch.diff",
        "helper": instance_dir / "_vartracer_trace_helper.py",
    }
    write_json(files["row"], row)
    files["patch"].write_text(str(row.get("patch") or ""), encoding=UTF8)
    files["test_patch"].write_text(str(row.get("test_patch") or ""), encoding=UTF8)
    files["helper"].write_text(build_trace_helper_source(), encoding=UTF8)
    return files


def load_sure_rows(input_path: Path) -> tuple[int, list[dict[str, Any]]]:
    rows = [json.loads(line) for line in input_path.open(encoding=UTF8)]
    sure_rows = [row for row in rows if row.get("ground_truth_certainty") == "sure"]
    return len(rows), sure_rows


def select_rows(rows: list[dict[str, Any]], *, test: bool, seed: int | None) -> list[dict[str, Any]]:
    if not test:
        return rows
    rng = random.Random(seed) if seed is not None else random.Random()
    return rng.sample(rows, min(TEST_SAMPLE_SIZE, len(rows)))


def run_command(command: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {shlex.join(command)}\n"
            f"{result.stdout}"
        )
    return result


def docker_exec(
    container_name: str,
    command: list[str],
    *,
    workdir: str | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    docker_command = ["docker", "exec", "-u", DOCKER_USER]
    if workdir:
        docker_command.extend(["-w", workdir])
    return run_command([*docker_command, container_name, *command], check=check, timeout=timeout)


def env_shell(command: str) -> str:
    return (
        "source /opt/miniconda3/etc/profile.d/conda.sh && "
        "conda activate testbed && "
        f"cd {DOCKER_WORKDIR} && "
        f"export PYTHONPATH={shlex.quote(DOCKER_WORKDIR)}${{PYTHONPATH:+:$PYTHONPATH}} && "
        f"{command}"
    )


def docker_shell(
    container_name: str,
    command: str,
    *,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return docker_exec(
        container_name,
        ["bash", "-lc", env_shell(command)],
        workdir=DOCKER_WORKDIR,
        check=check,
        timeout=timeout,
    )


def copy_to_container(container_name: str, source: Path, target: str) -> None:
    run_command(["docker", "cp", str(source), f"{container_name}:{target}"])


def copy_from_container(container_name: str, source: str, target: Path) -> subprocess.CompletedProcess[str]:
    target.mkdir(parents=True, exist_ok=True)
    return run_command(["docker", "cp", f"{container_name}:{source}/.", str(target)], check=False)


def django_test_label(test_case: TestCase) -> str:
    return django_test_label_from_parts(
        test_case.file_path,
        test_case.class_path,
        test_case.function_name,
    )


def pytest_test_command(test_case: TestCase, timeout: int) -> str:
    return " ".join(
        [
            f"timeout {int(timeout)}s",
            "pytest",
            shlex.quote(test_case.pytest_nodeid),
            "-q",
            "-s",
        ]
    )


def django_test_command(test_case: TestCase, timeout: int) -> str:
    return " ".join(
        [
            f"timeout {int(timeout)}s",
            "./tests/runtests.py",
            "--verbosity 2",
            "--settings=test_sqlite",
            "--parallel 1",
            shlex.quote(django_test_label(test_case)),
        ]
    )


def sphinx_test_command(test_case: TestCase, timeout: int) -> str:
    return " ".join(
        [
            f"timeout {int(timeout)}s",
            "tox",
            "--current-env",
            "-epy39",
            "-v",
            "--",
            shlex.quote(test_case.pytest_nodeid),
        ]
    )


def sympy_test_command(test_case: TestCase, timeout: int) -> str:
    return " ".join(
        [
            "PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning",
            f"timeout {int(timeout)}s",
            "bin/test",
            "-C",
            "--verbose",
            shlex.quote(test_case.file_path),
            "-k",
            shlex.quote(test_case.function_name),
        ]
    )


def build_test_command(row: dict[str, Any], test_case: TestCase, timeout: int) -> str:
    repo = row.get("repo")
    if repo == "django/django":
        return django_test_command(test_case, timeout)
    if repo == "sphinx-doc/sphinx":
        return sphinx_test_command(test_case, timeout)
    if repo == "sympy/sympy":
        return sympy_test_command(test_case, timeout)
    return pytest_test_command(test_case, timeout)


def test_case_base_name(test_case: TestCase) -> str:
    qualname = "::".join([*test_case.class_path, test_case.function_name])
    return f"{test_case.file_path}::{qualname}"


def pytest_collect_command(row: dict[str, Any], base_nodeid: str, timeout: int) -> str:
    if row.get("repo") == "sphinx-doc/sphinx":
        return " ".join(
            [
                f"timeout {int(timeout)}s",
                "tox",
                "--current-env",
                "-epy39",
                "-v",
                "--",
                shlex.quote(base_nodeid),
                "--collect-only",
                "-q",
            ]
        )

    return " ".join(
        [
            f"timeout {int(timeout)}s",
            "pytest",
            shlex.quote(base_nodeid),
            "--collect-only",
            "-q",
        ]
    )


def parse_collected_nodeids(output_text: str, base_nodeid: str) -> list[str]:
    nodeids: list[str] = []
    seen: set[str] = set()
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if line == base_nodeid or line.startswith(f"{base_nodeid}["):
            if line not in seen:
                seen.add(line)
                nodeids.append(line)
    return nodeids


def test_case_from_collected_nodeid(instance_id: str, exemplar: TestCase, base_nodeid: str, nodeid: str) -> TestCase:
    suffix = nodeid[len(base_nodeid) :]
    param_case_id = suffix[1:-1] if suffix.startswith("[") and suffix.endswith("]") else None
    return TestCase(
        instance_id=instance_id,
        test_case_id=make_test_case_id(instance_id, nodeid),
        test_case_name=nodeid,
        pytest_nodeid=nodeid,
        file_path=exemplar.file_path,
        class_path=exemplar.class_path,
        function_name=exemplar.function_name,
        param_case_id=param_case_id,
    )


def resolve_pytest_collected_test_cases(
    *,
    row: dict[str, Any],
    container_name: str,
    extraction: ExtractionResult,
    timeout: int,
    logger: logging.LoggerAdapter,
) -> list[TestCase]:
    # Django and SymPy use repo-native runners rather than pytest nodeids.
    # Every pytest-backed candidate is collected in the prepared container so
    # a name-shaped function cannot reach VarTracer unless pytest recognizes it.
    if row.get("repo") in {"django/django", "sympy/sympy"}:
        return extraction.test_cases

    instance_id = str(get_required_field(row, "instance_id"))
    grouped: dict[str, list[TestCase]] = {}
    for test_case in extraction.test_cases:
        grouped.setdefault(test_case_base_name(test_case), []).append(test_case)

    resolved: list[TestCase] = []
    seen_names: set[str] = set()
    handled_basenames: set[str] = set()
    for test_case in extraction.test_cases:
        base_nodeid = test_case_base_name(test_case)
        if base_nodeid in handled_basenames:
            continue
        handled_basenames.add(base_nodeid)

        command = pytest_collect_command(row, base_nodeid, timeout)
        collect_result = docker_shell(container_name, command, check=False, timeout=timeout + 60)
        nodeids = parse_collected_nodeids(collect_result.stdout, base_nodeid)
        if collect_result.returncode != 0 or not nodeids:
            raise RuntimeError(
                f"Failed to collect pytest nodeids for {base_nodeid}; "
                f"exit_code={collect_result.returncode}\n{collect_result.stdout}"
            )
        logger.info("collected %d pytest nodeids for %s", len(nodeids), base_nodeid)
        exemplar = grouped[base_nodeid][0]
        for nodeid in nodeids:
            if nodeid in seen_names:
                continue
            seen_names.add(nodeid)
            resolved.append(test_case_from_collected_nodeid(instance_id, exemplar, base_nodeid, nodeid))

    return resolved


def apply_patch_file(container_name: str, patch_file: Path) -> None:
    if not patch_file.read_text(encoding=UTF8).strip():
        return
    container_path = f"/tmp/{patch_file.name}"
    copy_to_container(container_name, patch_file, container_path)
    docker_exec(container_name, ["git", "apply", "-v", container_path], workdir=DOCKER_WORKDIR)


def docker_image_size_bytes() -> int:
    output = run_command(["docker", "system", "df", "--format", "{{json .}}"]).stdout
    for line in output.splitlines():
        row = json.loads(line)
        if row.get("Type") != "Images":
            continue
        match = re.fullmatch(r"\s*([\d.]+)\s*([kmgtp]?i?b)\s*", row["Size"], re.IGNORECASE)
        if match is None:
            raise ValueError(f"Unsupported Docker image size: {row['Size']}")
        unit = match.group(2).lower().replace("i", "")
        factors = {"b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4, "pb": 1000**5}
        return int(float(match.group(1)) * factors[unit])
    raise RuntimeError("docker system df did not report image usage")


def official_image_exists(client: Any, image_key: str) -> bool:
    docker_module, _, _, _ = load_docker_dependencies()
    try:
        client.images.get(image_key)
    except docker_module.errors.ImageNotFound:
        return False
    return True


def container_python_version(container_name: str) -> tuple[int, int, int]:
    result = docker_shell(
        container_name,
        "python -c 'import sys; print(\"%d.%d.%d\" % sys.version_info[:3])'",
    )
    version_text = result.stdout.strip().splitlines()[-1]
    major, minor, micro = (int(part) for part in version_text.split("."))
    return major, minor, micro


def install_vartracer(container_name: str, wheel_path: Path) -> None:
    python_version = container_python_version(container_name)
    if python_version < (3, 6, 0):
        version_text = ".".join(str(part) for part in python_version)
        raise RuntimeError(f"Container Python {version_text} is too old for VarTracer instrumentation")

    container_wheel = f"/tmp/{wheel_path.name}"
    copy_to_container(container_name, wheel_path, container_wheel)
    docker_shell(
        container_name,
        f"python -m pip install --no-deps --force-reinstall --ignore-requires-python {shlex.quote(container_wheel)}",
    )


def prepare_container(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    client: Any,
    files: dict[str, Path],
    logger: logging.LoggerAdapter,
) -> Any:
    _, build_container, _, make_test_spec = load_docker_dependencies()
    test_spec = make_test_spec(row, namespace=OFFICIAL_NAMESPACE)
    if official_image_exists(client, test_spec.instance_image_key):
        logger.info("using local official historical image; image=%s", test_spec.instance_image_key)
    else:
        logger.info(
            "official historical image is not available locally; delegating to SWE-bench build_container; image=%s",
            test_spec.instance_image_key,
        )

    logger.info("creating official historical container; image=%s", test_spec.instance_image_key)
    container = None
    try:
        container = build_container(
            test_spec=test_spec,
            client=client,
            run_id=args.run_id,
            logger=logger,
            nocache=False,
            force_rebuild=False,
        )
        run_command(["docker", "start", container.name])

        install_vartracer(container.name, args.vartracer_wheel)
        copy_to_container(container.name, files["helper"], f"{DOCKER_WORKDIR}/_vartracer_trace_helper.py")

        base_commit = str(get_required_field(row, "base_commit"))
        docker_exec(container.name, ["git", "checkout", "-f", base_commit], workdir=DOCKER_WORKDIR)
        apply_patch_file(container.name, files["patch"])
        apply_patch_file(container.name, files["test_patch"])
        return container, test_spec.instance_image_key
    except Exception:
        if container is not None:
            run_command(["docker", "rm", "-f", container.name], check=False)
        raise


def run_traced_test_case(
    *,
    row: dict[str, Any],
    container_name: str,
    test_case: TestCase,
    module_sources: dict[str, str],
    instance_dir: Path,
    timeout: int,
    logger: logging.LoggerAdapter,
) -> dict[str, Any]:
    test_dir = instance_dir / test_case.test_case_id
    container_output_dir = f"/tmp/vartracer_results/{test_case.test_case_id}"
    instrumented_source = inject_vartracer_into_source(
        module_sources[test_case.file_path],
        test_case.class_path,
        test_case.function_name,
        container_output_dir,
    )

    test_dir.mkdir(parents=True, exist_ok=True)
    instrumented_path = test_dir / "instrumented_test.py"
    instrumented_path.write_text(instrumented_source, encoding=UTF8)
    copy_to_container(container_name, instrumented_path, f"{DOCKER_WORKDIR}/{test_case.file_path}")

    # The same test file can be overwritten several times in quick succession.
    # Remove its bytecode cache right here so Python cannot reuse stale pyc for
    # the next test case and accidentally skip the current VarTracer export.
    container_test_file = Path(DOCKER_WORKDIR) / test_case.file_path
    container_pycache_dir = container_test_file.parent / "__pycache__"
    docker_exec(container_name, ["rm", "-rf", str(container_pycache_dir)], workdir=DOCKER_WORKDIR, check=False)

    try:
        instrumented_path.unlink()
    except OSError:
        pass

    command = build_test_command(row, test_case, timeout)

    metadata: dict[str, Any] = {
        **test_case_to_jsonable(test_case),
        "runtime_seconds": None,
        "exit_code": None,
        "timed_out": False,
        "stdout_path": str(test_dir / "stdout.txt"),
        "copy_result_exit_code": None,
        "copy_result_skipped_reason": None,
        "vartracer_only_project_root": DOCKER_WORKDIR,
        "command": command,
    }

    start_time = time.monotonic()
    try:
        result = docker_shell(container_name, command, check=False, timeout=timeout + 60)
        output_text = result.stdout
        metadata["exit_code"] = result.returncode
        metadata["timed_out"] = result.returncode == 124
    except subprocess.TimeoutExpired as exc:
        output_text = exc.stdout or ""
        if isinstance(output_text, bytes):
            output_text = output_text.decode(UTF8, errors="replace")
        metadata["timed_out"] = True
    metadata["runtime_seconds"] = time.monotonic() - start_time

    (test_dir / "stdout.txt").write_text(output_text, encoding=UTF8)

    # A failed or timed-out test can leave an enormous partial VarTracer dump.
    # Keep the failure metadata/stdout, but do not copy partial trace artifacts
    # back to the host; clean them inside the container before moving on.
    if metadata["exit_code"] == 0 and not metadata["timed_out"]:
        copy_result = copy_from_container(container_name, container_output_dir, test_dir)
        metadata["copy_result_exit_code"] = copy_result.returncode
    else:
        metadata["copy_result_skipped_reason"] = "test_timed_out_or_failed"
        docker_exec(container_name, ["rm", "-rf", container_output_dir], workdir=DOCKER_WORKDIR, check=False)

    write_json(test_dir / "metadata.json", metadata)
    logger.info(
        "traced test_case_id=%s exit_code=%s timed_out=%s",
        test_case.test_case_id,
        metadata["exit_code"],
        metadata["timed_out"],
    )
    return metadata


def run_instance(
    *,
    index: int,
    total: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    history_state: dict[str, Any],
    client: Any,
    logger: logging.LoggerAdapter,
    extraction: ExtractionResult | None = None,
) -> dict[str, Any]:
    instance_id = str(get_required_field(row, "instance_id"))
    instance_dir = args.output_dir / instance_id
    summary: dict[str, Any] = {
        "instance_id": instance_id,
        "status": "completed",
        "finalized": False,
        "recovery_policy": RECOVERY_POLICY,
        "changed_regions": 0,
        "matched_changed_regions": 0,
        "unmatched_changed_regions": 0,
        "unmatched_changes": [],
        "test_cases": 0,
        "completed_test_cases": 0,
        "failed_test_cases": 0,
        "timed_out_test_cases": 0,
        "results": [],
    }
    container = None
    image_key: str | None = None

    try:
        files = write_instance_inputs(row, instance_dir)
        if extraction is None:
            logger.info("[Instance %d/%d] extracting feature-positive test cases", index, total)
            extraction = extract_feature_test_cases(row, history_state)
        else:
            logger.info("[Instance %d/%d] using pre-screened test cases", index, total)
        summary["test_cases"] = len(extraction.test_cases)
        summary["changed_regions"] = extraction.changed_regions
        summary["matched_changed_regions"] = extraction.matched_changed_regions
        summary["unmatched_changed_regions"] = extraction.unmatched_changed_regions
        summary["unmatched_changes"] = extraction.unmatched_changes

        if not extraction.test_cases:
            summary["status"] = "no_test_cases"
            summary["finalized"] = True
            test_cases_path = write_test_cases_jsonl(instance_dir, [])
            summary["test_cases_path"] = str(test_cases_path)
            write_json(instance_dir / "summary.json", summary)
            return summary

        if not extraction.fully_recovered:
            summary["status"] = "test_cases_not_fully_recovered"
            summary["finalized"] = True
            test_cases_path = write_test_cases_jsonl(instance_dir, extraction.test_cases)
            summary["test_cases_path"] = str(test_cases_path)
            write_json(instance_dir / "summary.json", summary)
            return summary

        logger.info("[Instance %d/%d] preparing Docker container", index, total)
        container, image_key = prepare_container(row=row, args=args, client=client, files=files, logger=logger)
        test_cases = resolve_pytest_collected_test_cases(
            row=row,
            container_name=container.name,
            extraction=extraction,
            timeout=args.timeout,
            logger=logger,
        )
        test_cases_path = write_test_cases_jsonl(instance_dir, test_cases)
        summary["test_cases"] = len(test_cases)
        summary["test_cases_path"] = str(test_cases_path)
        if len(test_cases) > MAX_TEST_CASES_PER_INSTANCE:
            summary["status"] = "failed"
            summary["finalized"] = True
            summary["error"] = (
                f"too_many_test_cases: {len(test_cases)} > {MAX_TEST_CASES_PER_INSTANCE}"
            )
            logger.warning(
                "[Instance %d/%d] skipping trace because collected test_cases=%d exceeds limit=%d",
                index,
                total,
                len(test_cases),
                MAX_TEST_CASES_PER_INSTANCE,
            )
            return summary

        for test_case in test_cases:
            result = run_traced_test_case(
                row=row,
                container_name=container.name,
                test_case=test_case,
                module_sources=extraction.module_sources,
                instance_dir=instance_dir,
                timeout=args.timeout,
                logger=logger,
            )
            summary["results"].append(result)
            if result["timed_out"]:
                summary["timed_out_test_cases"] += 1
            elif result["exit_code"] == 0 and result["copy_result_exit_code"] == 0:
                summary["completed_test_cases"] += 1
            else:
                summary["failed_test_cases"] += 1

        if summary["failed_test_cases"] or summary["timed_out_test_cases"]:
            summary["status"] = "failed"
        summary["finalized"] = True

    except Exception as exc:
        summary["status"] = "failed"
        summary["finalized"] = True
        summary["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("[Instance %d/%d] failed", index, total)
    except BaseException as exc:
        summary["status"] = "interrupted"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("[Instance %d/%d] interrupted", index, total)
        raise
    finally:
        if container is not None:
            run_command(["docker", "rm", "-f", container.name], check=False)
        if image_key is not None:
            try:
                image_size = docker_image_size_bytes()
                summary["docker_image_size_bytes"] = image_size
                if image_size > IMAGE_LIMIT_BYTES:
                    _, _, remove_image, _ = load_docker_dependencies()
                    remove_image(client, image_key, "quiet")
                    summary["removed_instance_image"] = image_key
            except Exception as exc:
                summary["docker_cleanup_error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("Docker cleanup check failed")
        write_json(instance_dir / "summary.json", summary)

    return summary


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
            "test_patch_source_failures": 0,
        },
    }


def screen_trace_instances(
    rows: list[dict[str, Any]],
    history_state: dict[str, Any],
) -> ScreeningResult:
    extractions_by_instance_id: dict[str, ExtractionResult] = {}
    eligible_instance_ids: list[str] = []
    category_counts = {
        "trace_eligible": 0,
        "no_test_cases": 0,
        "test_cases_not_fully_recovered": 0,
    }

    for row in tqdm(rows, desc="screen_trace_eligible", unit="instance"):
        instance_id = str(get_required_field(row, "instance_id"))
        extraction = extract_feature_test_cases(row, history_state)
        extractions_by_instance_id[instance_id] = extraction
        if not extraction.test_cases:
            category_counts["no_test_cases"] += 1
        elif not extraction.fully_recovered:
            category_counts["test_cases_not_fully_recovered"] += 1
        else:
            category_counts["trace_eligible"] += 1
            eligible_instance_ids.append(instance_id)

    return ScreeningResult(
        extractions_by_instance_id=extractions_by_instance_id,
        eligible_instance_ids=eligible_instance_ids,
        category_counts=category_counts,
    )


def select_trace_eligible_instance_ids(
    rows: list[dict[str, Any]],
    history_state: dict[str, Any],
) -> tuple[list[str], dict[str, int]]:
    screening = screen_trace_instances(rows, history_state)
    return screening.eligible_instance_ids, screening.category_counts


def main() -> None:
    repo_root = get_repo_root()
    args = parse_args(repo_root)
    os.chdir(repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.output_dir / "trace_test_cases.log"
    logger = setup_logger(log_file)
    batch_logger = instance_logger(logger, "batch")
    print(f"workflow_log = {log_file}")

    try:
        total_rows, sure_rows = load_sure_rows(args.input)
        scheduled_rows = select_rows(sure_rows, test=bool(args.test), seed=args.seed)
        batch_logger.info(
            "loaded dataset_rows=%d sure_instances=%d scheduled_instances=%d test_mode=%s",
            total_rows,
            len(sure_rows),
            len(scheduled_rows),
            bool(args.test),
        )
        if args.resume:
            batch_logger.info("resume enabled")

        history_state = build_history_state(args.cache_dir)
        screening = screen_trace_instances(
            scheduled_rows,
            history_state,
        )
        trace_eligible_instance_ids = screening.eligible_instance_ids
        screening_category_counts = screening.category_counts
        screening_history_stats = dict(history_state["stats"])
        trace_eligible_path = repo_root / "dataset" / TRACE_ELIGIBLE_TOML_NAME
        write_instance_ids_toml(trace_eligible_path, trace_eligible_instance_ids)
        batch_logger.info(
            "screening completed trace_eligible=%d no_test_cases=%d "
            "test_cases_not_fully_recovered=%d output=%s",
            screening_category_counts["trace_eligible"],
            screening_category_counts["no_test_cases"],
            screening_category_counts["test_cases_not_fully_recovered"],
            trace_eligible_path,
        )

        docker_module, _, _, _ = load_docker_dependencies()
        client = docker_module.from_env()
        summaries: list[dict[str, Any]] = []
        for index, row in enumerate(tqdm(scheduled_rows, desc="trace_test_cases", unit="instance"), start=1):
            instance_id = str(row["instance_id"])
            extraction = screening.extractions_by_instance_id.pop(instance_id)
            logger_adapter = instance_logger(logger, instance_id)
            if args.resume:
                existing_summary = load_existing_resume_summary(
                    index=index,
                    total=len(scheduled_rows),
                    instance_id=instance_id,
                    instance_dir=args.output_dir / instance_id,
                    logger=logger_adapter,
                )
                if existing_summary is not None:
                    summaries.append(existing_summary)
                    continue
            summaries.append(
                run_instance(
                    index=index,
                    total=len(scheduled_rows),
                    row=row,
                    args=args,
                    history_state=history_state,
                    client=client,
                    logger=logger_adapter,
                    extraction=extraction,
                )
            )

        batch_summary = {
            "dataset_rows": total_rows,
            "sure_instances": len(sure_rows),
            "scheduled_instances": len(scheduled_rows),
            "trace_eligible_instances": len(trace_eligible_instance_ids),
            "trace_eligible_instances_path": str(trace_eligible_path),
            "completed_instances": sum(summary["status"] == "completed" for summary in summaries),
            "failed_instances": sum(summary["status"] == "failed" for summary in summaries),
            "no_test_case_instances": sum(summary["status"] == "no_test_cases" for summary in summaries),
            "test_cases_not_fully_recovered_instances": sum(
                summary["status"] == "test_cases_not_fully_recovered" for summary in summaries
            ),
            "screening_history_stats": screening_history_stats,
            "history_stats": history_state["stats"],
            "instances": summaries,
        }
        write_json(args.output_dir / "summary.json", batch_summary)
        completed_instance_ids = [
            str(summary["instance_id"])
            for summary in summaries
            if summary["status"] == "completed"
        ]
        write_instance_ids_toml(
            repo_root / "dataset" / TRACE_COMPLETED_TOML_NAME,
            completed_instance_ids,
        )
        print(json.dumps({key: value for key, value in batch_summary.items() if key != "instances"}, ensure_ascii=False, indent=2))
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
