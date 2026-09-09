#!/usr/bin/env python3
"""Merge a complete retry run into an earlier experiment result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


METHODS = ("baseline", "dalic")
DEFAULT_SELECTED_INSTANCES = Path("dataset/trace_completed_instance.toml")
DEFAULT_TRUTH_INPUT = Path("dataset/swebench_feature_with_truth.jsonl")
DEFAULT_ORIGINAL_ERROR_MARKER = "credit_balance_exhausted"
INSTANCE_START_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} .*? \w+ \[([^]]+)] Starting (?:baseline|dalic) flow\.\s*$"
)


class MergeError(RuntimeError):
    """Raised when source runs are unsafe or incomplete for aggregation."""


@dataclass(frozen=True)
class RunSummary:
    success_count: int
    failure_count: int
    failed_instances: tuple[str, ...]
    failure_reasons: dict[str, str]


@dataclass(frozen=True)
class MergeConfig:
    repo_root: Path
    method: str
    original_root: Path
    retry_root: Path
    output_root: Path
    retry_instances_path: Path
    selected_instances_path: Path
    truth_input_path: Path
    workbench_path: Path
    original_error_marker: str | None


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise MergeError("Could not locate the repository root.")


def resolve_path(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def validate_model_component(value: str, option_name: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise MergeError(f"{option_name} must be one safe directory name: {value!r}")
    return value


def load_instance_ids(path: Path, description: str) -> list[str]:
    try:
        with path.open("rb") as input_file:
            payload = tomllib.load(input_file)
    except FileNotFoundError as exc:
        raise MergeError(f"{description} file is missing: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise MergeError(f"{description} is not valid TOML: {path}: {exc}") from exc

    instance_ids = payload.get("instance_ids")
    if not isinstance(instance_ids, list) or any(
        not isinstance(instance_id, str) or not instance_id
        for instance_id in instance_ids
    ):
        raise MergeError(
            f"{description} must define instance_ids as a list of nonempty strings: {path}"
        )
    duplicates = sorted(
        instance_id
        for instance_id, count in Counter(instance_ids).items()
        if count > 1
    )
    if duplicates:
        raise MergeError(f"{description} contains duplicate instance IDs: {duplicates}")
    return instance_ids


def parse_summary(path: Path) -> RunSummary:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise MergeError(f"Run summary is missing: {path}") from exc

    try:
        success_count = int(lines[0].removeprefix("success_count: "))
        failure_count = int(lines[1].removeprefix("failure_count: "))
        if lines[2] != "failed_instances:":
            raise ValueError("missing failed_instances header")
        reasons_index = lines.index("failure_reasons:", 3)
    except (IndexError, ValueError) as exc:
        raise MergeError(f"Run summary has an invalid structure: {path}") from exc

    failed_lines = lines[3:reasons_index]
    failed_instances = tuple(() if failed_lines == ["None"] else failed_lines)
    reason_lines = lines[reasons_index + 1 :]
    failure_reasons: dict[str, str] = {}
    if reason_lines != ["None"]:
        for line in reason_lines:
            if ": " not in line:
                raise MergeError(f"Invalid failure reason in {path}: {line!r}")
            instance_id, reason = line.split(": ", 1)
            failure_reasons[instance_id] = reason

    if success_count < 0 or failure_count < 0:
        raise MergeError(f"Run summary contains a negative count: {path}")
    if len(failed_instances) != failure_count:
        raise MergeError(
            f"Run summary failure_count={failure_count}, but lists "
            f"{len(failed_instances)} failed instances: {path}"
        )
    if len(set(failed_instances)) != len(failed_instances):
        raise MergeError(f"Run summary contains duplicate failed instances: {path}")
    if set(failure_reasons) != set(failed_instances):
        raise MergeError(
            f"Run summary failure reasons do not match failed instances: {path}"
        )
    return RunSummary(
        success_count=success_count,
        failure_count=failure_count,
        failed_instances=failed_instances,
        failure_reasons=failure_reasons,
    )


def instance_directories(root: Path) -> set[str]:
    if not root.is_dir():
        raise MergeError(f"Result directory is missing: {root}")
    return {path.name for path in root.iterdir() if path.is_dir() and path.name != "_provenance"}


def validate_complete_run(
    root: Path,
    expected_instance_ids: set[str],
    description: str,
) -> RunSummary:
    summary = parse_summary(root / "summary.txt")
    completed_count = summary.success_count + summary.failure_count
    if completed_count != len(expected_instance_ids):
        raise MergeError(
            f"{description} is incomplete: summary accounts for {completed_count} of "
            f"{len(expected_instance_ids)} expected instances"
        )

    actual_directories = instance_directories(root)
    missing = sorted(expected_instance_ids - actual_directories)
    extra = sorted(actual_directories - expected_instance_ids)
    if missing or extra:
        raise MergeError(
            f"{description} instance directories do not match the expected set; "
            f"missing={missing}, extra={extra}"
        )
    unknown_failures = sorted(set(summary.failed_instances) - expected_instance_ids)
    if unknown_failures:
        raise MergeError(
            f"{description} summary contains unexpected failed instances: {unknown_failures}"
        )
    return summary


def failed_instances_with_marker(
    root: Path,
    summary: RunSummary,
    marker: str,
) -> set[str]:
    matched: set[str] = set()
    for instance_id in summary.failed_instances:
        log_path = root / instance_id / "docker_run.log"
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError as exc:
            raise MergeError(
                f"Cannot verify original error marker because a log is missing: {log_path}"
            ) from exc
        if marker in log_text:
            matched.add(instance_id)
    return matched


def extract_instance_log_blocks(path: Path) -> dict[str, list[str]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError as exc:
        raise MergeError(f"Run log is missing: {path}") from exc

    blocks: dict[str, list[str]] = {}
    current_id: str | None = None
    for line in lines:
        match = INSTANCE_START_RE.match(line)
        if match:
            current_id = match.group(1)
            if current_id in blocks:
                raise MergeError(f"Duplicate log block for {current_id} in {path}")
            blocks[current_id] = []
        if current_id is not None and " Run finished. success_count=" not in line:
            blocks[current_id].append(line)
    return blocks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_file_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    manifest: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise MergeError(f"Symbolic links are not allowed in result directories: {path}")
        if path.is_file():
            relative_path = path.relative_to(root).as_posix()
            manifest[relative_path] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return manifest


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def build_config(args: argparse.Namespace) -> MergeConfig:
    repo_root = get_repo_root().resolve()
    original_model = validate_model_component(
        args.original_model_name, "--original-model-name"
    )
    retry_model = validate_model_component(args.retry_model_name, "--retry-model-name")
    output_model = validate_model_component(
        args.output_model_name or f"{original_model}_merged", "--output-model-name"
    )
    method_root = repo_root / "dataset" / "agent_results" / args.method
    workbench_path = args.workbench_path or Path(f"dataset/agent_workbench/{args.method}")
    return MergeConfig(
        repo_root=repo_root,
        method=args.method,
        original_root=method_root / original_model,
        retry_root=method_root / retry_model,
        output_root=method_root / output_model,
        retry_instances_path=resolve_path(repo_root, args.retry_instances),
        selected_instances_path=resolve_path(repo_root, args.selected_instances),
        truth_input_path=resolve_path(repo_root, args.input),
        workbench_path=resolve_path(repo_root, workbench_path),
        original_error_marker=(
            None if args.no_original_error_check else args.original_error_marker
        ),
    )


def preflight(config: MergeConfig) -> tuple[list[str], set[str], RunSummary, RunSummary]:
    selected_ids = load_instance_ids(config.selected_instances_path, "Selected instances")
    retry_ids_list = load_instance_ids(config.retry_instances_path, "Retry instances")
    selected_set = set(selected_ids)
    retry_set = set(retry_ids_list)

    outside_benchmark = sorted(retry_set - selected_set)
    if outside_benchmark:
        raise MergeError(f"Retry instances are outside the selected benchmark: {outside_benchmark}")

    original_summary = validate_complete_run(
        config.original_root, selected_set, "Original run"
    )
    if not retry_set <= set(original_summary.failed_instances):
        unexpected = sorted(retry_set - set(original_summary.failed_instances))
        raise MergeError(
            f"Retry list would replace instances that succeeded in the original run: {unexpected}"
        )

    if config.original_error_marker is not None:
        marker_matches = failed_instances_with_marker(
            config.original_root, original_summary, config.original_error_marker
        )
        missing = sorted(marker_matches - retry_set)
        extra = sorted(retry_set - marker_matches)
        if missing or extra:
            raise MergeError(
                f"Retry list does not exactly match original failures containing "
                f"{config.original_error_marker!r}; missing={missing}, extra={extra}"
            )

    retry_summary = validate_complete_run(config.retry_root, retry_set, "Retry run")
    return sorted(selected_set), retry_set, original_summary, retry_summary


def render_summary(
    selected_ids: list[str],
    retry_ids: set[str],
    original_summary: RunSummary,
    retry_summary: RunSummary,
) -> tuple[str, set[str]]:
    original_failed = set(original_summary.failed_instances)
    retry_failed = set(retry_summary.failed_instances)
    final_failed: list[str] = []
    reasons: dict[str, str] = {}
    for instance_id in selected_ids:
        if instance_id in retry_ids:
            if instance_id in retry_failed:
                final_failed.append(instance_id)
                reasons[instance_id] = retry_summary.failure_reasons[instance_id]
        elif instance_id in original_failed:
            final_failed.append(instance_id)
            reasons[instance_id] = original_summary.failure_reasons[instance_id]

    lines = [
        f"success_count: {len(selected_ids) - len(final_failed)}",
        f"failure_count: {len(final_failed)}",
        "failed_instances:",
    ]
    lines.extend(final_failed or ["None"])
    lines.append("failure_reasons:")
    if final_failed:
        lines.extend(f"{instance_id}: {reasons[instance_id]}" for instance_id in final_failed)
    else:
        lines.append("None")
    return "\n".join(lines) + "\n", set(final_failed)


def render_aggregate_log(
    config: MergeConfig,
    selected_ids: list[str],
    retry_ids: set[str],
    original_blocks: dict[str, list[str]],
    retry_blocks: dict[str, list[str]],
    success_count: int,
    failure_count: int,
) -> str:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"{timestamp} INFO Aggregated {config.method} log created from original and retry runs.",
        f"{timestamp} INFO Original results root: {config.original_root}",
        f"{timestamp} INFO Retry results root: {config.retry_root}",
        f"{timestamp} INFO Retry instances path: {config.retry_instances_path}",
        f"{timestamp} INFO Replacement policy: every listed retry instance uses the retry attempt.",
    ]
    for instance_id in selected_ids:
        blocks = retry_blocks if instance_id in retry_ids else original_blocks
        source_name = "retry" if instance_id in retry_ids else "original"
        block = blocks.get(instance_id)
        if not block:
            raise MergeError(
                f"Missing {source_name} root-log block for selected instance {instance_id}"
            )
        lines.extend(block)
    lines.append(
        f"{timestamp} INFO Aggregated run finished. success_count={success_count} "
        f"failure_count={failure_count}"
    )
    return "\n".join(lines) + "\n"


def collect_trajectory_totals(output_root: Path, selected_ids: list[str]) -> dict[str, object]:
    total_cost = 0.0
    total_api_calls = 0
    exit_status_counts: Counter[str] = Counter()
    missing_or_invalid: list[str] = []
    for instance_id in selected_ids:
        trajectory_path = output_root / instance_id / "last_mini_run.traj.json"
        try:
            payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
            info = payload.get("info", {})
            model_stats = info.get("model_stats", {})
            total_cost += float(model_stats.get("instance_cost", 0.0))
            total_api_calls += int(model_stats.get("api_calls", 0))
            exit_status_counts[str(info.get("exit_status", "<missing>"))] += 1
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            missing_or_invalid.append(instance_id)
    return {
        "trajectory_count": len(selected_ids) - len(missing_or_invalid),
        "recorded_instance_cost_sum": round(total_cost, 12),
        "api_calls_sum": total_api_calls,
        "exit_status_counts": dict(sorted(exit_status_counts.items())),
        "missing_or_invalid_trajectories": missing_or_invalid,
    }


def run_evaluation(config: MergeConfig, output_root: Path) -> None:
    command = [
        sys.executable,
        str(config.repo_root / "scripts" / "evaluate.py"),
        "--input",
        str(config.truth_input_path),
        "--selected_instances",
        str(config.selected_instances_path),
        "--workbench_path",
        str(config.workbench_path),
        "--result_folder",
        str(output_root),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        raise MergeError(f"Evaluation failed with exit code {completed.returncode}: {detail}")


def evaluation_counts(path: Path) -> tuple[int, int, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        for key in ("sure_ground_truth_instances", "successful_instances", "failed_instances"):
            prefix = f"{key}: "
            if line.startswith(prefix) and key not in values:
                values[key] = int(line.removeprefix(prefix))
    missing = sorted(
        {"sure_ground_truth_instances", "successful_instances", "failed_instances"} - set(values)
    )
    if missing:
        raise MergeError(f"Evaluation output is missing count fields {missing}: {path}")
    return (
        values["sure_ground_truth_instances"],
        values["successful_instances"],
        values["failed_instances"],
    )


def merge_results(
    config: MergeConfig,
    evaluator: Callable[[MergeConfig, Path], None] = run_evaluation,
) -> Path:
    selected_ids, retry_ids, original_summary, retry_summary = preflight(config)
    if config.output_root.exists():
        raise MergeError(
            f"Output directory already exists and will not be overwritten: {config.output_root}"
        )
    config.output_root.parent.mkdir(parents=True, exist_ok=True)

    original_log_path = config.original_root / f"{config.method}.log"
    retry_log_path = config.retry_root / f"{config.method}.log"
    original_blocks = extract_instance_log_blocks(original_log_path)
    retry_blocks = extract_instance_log_blocks(retry_log_path)
    summary_text, final_failed = render_summary(
        selected_ids, retry_ids, original_summary, retry_summary
    )
    final_success_count = len(selected_ids) - len(final_failed)

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_root.name}.tmp-",
            dir=config.output_root.parent,
        )
    )
    try:
        instance_manifest: dict[str, object] = {}
        for instance_id in selected_ids:
            use_retry = instance_id in retry_ids
            source_root = config.retry_root if use_retry else config.original_root
            source_dir = source_root / instance_id
            expected_files = directory_file_manifest(source_dir)
            destination_dir = temporary_root / instance_id
            shutil.copytree(source_dir, destination_dir, symlinks=True)
            copied_files = directory_file_manifest(destination_dir)
            if copied_files != expected_files:
                raise MergeError(f"Copied files failed hash verification for {instance_id}")
            instance_manifest[instance_id] = {
                "chosen_attempt": "retry" if use_retry else "original",
                "source_directory": display_path(source_dir, config.repo_root),
                "files": expected_files,
            }

        (temporary_root / "summary.txt").write_text(summary_text, encoding="utf-8")
        provenance_root = temporary_root / "_provenance"
        provenance_root.mkdir()
        shutil.copy2(original_log_path, provenance_root / f"{config.method}.original.log")
        shutil.copy2(retry_log_path, provenance_root / f"{config.method}.retry.log")
        shutil.copy2(config.original_root / "summary.txt", provenance_root / "summary.original.txt")
        shutil.copy2(config.retry_root / "summary.txt", provenance_root / "summary.retry.txt")
        shutil.copy2(config.retry_instances_path, provenance_root / "retry_instances.toml")

        aggregate_log = render_aggregate_log(
            config,
            selected_ids,
            retry_ids,
            original_blocks,
            retry_blocks,
            final_success_count,
            len(final_failed),
        )
        (temporary_root / f"{config.method}.log").write_text(
            aggregate_log, encoding="utf-8"
        )

        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "method": config.method,
            "replacement_policy": (
                "Every retry-list instance uses the retry attempt regardless of outcome; "
                "all other instances use the original attempt."
            ),
            "selected_instances_path": display_path(
                config.selected_instances_path, config.repo_root
            ),
            "retry_instances_path": display_path(
                config.retry_instances_path, config.repo_root
            ),
            "original_results_root": display_path(config.original_root, config.repo_root),
            "retry_results_root": display_path(config.retry_root, config.repo_root),
            "output_results_root": display_path(config.output_root, config.repo_root),
            "instance_count": len(selected_ids),
            "retry_instance_count": len(retry_ids),
            "final_success_count": final_success_count,
            "final_failure_count": len(final_failed),
            "original_error_marker": config.original_error_marker,
            "instances": instance_manifest,
        }
        manifest["trajectory_totals"] = collect_trajectory_totals(
            temporary_root, selected_ids
        )
        (temporary_root / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        evaluator(config, temporary_root)
        evaluation_total, evaluation_success, evaluation_failure = evaluation_counts(
            temporary_root / "evaluation.txt"
        )
        expected_counts = (len(selected_ids), final_success_count, len(final_failed))
        actual_counts = (evaluation_total, evaluation_success, evaluation_failure)
        if actual_counts != expected_counts:
            raise MergeError(
                f"Summary and evaluation counts disagree; summary={expected_counts}, "
                f"evaluation={actual_counts}"
            )

        temporary_root.rename(config.output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return config.output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a standalone result directory by replacing infrastructure failures "
            "with a completed retry run. Source directories are never modified."
        )
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--original-model-name", required=True)
    parser.add_argument("--retry-model-name", required=True)
    parser.add_argument("--output-model-name")
    parser.add_argument("--retry-instances", type=Path, required=True)
    parser.add_argument(
        "--selected-instances", type=Path, default=DEFAULT_SELECTED_INSTANCES
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_TRUTH_INPUT)
    parser.add_argument("--workbench-path", type=Path)
    parser.add_argument(
        "--original-error-marker", default=DEFAULT_ORIGINAL_ERROR_MARKER
    )
    parser.add_argument(
        "--no-original-error-check",
        action="store_true",
        help="Allow a retry list that is not exactly the failures containing the marker.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source completeness and retry policy without creating output.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        config = build_config(args)
        if args.check_only:
            selected_ids, retry_ids, _, retry_summary = preflight(config)
            print(
                f"READY: method={config.method} instances={len(selected_ids)} "
                f"retry_instances={len(retry_ids)} retry_success={retry_summary.success_count} "
                f"retry_failure={retry_summary.failure_count}"
            )
            return 0
        output_root = merge_results(config)
        print(f"Merged results created at {output_root}")
        return 0
    except MergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
