#!/usr/bin/env python3
"""Unified multi-run chain runner / watcher for dalic_vsm runs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "dataset/agent_results/dalic_vsm"
PYTHON = ROOT / "python_venv/dalic-nexp/bin/python"
POLL_SECONDS = 15

RUNNER = "scripts/run_dalic_agent.py"
MINI = "scripts/run_mini_instance.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-run chain watcher for dalic_vsm experiments.")
    parser.add_argument("--model_prefix", type=str, required=True, help="Prefix for model runs (e.g. Qwen35-4B)")
    parser.add_argument("--env_file", type=Path, required=True, help="Path to docker .env file")
    parser.add_argument("--config_yaml", type=str, default="dalic_v1.yaml", help="Prompt yaml configuration")
    parser.add_argument("--expected_api_model", type=str, default=None, help="Expected MSWEA_MODEL_NAME in env file")
    parser.add_argument("--runs", type=int, default=3, help="Number of chained runs (default: 3)")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=ROOT / "dataset/trace_completed_instance_validated.toml",
        help="Path to validated instances TOML",
    )
    return parser.parse_args()


def report(log_file: Path, message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def proc_start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    fields = stat.rsplit(") ", 1)[1].split()
    if fields[0] == "Z":
        return None
    return int(fields[19])


def cmdline(pid: int) -> list[str]:
    try:
        return [part.decode() for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
    except OSError:
        return []


def find_running_pid(model_name: str) -> int | None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            pid = int(entry.name)
            args = cmdline(pid)
            if RUNNER in args and "--model_name" in args:
                idx = args.index("--model_name")
                if idx + 1 < len(args) and args[idx + 1] == model_name:
                    return pid
        except (OSError, ValueError):
            continue
    return None


def api_model(env_path: Path) -> str | None:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MSWEA_MODEL_NAME="):
            return line.partition("=")[2].strip()
    return None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_completed(log_file: Path, model_name: str, selected_count: int) -> bool:
    target_dir = RESULTS / model_name
    log_path = target_dir / "dalic.log"
    summary_path = target_dir / "summary.txt"
    if not log_path.is_file() or not summary_path.is_file():
        return False
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    if not log_lines or "Run finished. success_count=" not in log_lines[-1]:
        return False
    summary_lines = summary_path.read_text(encoding="utf-8").splitlines()
    counts = dict(line.split(": ", 1) for line in summary_lines[:2] if ": " in line)
    success = int(counts.get("success_count", -1))
    failure = int(counts.get("failure_count", -1))
    report(log_file, f"[{model_name}] status check: {success} successful, {failure} failed, {selected_count} expected")
    return success + failure == selected_count


def wait_for_pid(log_file: Path, pid: int, model_name: str) -> None:
    initial_ticks = proc_start_ticks(pid)
    report(log_file, f"[{model_name}] watching active PID {pid}")
    while proc_start_ticks(pid) == initial_ticks:
        time.sleep(POLL_SECONDS)
    report(log_file, f"[{model_name}] PID {pid} finished")


def execute_run(log_file: Path, model_name: str, cohort: Path, env_file: Path, config_yaml: str) -> int:
    command = [
        str(PYTHON),
        RUNNER,
        "--model_name",
        model_name,
        "--selected_instances",
        str(cohort),
        "--env_file",
        str(env_file),
        "--config_yaml",
        config_yaml,
        "--resume",
    ]
    report(log_file, f"[{model_name}] starting runner: {' '.join(command)}")
    console_log = RESULTS / f"{model_name}.console.log"
    with console_log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    report(log_file, f"[{model_name}] runner exited with code {result.returncode}; console log: {console_log}")
    return result.returncode


def run_pipeline(args: argparse.Namespace, log_file: Path) -> int:
    if not PYTHON.is_file():
        raise RuntimeError(f"Python environment missing: {PYTHON}")

    cohort_path = args.cohort.resolve()
    selected = tomllib.loads(cohort_path.read_text(encoding="utf-8"))["instance_ids"]
    selected_count = len(selected)

    env_path = (ROOT / args.env_file).resolve() if not args.env_file.is_absolute() else args.env_file
    if not env_path.is_file():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    actual_model = api_model(env_path)
    if args.expected_api_model and actual_model != args.expected_api_model:
        raise RuntimeError(
            f"API model in {env_path} is '{actual_model}', expected '{args.expected_api_model}'"
        )

    prompt_path = ROOT / "scripts/prompts" / Path(args.config_yaml).name
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt configuration not found: {prompt_path}")

    input_paths = (ROOT / RUNNER, ROOT / MINI, prompt_path, env_path, cohort_path)
    snapshots = {path: digest(path) for path in input_paths}

    if args.runs > 1:
        run_models = [f"{args.model_prefix}-run{i}" for i in range(1, args.runs + 1)]
    else:
        run_models = [args.model_prefix]

    report(log_file, f"armed: consecutive runs {run_models} on {selected_count} validated instances")

    for model_name in run_models:
        if any(digest(p) != snapshots[p] for p in input_paths):
            report(log_file, f"[{model_name}] aborted: files modified during execution")
            return 1

        if is_completed(log_file, model_name, selected_count):
            report(log_file, f"[{model_name}] already completed; skipping to next run")
            continue

        running_pid = find_running_pid(model_name)
        if running_pid is not None:
            wait_for_pid(log_file, running_pid, model_name)
        else:
            ret = execute_run(log_file, model_name, cohort_path, env_path, args.config_yaml)
            if ret != 0:
                report(log_file, f"[{model_name}] run exited with non-zero exit code {ret}")

        if not is_completed(log_file, model_name, selected_count):
            report(log_file, f"[{model_name}] did not complete all {selected_count} instances; stopping chain")
            return 1

    report(log_file, f"all runs for {args.model_prefix} completed successfully")
    return 0


def main() -> int:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock_file = RESULTS / f"overnight-chain-{args.model_prefix}.lock"
    log_file = RESULTS / f"overnight-chain-{args.model_prefix}.log"

    with lock_file.open("a", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another chain runner for {args.model_prefix} is already running") from None
        return run_pipeline(args, log_file)


if __name__ == "__main__":
    args = parse_args()
    log_file = RESULTS / f"overnight-chain-{args.model_prefix}.log"
    try:
        sys.exit(main())
    except Exception as exc:
        report(log_file, f"failed: {exc}")
        sys.exit(1)
