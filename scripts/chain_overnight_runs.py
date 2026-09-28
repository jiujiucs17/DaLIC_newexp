#!/usr/bin/env python3
"""Automated chain runner for 27B no-runtime overnight runs (run1 -> run2 -> run3)."""

from __future__ import annotations

import fcntl
import hashlib
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "dataset/agent_results/dalic_no_runtime"
COHORT = ROOT / "dataset/trace_completed_instance_validated.toml"
PYTHON = ROOT / "python_venv/dalic-nexp/bin/python"
LOG = RESULTS / "overnight-chain-27b.log"
LOCK = RESULTS / "overnight-chain-27b.lock"
POLL_SECONDS = 15

RUNNER = "scripts/run_dalic_agent_no_runtime.py"
MINI = "scripts/run_mini_instance.py"
PROMPT = "scripts/prompts/dalic_no_runtime.yaml"
ENV_FILE = "docker/mini-swe-agent-dalic.env"
EXPECTED_API_MODEL = "openai/Qwen3.5-27B"

RUN_MODELS = (
    "Qwen35-27B-no-runtime-run1",
    "Qwen35-27B-no-runtime-run2",
    "Qwen35-27B-no-runtime-run3",
)


def report(message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
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
            return line.partition("=")[2]
    return None


def input_paths() -> tuple[Path, ...]:
    return tuple(ROOT / name for name in (RUNNER, MINI, PROMPT, ENV_FILE)) + (COHORT,)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_completed(model_name: str, selected_count: int) -> bool:
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
    report(f"[{model_name}] status check: {success} successful, {failure} failed, {selected_count} expected")
    return success + failure == selected_count


def wait_for_pid(pid: int, model_name: str) -> None:
    initial_ticks = proc_start_ticks(pid)
    report(f"[{model_name}] watching active PID {pid}")
    while proc_start_ticks(pid) == initial_ticks:
        time.sleep(POLL_SECONDS)
    report(f"[{model_name}] PID {pid} finished")


def execute_run(model_name: str) -> int:
    command = [
        str(PYTHON),
        RUNNER,
        "--model_name",
        model_name,
        "--selected_instances",
        str(COHORT),
        "--resume",
    ]
    report(f"[{model_name}] starting runner: {' '.join(command)}")
    console_log = RESULTS / f"{model_name}.console.log"
    with console_log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    report(f"[{model_name}] runner exited with code {result.returncode}; console log: {console_log}")
    return result.returncode


def run_pipeline() -> int:
    if not PYTHON.is_file():
        raise RuntimeError(f"Python environment missing: {PYTHON}")
    selected = tomllib.loads(COHORT.read_text(encoding="utf-8"))["instance_ids"]
    selected_count = len(selected)

    actual_model = api_model(ROOT / ENV_FILE)
    if actual_model != EXPECTED_API_MODEL:
        raise RuntimeError(f"API model in {ENV_FILE} is '{actual_model}', expected '{EXPECTED_API_MODEL}'")

    snapshots = {path: digest(path) for path in input_paths()}
    report(f"armed: consecutive runs {RUN_MODELS} on {selected_count} validated instances")

    for model_name in RUN_MODELS:
        # Check if code or environment changed
        if any(digest(p) != snapshots[p] for p in input_paths()):
            report(f"[{model_name}] aborted: files modified during execution")
            return 1

        if is_completed(model_name, selected_count):
            report(f"[{model_name}] already completed; skipping to next run")
            continue

        running_pid = find_running_pid(model_name)
        if running_pid is not None:
            wait_for_pid(running_pid, model_name)
        else:
            ret = execute_run(model_name)
            if ret != 0:
                report(f"[{model_name}] run exited with non-zero exit code {ret}")

        # Verify completion before proceeding to next run
        if not is_completed(model_name, selected_count):
            report(f"[{model_name}] did not complete all {selected_count} instances; stopping chain")
            return 1

    report("all 27B runs completed successfully")
    return 0


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another overnight chain runner is already running") from None
        return run_pipeline()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        report(f"failed: {exc}")
        sys.exit(1)
