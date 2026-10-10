#!/usr/bin/env python3
"""Automated chain runner for consecutive overnight runs (run1 -> run2 -> run3).

Profiles:
  baseline-qwen9b   (default) baseline agent, results in dataset/agent_results/baseline
  dalic-vsm-qwen9b  DaLIC agent on the VSM workbench, results in dataset/agent_results/dalic_vsm
  dalic-bm25-qwen4b DaLIC agent on the BM25 workbench, results in dataset/agent_results/dalic_bm25
  dalic-bm25-gpt    DaLIC agent on the BM25 workbench, results in dataset/agent_results/dalic_bm25

Options:
  --profile NAME    default: baseline-qwen9b
  --fresh           delete existing result directories of the profile's runs once at
                    start-up, then run them from scratch. Restart without --fresh to
                    resume an interrupted chain.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "dataset/trace_completed_instance_validated.toml"
PYTHON = ROOT / "python_venv/dalic-nexp/bin/python"
MINI = "scripts/run_mini_instance.py"
POLL_SECONDS = 15


@dataclass(frozen=True)
class Profile:
    results: Path
    log_name: str
    runner: str
    prompt: str
    env_file: str
    run_models: tuple[str, ...]
    expected_api_model: str
    stale_runs: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    extra_inputs: tuple[str, ...] = ()


PROFILES: dict[str, Profile] = {
    "baseline-qwen9b": Profile(
        results=ROOT / "dataset/agent_results/baseline",
        log_name="overnight-chain-qwen9b",
        runner="scripts/run_baseline_agent.py",
        prompt="scripts/prompts/baseline.yaml",
        env_file="docker/mini-swe-agent-baseline.env",
        expected_api_model="openai/Qwen3.5-9B",
        run_models=(
            "Qwen35-9B-runpod-run1",
            "Qwen35-9B-runpod-run2",
            "Qwen35-9B-runpod-run3",
        ),
    ),
    "dalic-vsm-qwen9b": Profile(
        results=ROOT / "dataset/agent_results/dalic_vsm",
        log_name="overnight-chain-dalic-vsm-qwen9b",
        runner="scripts/run_dalic_agent.py",
        prompt="scripts/prompts/dalic_v1-qwen-gpt.yaml",
        env_file="docker/mini-swe-agent-qwen35-9B.env",
        expected_api_model="openai/Qwen3.5-9B",
        run_models=(
            "Qwen35-9B-runpod-run1",
            "Qwen35-9B-runpod-run2",
            "Qwen35-9B-runpod-run3",
        ),
        stale_runs=(
            "Qwen35-9B-runpod-run1-low",
            "Qwen35-9B-runpod-run3-low",
        ),
        extra_args=(
            "--workbench_path",
            "dataset/agent_workbench/dalic_vsm",
            "--env_file",
            "docker/mini-swe-agent-qwen35-9B.env",
            "--config_yaml",
            "dalic_v1-qwen-gpt.yaml",
        ),
        extra_inputs=("scripts/prompts/task.txt",),
    ),
    "dalic-bm25-qwen4b": Profile(
        results=ROOT / "dataset/agent_results/dalic_bm25",
        log_name="overnight-chain-dalic-bm25-qwen4b",
        runner="scripts/run_dalic_agent.py",
        prompt="scripts/prompts/dalic_v1_bm25-qwen-gpt.yaml",
        env_file="docker/mini-swe-agent-qwen35-4B.env",
        expected_api_model="openai/Qwen3.5-4B",
        run_models=(
            "qwen35-4b-runpod-run1",
            "qwen35-4b-runpod-run2",
            "qwen35-4b-runpod-run3",
        ),
        extra_args=(
            "--workbench_path",
            "dataset/agent_workbench/dalic_bm25",
            "--results_base",
            "dataset/agent_results/dalic_bm25",
            "--env_file",
            "docker/mini-swe-agent-qwen35-4B.env",
            "--config_yaml",
            "dalic_v1_bm25-qwen-gpt.yaml",
        ),
        extra_inputs=("scripts/prompts/task.txt",),
    ),
    "dalic-bm25-gpt": Profile(
        results=ROOT / "dataset/agent_results/dalic_bm25",
        log_name="overnight-chain-dalic-bm25-gpt",
        runner="scripts/run_dalic_agent.py",
        prompt="scripts/prompts/dalic_v1_bm25-qwen-gpt.yaml",
        env_file="docker/mini-swe-agent-gpt.env",
        expected_api_model="openai/gpt-5.4-2026-03-05",
        run_models=(
            "gpt54-0305-openai-run1",
            "gpt54-0305-openai-run2",
            "gpt54-0305-openai-run3",
        ),
        extra_args=(
            "--workbench_path",
            "dataset/agent_workbench/dalic_bm25",
            "--results_base",
            "dataset/agent_results/dalic_bm25",
            "--env_file",
            "docker/mini-swe-agent-gpt.env",
            "--config_yaml",
            "dalic_v1_bm25-qwen-gpt.yaml",
        ),
        extra_inputs=("scripts/prompts/task.txt",),
    ),
}

P: Profile
LOG: Path
LOCK: Path


def report(message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    P.results.mkdir(parents=True, exist_ok=True)
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
            if any(arg.endswith(P.runner) for arg in args) and "--model_name" in args:
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
    names = (P.runner, MINI, P.prompt, P.env_file, *P.extra_inputs)
    return tuple(ROOT / name for name in names) + (COHORT,)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_completed(model_name: str, selected_count: int) -> bool:
    target_dir = P.results / model_name
    log_path = target_dir / "baseline.log"
    if not log_path.is_file():
        log_path = target_dir / "dalic.log"
    summary_path = target_dir / "summary.txt"
    if not log_path.is_file() or not summary_path.is_file():
        return False
    log_lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not log_lines or not any("Run finished. success_count=" in line for line in log_lines[-5:]):
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
        P.runner,
        "--model_name",
        model_name,
        "--selected_instances",
        str(COHORT),
        *P.extra_args,
        "--resume",
    ]
    report(f"[{model_name}] starting runner: {' '.join(command)}")
    console_log = P.results / f"{model_name}.console.log"
    with console_log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    report(f"[{model_name}] runner exited with code {result.returncode}; console log: {console_log}")
    return result.returncode


def remove_previous_results() -> None:
    for model_name in (*P.run_models, *P.stale_runs):
        if find_running_pid(model_name) is not None:
            raise RuntimeError(f"[{model_name}] a runner is active; refusing to delete its results")
        for path in (P.results / model_name, P.results / f"{model_name}.console.log"):
            if path.is_dir():
                shutil.rmtree(path)
                report(f"[{model_name}] --fresh: removed {path}")
            elif path.is_file():
                path.unlink()
                report(f"[{model_name}] --fresh: removed {path}")


def run_pipeline(fresh: bool) -> int:
    if not PYTHON.is_file():
        raise RuntimeError(f"Python environment missing: {PYTHON}")
    selected = tomllib.loads(COHORT.read_text(encoding="utf-8"))["instance_ids"]
    selected_count = len(selected)

    actual_model = api_model(ROOT / P.env_file)
    if actual_model != P.expected_api_model:
        raise RuntimeError(f"API model in {P.env_file} is '{actual_model}', expected '{P.expected_api_model}'")

    if fresh:
        remove_previous_results()

    snapshots = {path: digest(path) for path in input_paths()}
    report(f"armed: consecutive runs {P.run_models} on {selected_count} validated instances")

    for model_name in P.run_models:
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

    report(f"all runs of profile completed successfully: {P.run_models}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="baseline-qwen9b")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    global P, LOG, LOCK
    args = parse_args()
    P = PROFILES[args.profile]
    LOG = P.results / f"{P.log_name}.log"
    LOCK = P.results / f"{P.log_name}.lock"
    P.results.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another overnight chain runner is already running") from None
        return run_pipeline(args.fresh)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        if "P" in globals():
            report(f"failed: {exc}")
        else:
            print(f"failed: {exc}", file=sys.stderr)
        sys.exit(1)
