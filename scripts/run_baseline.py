#!/usr/bin/env python3
# Usage:
#   python3 scripts/run_baseline.py
#   python3 scripts/run_baseline.py --model_name Qwen_Qwen3.5-9B
#   python3 scripts/run_baseline.py --model_name some/org-model --test

from __future__ import annotations

import argparse
import logging
import random
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from tqdm import tqdm


DEFAULT_MODEL_NAME = "Qwen_Qwen3.5-9B"
CONTAINER_RESULTS_DIR = "/home/miniagent/.config/mini-swe-agent"
ARTIFACT_NAMES = ("results.txt", "last_mini_run.traj.json")


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def prepare_results_root(repo_root: Path, model_name: str) -> Path:
    results_root = repo_root / "dataset" / "agent_results" / "baseline" / model_name
    if results_root.exists():
        shutil.rmtree(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("run_baseline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def list_instance_ids(workbench_dir: Path, test_mode: bool) -> tuple[list[str], int | None]:
    instance_ids = sorted(
        path.name
        for path in workbench_dir.iterdir()
        if path.is_dir() and path.name != ".repo_mirrors"
    )
    if test_mode:
        sample_size = min(3, len(instance_ids))
        seed = time.time_ns()
        rng = random.Random(seed)
        return rng.sample(instance_ids, sample_size), seed
    return instance_ids, None


def write_command_log(
    log_path: Path,
    command: list[str],
    returncode: int | None,
    stdout: str | None,
    stderr: str | None,
    exception_message: str | None = None,
) -> None:
    lines = [
        f"command: {shlex.join(command)}",
        f"returncode: {returncode if returncode is not None else 'N/A'}",
        "",
        "stdout:",
        stdout or "",
        "",
        "stderr:",
        stderr or "",
    ]
    if exception_message:
        lines.extend(["", "exception:", exception_message])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def remove_existing_container(
    container_name: str,
    logger: logging.Logger,
    instance_id: str,
) -> None:
    command = ["docker", "rm", "-f", container_name]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        logger.info("[%s] Removed existing container %s before run.", instance_id, container_name)
        return

    message = " ".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
    if "No such container" in message:
        logger.info("[%s] No existing container to remove before run.", instance_id)
        return

    logger.warning(
        "[%s] Failed to remove existing container %s before run. returncode=%s",
        instance_id,
        container_name,
        completed.returncode,
    )


def run_docker_instance(
    repo_root: Path,
    instance_id: str,
    container_name: str,
    instance_result_dir: Path,
    logger: logging.Logger,
) -> bool:
    command = [
        "docker",
        "run",
        "--name",
        container_name,
        "--env-file",
        str(repo_root / "docker" / "baseline.env"),
        "-v",
        f"{repo_root / 'dataset' / 'agent_workbench' / instance_id}:/workspace/instance:ro",
        "-v",
        f"{repo_root / 'scripts' / 'run_mini_instance.py'}:/opt/dalic/run_mini_instance.py:ro",
        "-v",
        f"{repo_root / 'scripts' / 'prompts'}:/opt/dalic/prompts:ro",
        "-w",
        "/workspace/instance",
        "dalic-mini-swe:2caffc56",
        "python",
        "/opt/dalic/run_mini_instance.py",
    ]
    log_path = instance_result_dir / "docker_run.log"

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        write_command_log(
            log_path,
            command,
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )
        logger.info("[%s] docker run completed successfully.", instance_id)
        return True
    except subprocess.CalledProcessError as exc:
        write_command_log(
            log_path,
            command,
            exc.returncode,
            exc.stdout,
            exc.stderr,
            exception_message=str(exc),
        )
        logger.error(
            "[%s] docker run failed at subprocess 'docker run'. returncode=%s. See %s",
            instance_id,
            exc.returncode,
            log_path,
        )
        return False
    except Exception as exc:
        write_command_log(
            log_path,
            command,
            None,
            None,
            None,
            exception_message=str(exc),
        )
        logger.exception("[%s] Unexpected error during docker run: %s", instance_id, exc)
        return False


def copy_container_artifacts(
    container_name: str,
    instance_result_dir: Path,
    logger: logging.Logger,
    instance_id: str,
) -> bool:
    step_success = True
    for artifact_name in ARTIFACT_NAMES:
        command = [
            "docker",
            "cp",
            f"{container_name}:{CONTAINER_RESULTS_DIR}/{artifact_name}",
            str(instance_result_dir / artifact_name),
        ]
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info("[%s] Copied %s from container.", instance_id, artifact_name)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "[%s] Failed at subprocess 'docker cp' for %s. returncode=%s",
                instance_id,
                artifact_name,
                exc.returncode,
            )
            step_success = False
        except Exception as exc:
            logger.exception("[%s] Unexpected error while copying %s: %s", instance_id, artifact_name, exc)
            step_success = False
    return step_success


def remove_container(
    container_name: str,
    logger: logging.Logger,
    instance_id: str,
) -> bool:
    command = ["docker", "rm", container_name]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info("[%s] Removed container %s after run.", instance_id, container_name)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(
            "[%s] Failed at subprocess 'docker rm' for %s. returncode=%s",
            instance_id,
            container_name,
            exc.returncode,
        )
        return False
    except Exception as exc:
        logger.exception("[%s] Unexpected error while removing container %s: %s", instance_id, container_name, exc)
        return False


def write_summary(results_root: Path, success_count: int, failed_instances: list[str]) -> None:
    summary_lines = [
        f"success_count: {success_count}",
        f"failure_count: {len(failed_instances)}",
        "failed_instances:",
    ]
    if failed_instances:
        summary_lines.extend(failed_instances)
    else:
        summary_lines.append("None")
    (results_root / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root().resolve()
    workbench_dir = repo_root / "dataset" / "agent_workbench"
    model_name = sanitize_model_name(args.model_name)
    results_root = prepare_results_root(repo_root, model_name)
    logger = configure_logger(results_root / "baseline.log")

    instance_ids, test_seed = list_instance_ids(workbench_dir, args.test)
    summary_path = results_root / "summary.txt"
    logger.info("Results root: %s", results_root)
    logger.info("Summary path: %s", summary_path)
    logger.info("Resolved repo root: %s", repo_root)
    logger.info("Model name: %s", model_name)
    logger.info("Test mode: %s", args.test)
    if test_seed is not None:
        logger.info("Test mode random seed: %s", test_seed)
        logger.info("Test mode selected instances: %s", instance_ids)
    logger.info("Selected %s instances from %s", len(instance_ids), workbench_dir)

    failed_instances: list[str] = []
    success_count = 0
    write_summary(results_root, success_count, failed_instances)

    try:
        progress = tqdm(instance_ids, desc="baseline", unit="instance")
        for instance_id in progress:
            container_name = f"mini-fl-{instance_id}"
            instance_result_dir = results_root / instance_id
            instance_result_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[%s] Starting baseline flow.", instance_id)

            instance_success = True
            try:
                progress.set_postfix_str(f"{instance_id}: remove old container", refresh=True)
                remove_existing_container(container_name, logger, instance_id)

                progress.set_postfix_str(f"{instance_id}: docker run", refresh=True)
                if not run_docker_instance(repo_root, instance_id, container_name, instance_result_dir, logger):
                    instance_success = False

                progress.set_postfix_str(f"{instance_id}: docker cp", refresh=True)
                if not copy_container_artifacts(container_name, instance_result_dir, logger, instance_id):
                    instance_success = False
            except Exception as exc:
                logger.exception("[%s] Unexpected top-level failure: %s", instance_id, exc)
                instance_success = False
            finally:
                progress.set_postfix_str(f"{instance_id}: docker rm", refresh=True)
                if not remove_container(container_name, logger, instance_id):
                    instance_success = False

            if instance_success:
                success_count += 1
                logger.info("[%s] Finished successfully.", instance_id)
            else:
                failed_instances.append(instance_id)
                logger.error("[%s] Finished with failures.", instance_id)

            write_summary(results_root, success_count, failed_instances)
    finally:
        write_summary(results_root, success_count, failed_instances)

    logger.info(
        "Run finished. success_count=%s failure_count=%s failed_instances=%s",
        success_count,
        len(failed_instances),
        failed_instances or ["None"],
    )


if __name__ == "__main__":
    main()
