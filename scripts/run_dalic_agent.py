#!/usr/bin/env python3
# Options:
#   --model_name TEXT
#       default: Qwen_Qwen3.5-9B
#   --test
#       default: False (run all selected instances; when enabled, randomly run at most 10)
#   --selected_instances PATH
#       default: None (do not filter instances; otherwise read instance_ids from TOML)
#   --workbench_path PATH
#       default: dataset/agent_workbench/dalic

from __future__ import annotations

import argparse
import json
import logging
import random
import shlex
import shutil
import subprocess
import time
import tomllib
from pathlib import Path, PurePosixPath

from tqdm import tqdm


DEFAULT_MODEL_NAME = "Qwen_Qwen3.5-9B"
DEFAULT_WORKBENCH_PATH = Path("dataset/agent_workbench/dalic")
CONTAINER_RESULTS_DIR = "/home/miniagent/.config/mini-swe-agent"
ARTIFACT_NAMES = ("results.txt", "last_mini_run.traj.json")
RESULTS_FIELDS = ("files", "classes", "functions")
TEST_INSTANCE_COUNT = 10


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--selected_instances", type=Path, default=None)
    parser.add_argument("--workbench_path", type=Path, default=DEFAULT_WORKBENCH_PATH)
    return parser.parse_args()


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def prepare_results_root(repo_root: Path, model_name: str) -> Path:
    results_root = repo_root / "dataset" / "agent_results" / "dalic" / model_name
    if results_root.exists():
        shutil.rmtree(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("run_dalic")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def resolve_cli_path(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def load_selected_instance_ids(selected_instances_path: Path) -> set[str]:
    with selected_instances_path.open("rb") as selected_instances_file:
        payload = tomllib.load(selected_instances_file)

    instance_ids = payload.get("instance_ids")
    if not isinstance(instance_ids, list) or any(
        not isinstance(instance_id, str) for instance_id in instance_ids
    ):
        raise ValueError(
            f"{selected_instances_path} must define 'instance_ids' as a list of strings"
        )
    return set(instance_ids)


def list_instance_ids(
    workbench_dir: Path,
    test_mode: bool,
    selected_instances_path: Path | None = None,
) -> tuple[list[str], int | None]:
    instance_ids = sorted(
        path.name
        for path in workbench_dir.iterdir()
        if path.is_dir() and path.name != ".repo_mirrors"
    )
    if selected_instances_path is not None:
        selected_instance_ids = load_selected_instance_ids(selected_instances_path)
        instance_ids = [
            instance_id
            for instance_id in instance_ids
            if instance_id in selected_instance_ids
        ]
    if test_mode:
        sample_size = min(TEST_INSTANCE_COUNT, len(instance_ids))
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


def validate_repo_relative_path(
    path_text: str, repository_root: Path
) -> str | None:
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


def validate_results_file(results_path: Path, repository_root: Path) -> str | None:
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "results.txt is missing"
    except json.JSONDecodeError as exc:
        return f"results.txt is not valid JSON: {exc.msg}"

    if not isinstance(payload, dict):
        return "top-level JSON value must be an object"
    if set(payload) != {"results"}:
        return "top-level JSON object must contain exactly one key: 'results'"

    results = payload["results"]
    if not isinstance(results, dict):
        return "'results' must be a JSON object"
    if set(results) != set(RESULTS_FIELDS):
        return "'results' must contain exactly these keys: files, classes, functions"

    for field_name in RESULTS_FIELDS:
        entries = results[field_name]
        if not isinstance(entries, list):
            return f"'{field_name}' must be a list"
        if any(not isinstance(entry, str) for entry in entries):
            return f"'{field_name}' must contain only strings"
        if len(entries) != len(set(entries)):
            return f"'{field_name}' contains duplicate entries"

    for file_path in results["files"]:
        path_error = validate_repo_relative_path(file_path, repository_root)
        if path_error:
            return f"invalid file entry '{file_path}': {path_error}"

    for field_name in ("classes", "functions"):
        for entry in results[field_name]:
            if entry.count("::") != 1:
                return f"invalid {field_name[:-1]} entry '{entry}': expected '<file_path>::<qualified_name>'"
            file_path, qualified_name = entry.split("::", 1)
            path_error = validate_repo_relative_path(file_path, repository_root)
            if path_error:
                return f"invalid {field_name[:-1]} entry '{entry}': {path_error}"
            if not qualified_name:
                return f"invalid {field_name[:-1]} entry '{entry}': qualified name is empty"

    return None


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
    workbench_dir: Path,
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
        str(repo_root / "docker" / "mini-swe-agent.env"),
        "-v",
        f"{workbench_dir / instance_id}:/workspace/instance:ro",
        "-v",
        f"{repo_root / 'scripts' / 'run_mini_instance.py'}:/opt/dalic/run_mini_instance.py:ro",
        "-v",
        f"{repo_root / 'scripts' / 'prompts'}:/opt/dalic/prompts:ro",
        "-w",
        "/workspace/instance",
        "mini-swe-agent:2caffc56",
        "python",
        "/opt/dalic/run_mini_instance.py",
        "--agent_version", "dalic"
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


def write_summary(
    results_root: Path,
    success_count: int,
    failed_instances: list[str],
    failure_reasons: dict[str, str],
) -> None:
    summary_lines = [
        f"success_count: {success_count}",
        f"failure_count: {len(failed_instances)}",
        "failed_instances:",
    ]
    if failed_instances:
        summary_lines.extend(failed_instances)
    else:
        summary_lines.append("None")
    summary_lines.append("failure_reasons:")
    if failed_instances:
        for instance_id in failed_instances:
            summary_lines.append(f"{instance_id}: {failure_reasons.get(instance_id, 'unknown failure')}")
    else:
        summary_lines.append("None")
    (results_root / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root().resolve()
    workbench_dir = resolve_cli_path(repo_root, args.workbench_path)
    selected_instances_path = (
        resolve_cli_path(repo_root, args.selected_instances)
        if args.selected_instances is not None
        else None
    )
    model_name = sanitize_model_name(args.model_name)
    results_root = prepare_results_root(repo_root, model_name)
    logger = configure_logger(results_root / "dalic.log")

    instance_ids, test_seed = list_instance_ids(
        workbench_dir,
        args.test,
        selected_instances_path,
    )
    summary_path = results_root / "summary.txt"
    logger.info("Results root: %s", results_root)
    logger.info("Summary path: %s", summary_path)
    logger.info("Resolved repo root: %s", repo_root)
    logger.info("Workbench path: %s", workbench_dir)
    logger.info("Selected instances path: %s", selected_instances_path)
    logger.info("Model name: %s", model_name)
    logger.info("Test mode: %s", args.test)
    if test_seed is not None:
        logger.info("Test mode random seed: %s", test_seed)
        logger.info("Test mode selected instances: %s", instance_ids)
    logger.info("Selected %s instances from %s", len(instance_ids), workbench_dir)

    failed_instances: list[str] = []
    failure_reasons: dict[str, str] = {}
    success_count = 0
    write_summary(results_root, success_count, failed_instances, failure_reasons)

    try:
        progress = tqdm(instance_ids, desc="dalic", unit="instance")
        for instance_id in progress:
            container_name = f"mini-fl-{instance_id}_dalic"
            instance_result_dir = results_root / instance_id
            instance_result_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[%s] Starting dalic flow.", instance_id)

            instance_success = True
            instance_fail_reasons: list[str] = []
            try:
                progress.set_postfix_str(f"{instance_id}: remove old container", refresh=True)
                remove_existing_container(container_name, logger, instance_id)

                progress.set_postfix_str(f"{instance_id}: docker run", refresh=True)
                if not run_docker_instance(
                    repo_root,
                    workbench_dir,
                    instance_id,
                    container_name,
                    instance_result_dir,
                    logger,
                ):
                    instance_success = False
                    instance_fail_reasons.append("docker run failed")

                progress.set_postfix_str(f"{instance_id}: docker cp", refresh=True)
                artifacts_copied = copy_container_artifacts(
                    container_name,
                    instance_result_dir,
                    logger,
                    instance_id,
                )
                if not artifacts_copied:
                    instance_success = False
                    instance_fail_reasons.append("artifact copy failed")

                results_path = instance_result_dir / "results.txt"
                if results_path.exists():
                    progress.set_postfix_str(f"{instance_id}: validate results", refresh=True)
                    repository_root = find_instance_repository_root(
                        workbench_dir, instance_id
                    )
                    validation_error = validate_results_file(
                        results_path, repository_root
                    )
                    if validation_error is None:
                        logger.info("[%s] results.txt validation passed.", instance_id)
                    else:
                        logger.error("[%s] results.txt validation failed: %s", instance_id, validation_error)
                        instance_success = False
                        instance_fail_reasons.append(f"invalid results.txt: {validation_error}")
                elif not artifacts_copied:
                    logger.info("[%s] Skipping results.txt validation because results.txt was not copied.", instance_id)
            except Exception as exc:
                logger.exception("[%s] Unexpected top-level failure: %s", instance_id, exc)
                instance_success = False
                instance_fail_reasons.append(f"unexpected top-level failure: {exc}")
            finally:
                progress.set_postfix_str(f"{instance_id}: docker rm", refresh=True)
                if not remove_container(container_name, logger, instance_id):
                    instance_success = False
                    instance_fail_reasons.append("docker rm failed")

            if instance_success:
                success_count += 1
                logger.info("[%s] Finished successfully.", instance_id)
            else:
                failed_instances.append(instance_id)
                failure_reason = "; ".join(dict.fromkeys(instance_fail_reasons)) or "unknown failure"
                failure_reasons[instance_id] = failure_reason
                logger.error("[%s] Finished with failures. reasons=%s", instance_id, failure_reason)

            write_summary(results_root, success_count, failed_instances, failure_reasons)
    finally:
        write_summary(results_root, success_count, failed_instances, failure_reasons)

    logger.info(
        "Run finished. success_count=%s failure_count=%s failed_instances=%s",
        success_count,
        len(failed_instances),
        failed_instances or ["None"],
    )


if __name__ == "__main__":
    main()
