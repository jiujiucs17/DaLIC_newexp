#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import docker
from tqdm import tqdm

from swebench.harness.docker_build import build_container
from swebench.harness.docker_utils import remove_image
from swebench.harness.test_spec.test_spec import make_test_spec


UTF8 = "utf-8"
DOCKER_USER = "root"
DOCKER_WORKDIR = "/testbed"
OFFICIAL_NAMESPACE = "swebench"
DEFAULT_JSONL = "dataset/swebench_feature_with_truth_sr_scripts.jsonl"
DEFAULT_OUTPUT_DIR = "ab_runs"
IMAGE_LIMIT_BYTES = 500 * 1000**3


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else repo_root / path).resolve()


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generated A/B scripts serially in official SWE-bench historical containers."
    )
    parser.add_argument("--jsonl", type=Path, default=repo_root / DEFAULT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_instances", type=int)
    parser.add_argument("--run-id", default="ab_batch")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-code-patch", action="store_true")
    parser.add_argument("--skip-test-patch", action="store_true")
    args = parser.parse_args()

    args.jsonl = resolve_path(args.jsonl, repo_root)
    args.output_dir = resolve_path(args.output_dir, repo_root)
    if not args.jsonl.is_file():
        parser.error(f"JSONL file does not exist: {args.jsonl}")
    if args.num_instances is not None and args.num_instances <= 0:
        parser.error("--num_instances must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def load_selected_rows(jsonl_path: Path) -> tuple[int, list[dict[str, Any]]]:
    rows = [json.loads(line) for line in jsonl_path.open(encoding=UTF8)]
    selected = [
        row
        for row in rows
        if str(row.get("A_B_Reliability", "")).startswith(("3.", "4."))
        and row.get("ground_truth_certainty") == "sure"
    ]
    return len(rows), selected


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding=UTF8)


def write_instance_files(row: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "row.json": output_dir / "row.json",
        "A.py": output_dir / "A.py",
        "B.py": output_dir / "B.py",
        "patch.diff": output_dir / "patch.diff",
        "test_patch.diff": output_dir / "test_patch.diff",
    }
    write_json(files["row.json"], row)
    files["A.py"].write_text(str(row.get("script_A") or ""), encoding=UTF8)
    files["B.py"].write_text(str(row.get("script_B") or ""), encoding=UTF8)
    files["patch.diff"].write_text(str(row.get("patch") or ""), encoding=UTF8)
    files["test_patch.diff"].write_text(str(row.get("test_patch") or ""), encoding=UTF8)
    return files


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_ab")
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


def instance_logger(logger: logging.Logger, instance_id: str, log_file: Path) -> logging.LoggerAdapter:
    adapter = logging.LoggerAdapter(logger, {"instance_id": instance_id})
    setattr(adapter, "log_file", log_file)
    return adapter


def run_command(
    command: list[str],
    *,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
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


def apply_patch(container_name: str, patch_file: Path) -> None:
    if not patch_file.read_text(encoding=UTF8).strip():
        return
    container_path = f"/tmp/{patch_file.name}"
    run_command(["docker", "cp", str(patch_file), f"{container_name}:{container_path}"])
    docker_exec(container_name, ["git", "apply", "-v", container_path], workdir=DOCKER_WORKDIR)


def docker_image_size_bytes() -> int:
    output = run_command(["docker", "system", "df", "--format", "{{json .}}"]).stdout
    for line in output.splitlines():
        row = json.loads(line)
        if row.get("Type") == "Images":
            match = re.fullmatch(r"\s*([\d.]+)\s*([kmgtp]?i?b)\s*", row["Size"], re.IGNORECASE)
            if match is None:
                raise ValueError(f"Unsupported Docker image size: {row['Size']}")
            unit = match.group(2).lower().replace("i", "")
            factors = {"b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4, "pb": 1000**5}
            return int(float(match.group(1)) * factors[unit])
    raise RuntimeError("docker system df did not report image usage")


def run_script(
    *,
    kind: str,
    args: argparse.Namespace,
    client: docker.DockerClient,
    test_spec: Any,
    files: dict[str, Path],
    logger: logging.LoggerAdapter,
) -> dict[str, Any]:
    group = {"A": 2, "B": 3}[kind]
    instance_id = test_spec.instance_id
    script_file = files[f"{kind}.py"]
    container_script = f"/tmp/{kind}.py"
    output_dir = args.output_dir / instance_id / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "instance_id": instance_id,
        "script_kind": kind,
        "script_path": str(script_file),
        "stdout_path": str(output_dir / "stdout.txt"),
        "container_id": None,
        "runtime_seconds": None,
        "exit_code": None,
        "timed_out": False,
    }
    container = None
    step = f"{group}.1"

    try:
        script_text = script_file.read_text(encoding=UTF8)
        if not script_text.strip():
            raise ValueError(f"Empty script: {script_file}")
        module = ast.parse(script_text, filename=str(script_file))
        if not any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in module.body):
            raise ValueError(f"Generated script does not define main(): {script_file}")
        logger.info("[Step %s] kind=%s using host script; path=%s", step, kind, script_file)

        step = f"{group}.2"
        logger.info("[Step %s] kind=%s creating official historical container; image=%s", step, kind, test_spec.instance_image_key)
        container = build_container(
            test_spec=test_spec,
            client=client,
            run_id=f"{args.run_id}-{kind.lower()}",
            logger=logger,
            nocache=False,
            force_rebuild=False,
        )
        metadata["container_id"] = container.id
        logger.info("[Step %s] kind=%s created container; name=%s id=%s", step, kind, container.name, container.id)

        step = f"{group}.3"
        logger.info("[Step %s] kind=%s starting container and copying script", step, kind)
        run_command(["docker", "start", container.name])
        run_command(["docker", "cp", str(script_file), f"{container.name}:{container_script}"])

        step = f"{group}.4"
        logger.info("[Step %s] kind=%s copying and applying patches", step, kind)
        if not args.skip_code_patch:
            apply_patch(container.name, files["patch.diff"])
        if not args.skip_test_patch:
            apply_patch(container.name, files["test_patch.diff"])
        diff_stat = docker_exec(
            container.name,
            ["git", "-c", "core.fileMode=false", "diff", "--stat"],
            workdir=DOCKER_WORKDIR,
        ).stdout.strip()
        logger.info("[Step %s] kind=%s patches applied; diff_stat=%r", step, kind, diff_stat)

        step = f"{group}.5"
        shell_command = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            f"cd {DOCKER_WORKDIR} && "
            f"python {container_script}"
        )
        logger.info("[Step %s] kind=%s executing script; timeout_seconds=%d", step, kind, args.timeout)
        start_time = time.monotonic()
        try:
            result = docker_exec(container.name, ["bash", "-lc", shell_command], check=False, timeout=args.timeout)
            output_text = result.stdout
            metadata["exit_code"] = result.returncode
        except subprocess.TimeoutExpired as exc:
            output_text = exc.stdout or ""
            if isinstance(output_text, bytes):
                output_text = output_text.decode(UTF8, errors="replace")
            metadata["timed_out"] = True
        metadata["runtime_seconds"] = time.monotonic() - start_time
        logger.info(
            "[Step %s] kind=%s execution finished; timed_out=%s runtime_seconds=%.3f exit_code=%s",
            step,
            kind,
            metadata["timed_out"],
            metadata["runtime_seconds"],
            metadata["exit_code"],
        )

        step = f"{group}.6"
        (output_dir / "stdout.txt").write_text(output_text, encoding=UTF8)
        if metadata["timed_out"]:
            raise TimeoutError(f"{instance_id} {kind} timed out after {args.timeout} seconds")
        return metadata
    except Exception as exc:
        metadata["failed_step"] = step
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("[Step %s] kind=%s failed", step, kind)
        raise
    finally:
        write_json(output_dir / "metadata.json", metadata)
        logger.info("[Step %s] kind=%s wrote metadata", f"{group}.6", kind)
        if container is not None:
            logger.info("[Step %s] kind=%s removing container; name=%s", f"{group}.6", kind, container.name)
            run_command(["docker", "rm", "-f", container.name], check=False)


def run_instance(
    *,
    index: int,
    total: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    client: docker.DockerClient,
    logger: logging.LoggerAdapter,
) -> dict[str, Any]:
    instance_id = row["instance_id"]
    instance_dir = args.output_dir / instance_id
    summary: dict[str, Any] = {"instance_id": instance_id, "status": "completed", "results": []}
    test_spec = None

    logger.info("[Instance %d/%d Step 1.1] preparing instance", index, total)
    try:
        files = write_instance_files(row, instance_dir)
        logger.info("[Instance %d/%d Step 1.2] wrote host files; dir=%s files=%s", index, total, instance_dir, ",".join(sorted(files)))
        test_spec = make_test_spec(row, namespace=OFFICIAL_NAMESPACE)
        for kind in ("A", "B"):
            summary["results"].append(
                run_script(kind=kind, args=args, client=client, test_spec=test_spec, files=files, logger=logger)
            )
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("[Instance %d/%d Step 4.1] instance failed", index, total)

    try:
        image_size = docker_image_size_bytes()
        summary["docker_image_size_bytes"] = image_size
        logger.info("[Instance %d/%d Step 4.2] Docker image usage; bytes=%d gb=%.3f", index, total, image_size, image_size / 1000**3)
        if image_size > IMAGE_LIMIT_BYTES and test_spec is not None:
            logger.info("[Instance %d/%d Step 4.2] removing instance image because Docker images exceed 500GB; image=%s", index, total, test_spec.instance_image_key)
            remove_image(client, test_spec.instance_image_key, "quiet")
            summary["removed_instance_image"] = test_spec.instance_image_key
    except Exception as exc:
        summary["status"] = "failed"
        summary["docker_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("[Instance %d/%d Step 4.2] Docker image usage check failed", index, total)

    write_json(instance_dir / "summary.json", summary)
    logger.info("[Instance %d/%d Step 4.3] wrote instance summary; status=%s", index, total, summary["status"])
    return summary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args(repo_root)
    os.chdir(repo_root)

    log_file = args.output_dir / "run_ab.log"
    logger = setup_logger(log_file)
    batch_logger = instance_logger(logger, "batch", log_file)
    print(f"workflow_log = {log_file}")

    try:
        batch_logger.info("[Batch Step 1.1] loading dataset; jsonl=%s", args.jsonl)
        total_rows, selected_rows = load_selected_rows(args.jsonl)
        scheduled_rows = selected_rows[: args.num_instances]
        batch_logger.info(
            "[Batch Step 1.2] selected subset; dataset_rows=%d selected_instances=%d scheduled_instances=%d",
            total_rows,
            len(selected_rows),
            len(scheduled_rows),
        )

        client = docker.from_env()
        summaries = []
        for index, row in enumerate(tqdm(scheduled_rows, desc="run_ab", unit="instance"), start=1):
            logger_for_instance = instance_logger(logger, row["instance_id"], log_file)
            summaries.append(
                run_instance(
                    index=index,
                    total=len(scheduled_rows),
                    row=row,
                    args=args,
                    client=client,
                    logger=logger_for_instance,
                )
            )

        image_size = docker_image_size_bytes()
        batch_summary = {
            "dataset_rows": total_rows,
            "selected_instances": len(selected_rows),
            "scheduled_instances": len(scheduled_rows),
            "completed_instances": sum(summary["status"] == "completed" for summary in summaries),
            "failed_instances": sum(summary["status"] == "failed" for summary in summaries),
            "docker_image_total_size_bytes": image_size,
            "docker_image_total_size_gb": image_size / 1000**3,
            "instances": summaries,
        }
        write_json(args.output_dir / "summary.json", batch_summary)
        batch_logger.info(
            "[Batch Step 2.1] workflow finished; completed_instances=%d failed_instances=%d docker_image_total_size_gb=%.3f",
            batch_summary["completed_instances"],
            batch_summary["failed_instances"],
            batch_summary["docker_image_total_size_gb"],
        )
        print(json.dumps({key: value for key, value in batch_summary.items() if key != "instances"}, ensure_ascii=False, indent=2))
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
