#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from string import Template
from tempfile import TemporaryDirectory
from typing import Any

import docker
from docker.errors import APIError, NotFound

from swebench.harness.constants import DOCKER_USER, DOCKER_WORKDIR, UTF8
from swebench.harness.docker_build import build_container, close_logger, setup_logger
from swebench.harness.docker_utils import cleanup_container, exec_run_with_timeout, write_to_container
from swebench.harness.test_spec.test_spec import make_test_spec


DEFAULT_INPUT_FILE = "swebench_feature_with_truth_sr_scripts.jsonl"
DEFAULT_OUTPUT_DIR = "ab_runs"
DEFAULT_RUN_ID = "ab_trace"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_NAMESPACE = "swebench"
DEFAULT_CONTAINER_SCRIPT_DIR = PurePosixPath("/tmp/ab_scripts")
DEFAULT_CONTAINER_SUPPORT_DIR = PurePosixPath("/tmp/ab_support")
DEFAULT_CONTAINER_TRACE_ROOT = PurePosixPath("/tmp/ab_traces")
DEFAULT_TRACE_FILENAME = "trace.json"
KEY_INSTANCE_ID = "instance_id"
EXIT_CODE_MARKER = "__AB_EXIT_CODE__="

APPLY_CMDS = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def resolve_path_from_repo_root(path: Path, repo_root: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run generated script_A/script_B inside the official SWE-bench historical "
            "Docker environment for a single instance."
        )
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=repo_root / "dataset" / DEFAULT_INPUT_FILE,
        help=f"Input JSONL with script_A/script_B. Default: dataset/{DEFAULT_INPUT_FILE}",
    )
    parser.add_argument(
        "--instance-id",
        required=True,
        help="Target SWE-bench instance_id.",
    )
    parser.add_argument(
        "--which",
        choices=("A", "B", "both"),
        default="both",
        help="Which generated script to run. Default: both",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / DEFAULT_OUTPUT_DIR,
        help=f"Directory for run logs and copied trace artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=f"Logical run identifier used in logs/container names. Default: {DEFAULT_RUN_ID}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout in seconds per A/B run. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--python-exe",
        default="python",
        help="Python executable to use inside the container. Default: python",
    )
    parser.add_argument(
        "--launch-command-template",
        default="$PYTHON_EXE $SCRIPT_PATH",
        help=(
            "Template for the final command that executes the generated A/B script inside the "
            "container. Template variables: $PYTHON_EXE, $SCRIPT_PATH, $TRACE_OUTPUT, "
            "$TRACE_OUTPUT_DIR, $INSTANCE_ID, $SCRIPT_KIND, $SUPPORT_ROOT, $WORKDIR."
        ),
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help=f"Image namespace passed to make_test_spec. Default: {DEFAULT_NAMESPACE}",
    )
    parser.add_argument(
        "--env-image-tag",
        default="latest",
        help="Environment image tag passed to make_test_spec. Default: latest",
    )
    parser.add_argument(
        "--instance-image-tag",
        default="latest",
        help="Instance image tag passed to make_test_spec. Default: latest",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild the SWE-bench instance image.",
    )
    parser.add_argument(
        "--keep-container",
        action="store_true",
        help="Keep the Docker container after execution for manual inspection.",
    )
    parser.add_argument(
        "--skip-code-patch",
        action="store_true",
        help="Do not apply row['patch'] before running the script.",
    )
    parser.add_argument(
        "--skip-test-patch",
        action="store_true",
        help="Do not apply row['test_patch'] before running the script.",
    )
    parser.add_argument(
        "--trace-preamble-file",
        type=Path,
        default=None,
        help=(
            "Optional local file whose contents are injected near the top of script_A/script_B. "
            "Template variables: $INSTANCE_ID, $SCRIPT_KIND, $TRACE_OUTPUT, "
            "$TRACE_OUTPUT_DIR, $SUPPORT_ROOT, $WORKDIR."
        ),
    )
    parser.add_argument(
        "--trace-epilogue-file",
        type=Path,
        default=None,
        help=(
            "Optional local file whose contents are appended to the end of script_A/script_B. "
            "Supports the same template variables as --trace-preamble-file."
        ),
    )
    parser.add_argument(
        "--support-path",
        action="append",
        type=Path,
        default=[],
        help=(
            "Local file or directory to copy into the container support directory and prepend "
            "to PYTHONPATH. Repeatable."
        ),
    )
    parser.add_argument(
        "--container-script-dir",
        default=str(DEFAULT_CONTAINER_SCRIPT_DIR),
        help=f"Container directory for the runnable A/B script. Default: {DEFAULT_CONTAINER_SCRIPT_DIR}",
    )
    parser.add_argument(
        "--container-support-dir",
        default=str(DEFAULT_CONTAINER_SUPPORT_DIR),
        help=f"Container directory for copied support files. Default: {DEFAULT_CONTAINER_SUPPORT_DIR}",
    )
    parser.add_argument(
        "--container-trace-root",
        default=str(DEFAULT_CONTAINER_TRACE_ROOT),
        help=f"Container root directory for trace outputs. Default: {DEFAULT_CONTAINER_TRACE_ROOT}",
    )
    parser.add_argument(
        "--trace-filename",
        default=DEFAULT_TRACE_FILENAME,
        help=f"Default trace filename exposed via $TRACE_OUTPUT. Default: {DEFAULT_TRACE_FILENAME}",
    )
    args = parser.parse_args()
    args.repo_root = repo_root
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    args.repo_root = args.repo_root.resolve()
    args.jsonl = resolve_path_from_repo_root(args.jsonl, args.repo_root)
    args.output_dir = resolve_path_from_repo_root(args.output_dir, args.repo_root)
    if args.trace_preamble_file is not None:
        args.trace_preamble_file = resolve_path_from_repo_root(args.trace_preamble_file, args.repo_root)
    if args.trace_epilogue_file is not None:
        args.trace_epilogue_file = resolve_path_from_repo_root(args.trace_epilogue_file, args.repo_root)
    args.support_path = [resolve_path_from_repo_root(path, args.repo_root) for path in args.support_path]

    if not args.jsonl.exists():
        raise ValueError(f"JSONL file does not exist: {args.jsonl}")
    if args.jsonl.suffix != ".jsonl":
        raise ValueError("--jsonl must point to a .jsonl file.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive.")
    if not str(args.namespace).strip():
        raise ValueError("--namespace must be non-empty.")
    if args.trace_preamble_file is not None and not args.trace_preamble_file.exists():
        raise ValueError(f"Trace preamble file does not exist: {args.trace_preamble_file}")
    if args.trace_epilogue_file is not None and not args.trace_epilogue_file.exists():
        raise ValueError(f"Trace epilogue file does not exist: {args.trace_epilogue_file}")
    for support_path in args.support_path:
        if not support_path.exists():
            raise ValueError(f"Support path does not exist: {support_path}")


def load_instance_row(jsonl_path: Path, instance_id: str) -> dict[str, Any]:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get(KEY_INSTANCE_ID) == instance_id:
                return row
    raise KeyError(f"Instance not found in {jsonl_path}: {instance_id}")


def read_text_if_present(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8")


def render_trace_snippet(snippet_text: str, variables: dict[str, str]) -> str:
    if not snippet_text.strip():
        return ""
    return Template(snippet_text).safe_substitute(variables).rstrip()


def inject_script_snippets(script_text: str, preamble: str, epilogue: str) -> str:
    text = script_text.rstrip() + "\n"
    if preamble:
        lines = text.splitlines()
        insert_at = 0
        while insert_at < len(lines):
            stripped = lines[insert_at].strip()
            if not stripped or stripped.startswith("#"):
                insert_at += 1
                continue
            break
        while insert_at < len(lines) and lines[insert_at].strip().startswith("from __future__ import"):
            insert_at += 1

        before = lines[:insert_at]
        after = lines[insert_at:]
        if before and before[-1].strip():
            before.append("")
        before.extend(preamble.splitlines())
        if after and before and before[-1].strip():
            before.append("")
        text = "\n".join(before + after).rstrip() + "\n"

    if epilogue:
        text = text.rstrip() + "\n\n" + epilogue + "\n"
    return text


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def put_path_to_container(container: docker.models.containers.Container, src: Path, dst: PurePosixPath) -> None:
    parent = str(dst.parent)
    container.exec_run(f"mkdir -p {shlex.quote(parent)}", user=DOCKER_USER)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.add(src, arcname=dst.name, recursive=True)
    buffer.seek(0)
    container.put_archive(parent, buffer.read())


def copy_path_from_container(
    container: docker.models.containers.Container,
    src: PurePosixPath,
    dst: Path,
) -> bool:
    try:
        archive_stream, _ = container.get_archive(str(src))
    except (NotFound, APIError):
        return False

    dst = dst.resolve()
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        archive_path = temp_root / "artifact.tar"
        with archive_path.open("wb") as archive_file:
            for chunk in archive_stream:
                archive_file.write(chunk)

        extract_root = temp_root / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, mode="r") as archive:
            archive.extractall(extract_root)

        extracted_path = extract_root / src.name
        if not extracted_path.exists():
            top_entries = list(extract_root.iterdir())
            if len(top_entries) == 1:
                extracted_path = top_entries[0]
            else:
                return False

        if extracted_path.is_dir():
            shutil.copytree(extracted_path, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extracted_path, dst)
        return True


def apply_diff_to_container(
    container: docker.models.containers.Container,
    diff_text: str,
    *,
    diff_name: str,
) -> None:
    if not diff_text.strip():
        return

    dst = PurePosixPath(f"/tmp/{diff_name}.diff")
    write_to_container(container, diff_text, dst)

    last_output = ""
    for apply_cmd in APPLY_CMDS:
        result = container.exec_run(
            f"{apply_cmd} {dst}",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER,
        )
        output_text = result.output.decode(UTF8, errors="replace")
        if result.exit_code == 0:
            return
        last_output = output_text
    raise RuntimeError(f"Failed to apply {diff_name}:\n{last_output}")


def make_trace_variables(
    *,
    instance_id: str,
    script_kind: str,
    trace_dir: PurePosixPath,
    trace_output: PurePosixPath,
    support_root: PurePosixPath,
) -> dict[str, str]:
    return {
        "INSTANCE_ID": instance_id,
        "SCRIPT_KIND": script_kind,
        "TRACE_OUTPUT_DIR": str(trace_dir),
        "TRACE_OUTPUT": str(trace_output),
        "SUPPORT_ROOT": str(support_root),
        "WORKDIR": DOCKER_WORKDIR,
    }


def split_exit_code_from_output(output_text: str) -> tuple[str, int | None]:
    lines = output_text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index].strip()
        if not line.startswith(EXIT_CODE_MARKER):
            continue
        suffix = line[len(EXIT_CODE_MARKER) :]
        try:
            exit_code = int(suffix)
        except ValueError:
            return output_text, None
        cleaned = lines[:index] + lines[index + 1 :]
        cleaned_text = "\n".join(cleaned)
        if output_text.endswith("\n"):
            cleaned_text += "\n"
        return cleaned_text, exit_code
    return output_text, None


def build_combined_script(
    *,
    row: dict[str, Any],
    script_kind: str,
    preamble_template: str,
    epilogue_template: str,
    trace_dir: PurePosixPath,
    trace_output: PurePosixPath,
    support_root: PurePosixPath,
) -> str:
    script_key = f"script_{script_kind}"
    base_script = str(row.get(script_key) or "")
    if not base_script.strip():
        raise ValueError(f"{script_key} is empty for {row[KEY_INSTANCE_ID]}")

    variables = make_trace_variables(
        instance_id=str(row[KEY_INSTANCE_ID]),
        script_kind=script_kind,
        trace_dir=trace_dir,
        trace_output=trace_output,
        support_root=support_root,
    )
    preamble = render_trace_snippet(preamble_template, variables)
    epilogue = render_trace_snippet(epilogue_template, variables)
    return inject_script_snippets(base_script, preamble, epilogue)


def run_script_kind(
    *,
    row: dict[str, Any],
    script_kind: str,
    args: argparse.Namespace,
    client: docker.DockerClient,
    preamble_template: str,
    epilogue_template: str,
) -> dict[str, Any]:
    instance_id = str(row[KEY_INSTANCE_ID])
    host_run_dir = args.output_dir / instance_id / script_kind
    host_run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(instance_id, host_run_dir / "run.log")
    container = None

    container_script_dir = PurePosixPath(args.container_script_dir)
    container_support_dir = PurePosixPath(args.container_support_dir)
    trace_dir = PurePosixPath(args.container_trace_root) / instance_id / script_kind
    trace_output = trace_dir / args.trace_filename
    script_filename = f"{instance_id}_{script_kind}.py".replace("/", "__")
    container_script_path = container_script_dir / script_filename

    combined_script = build_combined_script(
        row=row,
        script_kind=script_kind,
        preamble_template=preamble_template,
        epilogue_template=epilogue_template,
        trace_dir=trace_dir,
        trace_output=trace_output,
        support_root=container_support_dir,
    )
    combined_script_path = host_run_dir / f"script_{script_kind}_combined.py"
    combined_script_path.write_text(combined_script, encoding="utf-8")

    metadata: dict[str, Any] = {
        "instance_id": instance_id,
        "script_kind": script_kind,
        "A_B_Reliability": row.get("A_B_Reliability"),
        "container_script_path": str(container_script_path),
        "container_trace_dir": str(trace_dir),
        "container_trace_output": str(trace_output),
        "combined_script_path": str(combined_script_path),
        "stdout_path": str(host_run_dir / "stdout.txt"),
        "timed_out": False,
        "runtime_seconds": None,
        "exit_code": None,
        "container_id": None,
        "applied_code_patch": not args.skip_code_patch,
        "applied_test_patch": not args.skip_test_patch,
        "support_paths": [str(path) for path in args.support_path],
        "launch_command_template": args.launch_command_template,
    }

    test_spec = make_test_spec(
        row,
        namespace=str(args.namespace),
        env_image_tag=args.env_image_tag,
        instance_image_tag=args.instance_image_tag,
    )

    try:
        container = build_container(
            test_spec=test_spec,
            client=client,
            run_id=f"{args.run_id}-{script_kind.lower()}",
            logger=logger,
            nocache=False,
            force_rebuild=args.force_rebuild,
        )
        container.start()
        metadata["container_id"] = container.id

        container.exec_run(
            "mkdir -p "
            f"{shlex.quote(str(container_script_dir))} "
            f"{shlex.quote(str(container_support_dir))} "
            f"{shlex.quote(str(trace_dir))}",
            user=DOCKER_USER,
        )

        for support_path in args.support_path:
            put_path_to_container(container, support_path, container_support_dir / support_path.name)

        if not args.skip_code_patch:
            apply_diff_to_container(container, str(row.get("patch") or ""), diff_name="code_patch")
        if not args.skip_test_patch:
            apply_diff_to_container(container, str(row.get("test_patch") or ""), diff_name="test_patch")

        write_to_container(container, combined_script, container_script_path)
        launch_variables = make_trace_variables(
            instance_id=instance_id,
            script_kind=script_kind,
            trace_dir=trace_dir,
            trace_output=trace_output,
            support_root=container_support_dir,
        )
        launch_variables["PYTHON_EXE"] = args.python_exe
        launch_variables["SCRIPT_PATH"] = str(container_script_path)
        launch_command = Template(args.launch_command_template).safe_substitute(launch_variables)

        command_parts = [
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            f"cd {shlex.quote(DOCKER_WORKDIR)}",
            f"mkdir -p {shlex.quote(str(trace_dir))}",
            f"export PYTHONPATH={shlex.quote(str(container_support_dir))}:$PYTHONPATH",
            launch_command,
        ]
        shell_command = (
            "("
            + " && ".join(command_parts)
            + f"); status=$?; echo {EXIT_CODE_MARKER}$status; exit $status"
        )
        output_text, timed_out, total_runtime = exec_run_with_timeout(
            container,
            f"/bin/bash -lc {shlex.quote(shell_command)}",
            args.timeout,
        )
        output_text, exit_code = split_exit_code_from_output(output_text)
        metadata["timed_out"] = bool(timed_out)
        metadata["runtime_seconds"] = total_runtime
        metadata["exit_code"] = None if timed_out else exit_code

        stdout_path = host_run_dir / "stdout.txt"
        stdout_path.write_text(output_text, encoding="utf-8")

        copied_trace_dir = host_run_dir / "trace_artifacts"
        trace_dir_present = copy_path_from_container(container, trace_dir, copied_trace_dir)
        metadata["trace_artifacts_copied"] = trace_dir_present
        metadata["trace_artifacts_path"] = str(copied_trace_dir) if trace_dir_present else None

        if timed_out:
            raise TimeoutError(f"{instance_id} {script_kind} timed out after {args.timeout} seconds.")
    finally:
        write_json(host_run_dir / "metadata.json", metadata)
        if not args.keep_container:
            cleanup_container(client, container, logger)
        close_logger(logger)

    return metadata


def main() -> None:
    args = parse_args()
    os.chdir(args.repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    row = load_instance_row(args.jsonl, args.instance_id)
    preamble_template = read_text_if_present(args.trace_preamble_file)
    epilogue_template = read_text_if_present(args.trace_epilogue_file)
    client = docker.from_env()

    script_kinds = ("A", "B") if args.which == "both" else (args.which,)
    results: list[dict[str, Any]] = []
    for script_kind in script_kinds:
        result = run_script_kind(
            row=row,
            script_kind=script_kind,
            args=args,
            client=client,
            preamble_template=preamble_template,
            epilogue_template=epilogue_template,
        )
        results.append(result)

    summary = {
        "instance_id": args.instance_id,
        "which": args.which,
        "results": results,
    }
    write_json(args.output_dir / args.instance_id / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
