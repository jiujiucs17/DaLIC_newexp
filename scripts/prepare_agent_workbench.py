#!/usr/bin/env python3
# Options:
#   --input PATH
#       default: dataset/swebench_feature_with_truth.jsonl
#   --workbench-dir PATH
#       default: dataset/agent_workbench
#   --repo-cache-dir PATH
#       default: dataset/agent_workbench/.repo_mirrors
#   --insta_num INTEGER
#       default: None (prepare all eligible instances)
#   --selected_instances PATH
#       default: None (do not filter instances; otherwise read instance_ids from TOML)

from __future__ import annotations

import argparse
import json
import random
import shlex
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm


EXPECTED_SURE_COUNT = 573


@dataclass(frozen=True)
class InstanceRecord:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()
    default_input = repo_root / "dataset" / "swebench_feature_with_truth.jsonl"
    default_workbench_dir = repo_root / "dataset" / "agent_workbench"
    default_repo_cache_dir = default_workbench_dir / ".repo_mirrors"

    parser = argparse.ArgumentParser(
        description=(
            "Prepare sure SWE-bench instances under dataset/agent_workbench by creating "
            "an instance directory with a checked-out repo and a raw problem_statement.txt."
        )
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--workbench-dir", type=Path, default=default_workbench_dir)
    parser.add_argument(
        "--repo-cache-dir",
        type=Path,
        default=default_repo_cache_dir,
        help="Hidden cache directory for bare upstream mirrors.",
    )
    parser.add_argument(
        "--insta_num",
        type=int,
        default=None,
        help="Number of sure instances to randomly select. Omit to use all sure instances.",
    )
    parser.add_argument(
        "--selected_instances",
        type=Path,
        default=None,
        help="TOML file whose instance_ids list limits the instances to prepare.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    args.input = args.input.expanduser().resolve()
    args.workbench_dir = args.workbench_dir.expanduser().resolve()
    args.repo_cache_dir = args.repo_cache_dir.expanduser().resolve()
    if args.selected_instances is not None:
        args.selected_instances = args.selected_instances.expanduser().resolve()

    if not args.input.exists():
        raise ValueError(f"Input file does not exist: {args.input}")
    if not args.input.is_file():
        raise ValueError(f"Input path is not a file: {args.input}")
    if args.input.suffix != ".jsonl":
        raise ValueError(f"Input file must be a .jsonl file: {args.input}")
    if args.insta_num is not None and args.insta_num <= 0:
        raise ValueError("--insta_num must be a positive integer when provided.")


def require_text_field(row: dict[str, Any], field_name: str, input_path: Path, line_number: int) -> str:
    if field_name not in row or row[field_name] is None:
        raise ValueError(
            f"Missing required field '{field_name}' in {input_path} at line {line_number}."
        )
    return str(row[field_name])


def load_sure_instances(input_path: Path) -> list[InstanceRecord]:
    instances: list[InstanceRecord] = []
    seen_ids: set[str] = set()

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue

            row = json.loads(raw_line)
            if row.get("ground_truth_certainty") != "sure":
                continue

            instance = InstanceRecord(
                instance_id=require_text_field(row, "instance_id", input_path, line_number),
                repo=require_text_field(row, "repo", input_path, line_number),
                base_commit=require_text_field(row, "base_commit", input_path, line_number),
                problem_statement=require_text_field(
                    row, "problem_statement", input_path, line_number
                ),
                patch=str(row.get("patch") or ""),
                test_patch=str(row.get("test_patch") or ""),
            )

            if instance.instance_id in seen_ids:
                raise ValueError(f"Duplicate sure instance_id found: {instance.instance_id}")
            seen_ids.add(instance.instance_id)
            instances.append(instance)

    return instances


def load_selected_instance_ids(selected_instances_path: Path) -> set[str]:
    with selected_instances_path.open("rb") as selected_instances_file:
        payload = tomllib.load(selected_instances_file)
    return set(payload["instance_ids"])


def filter_selected_instances(
    instances: list[InstanceRecord],
    selected_instance_ids: set[str],
) -> list[InstanceRecord]:
    return [
        instance
        for instance in instances
        if instance.instance_id in selected_instance_ids
    ]


def select_instances(
    instances: list[InstanceRecord],
    insta_num: int | None,
) -> tuple[list[InstanceRecord], int | None]:
    if insta_num is None:
        return list(instances), None

    sample_size = min(insta_num, len(instances))
    seed = time.time_ns()
    rng = random.Random(seed)
    return rng.sample(instances, sample_size), seed


def repo_cache_path(repo_cache_dir: Path, repo: str) -> Path:
    repo_slug = repo.replace("/", "__")
    return repo_cache_dir / f"{repo_slug}.git"


def repo_name_from_instance_id(instance_id: str) -> str:
    owner_and_repo = instance_id.split("__", 1)
    if len(owner_and_repo) != 2 or not owner_and_repo[1]:
        raise ValueError(f"Invalid instance_id format: {instance_id}")

    repo_name_and_pr = owner_and_repo[1].rsplit("-", 1)
    if len(repo_name_and_pr) != 2 or not repo_name_and_pr[0] or not repo_name_and_pr[1]:
        raise ValueError(f"Invalid instance_id format: {instance_id}")

    return repo_name_and_pr[0]


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return

    message = completed.stderr.strip() or completed.stdout.strip() or "Unknown command failure."
    raise RuntimeError(f"{shlex.join(command)} failed: {message}")


def ensure_repo_mirror(repo: str, repo_cache_dir: Path) -> Path:
    mirror_path = repo_cache_path(repo_cache_dir, repo)
    if mirror_path.exists():
        if not mirror_path.is_dir():
            raise ValueError(f"Repo cache path exists but is not a directory: {mirror_path}")
        return mirror_path

    repo_cache_dir.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{repo}.git"
    run_command(["git", "clone", "--mirror", repo_url, str(mirror_path)])
    return mirror_path


def create_repo_snapshot(instance: InstanceRecord, repo_dir: Path, repo_cache_dir: Path) -> None:
    mirror_path = ensure_repo_mirror(instance.repo, repo_cache_dir)
    repo_url = f"https://github.com/{instance.repo}.git"

    run_command(["git", "init", str(repo_dir)])
    run_command(["git", "-C", str(repo_dir), "remote", "add", "origin", repo_url])
    run_command(
        [
            "git",
            "-C",
            str(repo_dir),
            "fetch",
            "--depth",
            "1",
            str(mirror_path),
            instance.base_commit,
        ]
    )
    run_command(["git", "-C", str(repo_dir), "checkout", "--detach", "FETCH_HEAD"])
    apply_patch_text(repo_dir, instance.patch, "patch")
    apply_patch_text(repo_dir, instance.test_patch, "test_patch")


def apply_patch_text(repo_dir: Path, patch_text: str, patch_name: str) -> None:
    if not patch_text.strip():
        return

    try:
        run_command(
            ["git", "-C", str(repo_dir), "apply", "--whitespace=nowarn", "-"],
            input_text=patch_text,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to apply {patch_name}: {exc}") from exc


def clean_repo_artifacts(repo_dir: Path) -> None:
    git_dir = repo_dir / ".git"
    if not git_dir.exists():
        raise ValueError(f"Expected git metadata directory is missing: {git_dir}")
    if not git_dir.is_dir():
        raise ValueError(f"Expected .git to be a directory: {git_dir}")
    shutil.rmtree(git_dir)

    hg_dir = repo_dir / ".hg"
    if hg_dir.exists():
        if not hg_dir.is_dir():
            raise ValueError(f"Expected .hg to be a directory: {hg_dir}")
        shutil.rmtree(hg_dir)

    github_dir = repo_dir / ".github"
    if github_dir.exists():
        if not github_dir.is_dir():
            raise ValueError(f"Expected .github to be a directory: {github_dir}")
        shutil.rmtree(github_dir)

    for pattern in ("*.patch", "*.diff"):
        for artifact_path in repo_dir.rglob(pattern):
            if artifact_path.is_file():
                artifact_path.unlink()


def prepare_instance(
    instance: InstanceRecord,
    workbench_dir: Path,
    repo_cache_dir: Path,
) -> str:
    instance_dir = workbench_dir / instance.instance_id
    if instance_dir.exists():
        if instance_dir.is_dir():
            return "skipped"
        raise ValueError(f"Instance path exists but is not a directory: {instance_dir}")

    temp_dir = workbench_dir / f".{instance.instance_id}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        repo_dir = temp_dir / "repo" / repo_name_from_instance_id(instance.instance_id)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        create_repo_snapshot(instance, repo_dir, repo_cache_dir)
        clean_repo_artifacts(repo_dir)
        # (temp_dir / "mini-swe-agent-results").mkdir()
        (temp_dir / "problem_statement.txt").write_text(
            instance.problem_statement,
            encoding="utf-8",
        )
        temp_dir.replace(instance_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return "prepared"


def main() -> None:
    args = parse_args()
    sure_instances = load_sure_instances(args.input)
    candidate_instances = sure_instances
    if args.selected_instances is not None:
        selected_instance_ids = load_selected_instance_ids(args.selected_instances)
        candidate_instances = filter_selected_instances(
            sure_instances,
            selected_instance_ids,
        )
    selected_instances, selection_seed = select_instances(
        candidate_instances,
        args.insta_num,
    )

    args.workbench_dir.mkdir(parents=True, exist_ok=True)

    sure_count = len(sure_instances)
    print(f"Loaded {sure_count} sure instances from {args.input}.")
    if sure_count != EXPECTED_SURE_COUNT:
        print(
            f"Warning: expected {EXPECTED_SURE_COUNT} sure instances but found {sure_count}."
        )
    candidate_count = len(candidate_instances)
    if args.insta_num is not None and args.insta_num > candidate_count:
        print(
            f"Requested {args.insta_num} instances but only {candidate_count} eligible "
            "instances are available."
        )
    # if selection_seed is not None:
    #     print(f"Random selection seed: {selection_seed}")

    prepared_count = 0
    skipped_count = 0
    failures: list[tuple[str, str]] = []

    for instance in tqdm(selected_instances, desc="Preparing instances", unit="instance"):
        try:
            status = prepare_instance(instance, args.workbench_dir, args.repo_cache_dir)
        except Exception as exc:
            failures.append((instance.instance_id, str(exc)))
            tqdm.write(f"[failed] {instance.instance_id}: {exc}")
            continue

        if status == "prepared":
            prepared_count += 1
        else:
            skipped_count += 1

    print(f"Selected {len(selected_instances)} instances.")
    print(f"Prepared {prepared_count} instance directories.")
    print(f"Skipped {skipped_count} existing instance directories.")

    if failures:
        print(f"Failed to prepare {len(failures)} instances.")
        for instance_id, message in failures[:20]:
            print(f"- {instance_id}: {message}")
        if len(failures) > 20:
            print(f"- ... and {len(failures) - 20} more failures.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
