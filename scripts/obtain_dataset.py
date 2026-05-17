#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from datasets import load_dataset
from tqdm import tqdm


DEFAULT_DATASET = "SWE-bench/SWE-bench"
DEFAULT_SPLIT = "test"
OPTIONAL_PASSTHROUGH_FIELDS = (
    "hints_text",
    "created_at",
    "version",
    "environment_setup_commit",
    "issue_id",
    "issue_url",
    "pr_url",
    "difficulty",
)


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(
        description="Load a SWE-bench dataset split and normalize it into the internal jsonl schema."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "dataset" / "swebench.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    args.output = args.output.expanduser().resolve()
    if args.output.suffix != ".jsonl":
        raise ValueError("--output must be a .jsonl file.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided.")


def normalize_list_field(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, tuple):
        return [str(item) for item in value]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if parsed is None:
            return []
        return [str(parsed)]

    return [str(value)]


def get_required_field(row: dict[str, Any], field_name: str) -> Any:
    if field_name not in row:
        raise ValueError(f"Missing required field '{field_name}' in source row: {row}")
    return row[field_name]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "instance_id": str(get_required_field(row, "instance_id")),
        "repo": str(get_required_field(row, "repo")),
        "base_commit": str(get_required_field(row, "base_commit")),
        "problem_statement": str(get_required_field(row, "problem_statement")),
        "patch": str(row.get("patch") or ""),
        "test_patch": str(row.get("test_patch") or ""),
        "FAIL_TO_PASS": normalize_list_field(row.get("FAIL_TO_PASS")),
        "PASS_TO_PASS": normalize_list_field(row.get("PASS_TO_PASS")),
    }

    for field_name in OPTIONAL_PASSTHROUGH_FIELDS:
        if field_name in row and row[field_name] is not None:
            normalized[field_name] = row[field_name]

    return normalized


def export_dataset(dataset_name: str, split: str, output_path: Path, limit: int | None) -> None:
    dataset = load_dataset(dataset_name, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(len(dataset), limit)))
    total_rows = len(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            progress_bar = tqdm(dataset, total=total_rows, desc="Exporting rows", unit="row")
            for row in progress_bar:
                temp_file.write(json.dumps(normalize_row(row), ensure_ascii=False) + "\n")

        if temp_path is None:
            raise RuntimeError("Temporary output file was not created.")
        temp_path.replace(output_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def main() -> None:
    args = parse_args()
    export_dataset(
        dataset_name=args.dataset,
        split=args.split,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
