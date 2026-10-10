#!/usr/bin/env python3
"""Preview removal of numbered -run labels from three result folders' logs.

Use --apply to back up and modify logs. Only matching bytes in .log files
change; directory names and all other files are left untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOTS = tuple(
    REPO_ROOT / "dataset" / "agent_results" / name
    for name in ("baseline", "dalic_no_runtime", "dalic_vsm")
)
RUN_LABEL = re.compile(rb"-run[0-9]+\b")


@dataclass(frozen=True)
class Change:
    path: Path
    original_hash: bytes
    updated_hash: bytes
    labels: Counter[bytes]


def digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def read_original(change: Change) -> bytes:
    if change.path.is_symlink():
        raise RuntimeError(f"Log became a symlink: {change.path}")
    data = change.path.read_bytes()
    if digest(data) != change.original_hash:
        raise RuntimeError(f"Log changed after preview: {change.path}")
    return data


def scan() -> tuple[int, list[Change]]:
    scanned = 0
    changes = []
    for root in RESULT_ROOTS:
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"Expected a real results directory: {root}")
        for path in sorted(root.rglob("*.log")):
            if path.is_symlink() or not path.is_file():
                continue
            scanned += 1
            original = path.read_bytes()
            labels = Counter(RUN_LABEL.findall(original))
            if labels:
                changes.append(Change(
                    path, digest(original), digest(RUN_LABEL.sub(b"", original)), labels
                ))
    return scanned, changes


def back_up(changes: list[Change], destination: Path) -> None:
    # Exclusive creation prevents overwriting a previous backup.
    with destination.open("xb") as output:
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for change in changes:
                original = read_original(change)
                info = archive.gettarinfo(
                    str(change.path), arcname=str(change.path.relative_to(REPO_ROOT))
                )
                info.size = len(original)
                archive.addfile(info, io.BytesIO(original))
    # Check the backup bytes before modifying any logs.
    with tarfile.open(destination, "r:gz") as archive:
        for change in changes:
            member = archive.extractfile(str(change.path.relative_to(REPO_ROOT)))
            if member is None:
                raise RuntimeError(f"Missing backup entry: {change.path}")
            with member:
                if digest(member.read()) != change.original_hash:
                    raise RuntimeError(f"Backup verification failed: {change.path}")


def apply_change(change: Change) -> None:
    updated = RUN_LABEL.sub(b"", read_original(change))
    if digest(updated) != change.updated_hash:
        raise RuntimeError(f"Unexpected replacement result: {change.path}")
    mode = stat.S_IMODE(change.path.stat().st_mode)
    # Replace each log atomically, preserving its permission bits.
    with tempfile.NamedTemporaryFile(dir=change.path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(updated)
            output.flush()
            os.fchmod(output.fileno(), mode)
            read_original(change)
            os.replace(temporary, change.path)
        finally:
            temporary.unlink(missing_ok=True)
    if change.path.read_bytes() != updated or RUN_LABEL.search(updated):
        raise RuntimeError(f"Written log verification failed: {change.path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up and edit logs.")
    parser.add_argument("--backup", type=Path, help="New .tar.gz backup path for --apply.")
    args = parser.parse_args()
    if args.backup is not None and not args.apply:
        parser.error("--backup requires --apply")

    scanned, changes = scan()
    count = sum(sum(change.labels.values()) for change in changes)
    print(f"Scanned {scanned} logs; {len(changes)} files contain {count} labels.")
    for root in RESULT_ROOTS:
        selected = [change for change in changes if change.path.is_relative_to(root)]
        print(f"{root.name}: {len(selected)} files, "
              f"{sum(sum(change.labels.values()) for change in selected)} labels")
    for change in changes:
        labels = ", ".join(
            f"{label.decode('ascii')}: {count}"
            for label, count in sorted(change.labels.items())
        )
        print(f"{change.path.relative_to(REPO_ROOT)} | {labels}")
    if not args.apply:
        print("Preview only. Use --apply to back up and edit these logs.")
        return
    if not changes:
        print("No changes needed.")
        return

    backup = args.backup
    if backup is None:
        backup = Path(tempfile.mkdtemp(prefix="dalic-log-run-labels-")) / "original_logs.tar.gz"
    print(f"Backup: {backup.resolve()}", flush=True)
    back_up(changes, backup)
    # Detect changes during backup before starting the edits.
    for change in changes:
        read_original(change)
    for change in changes:
        apply_change(change)
    print(f"Updated and verified {len(changes)} logs; removed {count} labels.")


if __name__ == "__main__":
    main()
