from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_retry_results import MergeConfig, MergeError, merge_results, preflight


def write_toml(path: Path, instance_ids: list[str]) -> None:
    entries = "\n".join(f'  "{instance_id}",' for instance_id in instance_ids)
    path.write_text(f"instance_ids = [\n{entries}\n]\n", encoding="utf-8")


def write_summary(path: Path, successful: int, failures: dict[str, str]) -> None:
    lines = [
        f"success_count: {successful}",
        f"failure_count: {len(failures)}",
        "failed_instances:",
        *(failures or ["None"]),
        "failure_reasons:",
    ]
    if failures:
        lines.extend(f"{instance_id}: {reason}" for instance_id, reason in failures.items())
    else:
        lines.append("None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_instance(root: Path, instance_id: str, label: str, *, credit_error: bool = False) -> None:
    instance_root = root / instance_id
    instance_root.mkdir(parents=True)
    instance_root.joinpath("results.txt").write_text(
        json.dumps(
            {
                "results": {
                    "files": [f"{label}.py"],
                    "classes": [],
                    "functions": [],
                }
            }
        ),
        encoding="utf-8",
    )
    instance_root.joinpath("last_mini_run.traj.json").write_text(
        json.dumps(
            {
                "info": {
                    "model_stats": {"instance_cost": 0.1, "api_calls": 2},
                    "exit_status": "Submitted",
                }
            }
        ),
        encoding="utf-8",
    )
    docker_log = f"source={label}\n"
    if credit_error:
        docker_log += "credit_balance_exhausted\n"
    instance_root.joinpath("docker_run.log").write_text(docker_log, encoding="utf-8")


def write_run_log(root: Path, method: str, instance_labels: list[tuple[str, str]]) -> None:
    lines: list[str] = []
    for index, (instance_id, label) in enumerate(instance_labels, start=1):
        lines.extend(
            [
                f"2026-09-04 00:00:{index:02d},000 INFO [{instance_id}] Starting {method} flow.",
                f"2026-09-04 00:00:{index:02d},001 INFO [{instance_id}] source={label}",
                f"2026-09-04 00:00:{index:02d},002 INFO [{instance_id}] Finished successfully.",
            ]
        )
    lines.append(
        f"2026-09-04 00:01:00,000 INFO Run finished. success_count={len(instance_labels)} failure_count=0"
    )
    root.joinpath(f"{method}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


class MergeRetryResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.original_root = self.root / "original"
        self.retry_root = self.root / "retry"
        self.output_root = self.root / "merged"
        self.original_root.mkdir()
        self.retry_root.mkdir()
        self.selected_path = self.root / "selected.toml"
        self.retry_path = self.root / "retry.toml"
        write_toml(self.selected_path, ["instance-a", "instance-b"])
        write_toml(self.retry_path, ["instance-b"])

        write_instance(self.original_root, "instance-a", "original-a")
        write_instance(
            self.original_root, "instance-b", "original-b", credit_error=True
        )
        write_instance(self.retry_root, "instance-b", "retry-b")
        write_summary(
            self.original_root / "summary.txt",
            successful=1,
            failures={"instance-b": "docker run failed; artifact copy failed"},
        )
        write_summary(self.retry_root / "summary.txt", successful=1, failures={})
        write_run_log(
            self.original_root,
            "baseline",
            [("instance-a", "original-a"), ("instance-b", "original-b")],
        )
        write_run_log(
            self.retry_root, "baseline", [("instance-b", "retry-b")]
        )
        self.config = MergeConfig(
            repo_root=self.root,
            method="baseline",
            original_root=self.original_root,
            retry_root=self.retry_root,
            output_root=self.output_root,
            retry_instances_path=self.retry_path,
            selected_instances_path=self.selected_path,
            truth_input_path=self.root / "truth.jsonl",
            workbench_path=self.root / "workbench",
            original_error_marker="credit_balance_exhausted",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def fake_evaluator(config: MergeConfig, output_root: Path) -> None:
        output_root.joinpath("evaluation.txt").write_text(
            "\n".join(
                [
                    "sure_ground_truth_instances: 2",
                    "successful_instances: 2",
                    "failed_instances: 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_merge_uses_retry_attempt_and_preserves_provenance(self) -> None:
        original_before = self.original_root.joinpath(
            "instance-b", "results.txt"
        ).read_bytes()
        retry_before = self.retry_root.joinpath("instance-b", "results.txt").read_bytes()

        output = merge_results(self.config, evaluator=self.fake_evaluator)

        self.assertEqual(output, self.output_root)
        self.assertEqual(
            output.joinpath("instance-a", "results.txt").read_bytes(),
            self.original_root.joinpath("instance-a", "results.txt").read_bytes(),
        )
        self.assertEqual(
            output.joinpath("instance-b", "results.txt").read_bytes(), retry_before
        )
        self.assertEqual(
            self.original_root.joinpath("instance-b", "results.txt").read_bytes(),
            original_before,
        )
        self.assertEqual(
            self.retry_root.joinpath("instance-b", "results.txt").read_bytes(),
            retry_before,
        )
        self.assertIn("success_count: 2", output.joinpath("summary.txt").read_text())
        self.assertTrue(output.joinpath("evaluation.txt").is_file())
        self.assertTrue(
            output.joinpath("_provenance", "baseline.original.log").is_file()
        )
        self.assertTrue(
            output.joinpath("_provenance", "baseline.retry.log").is_file()
        )

        manifest = json.loads(output.joinpath("merge_manifest.json").read_text())
        self.assertEqual(
            manifest["instances"]["instance-a"]["chosen_attempt"], "original"
        )
        self.assertEqual(
            manifest["instances"]["instance-b"]["chosen_attempt"], "retry"
        )
        self.assertEqual(manifest["trajectory_totals"]["trajectory_count"], 2)
        aggregate_log = output.joinpath("baseline.log").read_text()
        self.assertIn("source=original-a", aggregate_log)
        self.assertIn("source=retry-b", aggregate_log)
        self.assertNotIn("source=original-b", aggregate_log)

    def test_preflight_rejects_incomplete_retry(self) -> None:
        write_summary(self.retry_root / "summary.txt", successful=0, failures={})

        with self.assertRaisesRegex(MergeError, "Retry run is incomplete"):
            preflight(self.config)
        self.assertFalse(self.output_root.exists())

    def test_preflight_requires_exact_error_marker_set(self) -> None:
        self.original_root.joinpath("instance-a", "docker_run.log").write_text(
            "credit_balance_exhausted\n", encoding="utf-8"
        )
        write_summary(
            self.original_root / "summary.txt",
            successful=0,
            failures={
                "instance-a": "docker run failed",
                "instance-b": "docker run failed",
            },
        )

        with self.assertRaisesRegex(MergeError, "does not exactly match"):
            preflight(self.config)

    def test_merge_refuses_to_overwrite_output(self) -> None:
        self.output_root.mkdir()
        marker = self.output_root / "keep.txt"
        marker.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(MergeError, "will not be overwritten"):
            merge_results(self.config, evaluator=self.fake_evaluator)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
