from __future__ import annotations

import copy
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate as ev


def truth(files=("a.py",), classes=("a.py::A",), functions=("a.py::f",)):
    return ev.normalize_ground_truth({
        "added": {},
        "edited": {"files": list(files), "classes": list(classes), "functions": list(functions)},
    })


def item(instance_id, *, gt=None, valid=True, hit=True, ranked=None):
    gt = truth() if gt is None else gt
    if ranked is None:
        ranked = {g: sorted(gt["total"][g]) if hit and valid else [] for g in ev.GRANULARITIES}
    return ev.InstanceEvaluation(
        instance_id, {g: set(ranked[g]) for g in ev.GRANULARITIES}, gt,
        not valid, None if valid else "missing output", ranked,
    )


def report(items, method="baseline", model="model"):
    return ev.build_run_report(items, method=method, model=model)


class ScoringTests(unittest.TestCase):
    def test_failure_overrides_double_empty_for_new_and_legacy_scores(self):
        gt = truth(classes=(), functions=())
        failed = item("case", gt=gt, valid=False)
        row = ev.instance_record(failed)
        for g in ev.FEATURES:
            self.assertEqual(row["metrics"][g]["f1"], 0.0)
        self.assertIsNone(row["metrics"]["classes"]["ap"])
        self.assertEqual(row["metrics"]["files"]["ap"], 0)
        legacy = ev.build_macro_metrics([failed], ignore_failures=False)
        self.assertEqual(legacy["total"]["classes"].f1, 0)
        valid = ev.score_ranking([], set(), True)
        self.assertEqual([valid[m] for m in ("precision", "recall", "f1")], [1, 1, 1])
        for ranked, gt in ((["wrong"], set()), ([], {"a"})):
            score = ev.score_ranking(ranked, gt, True)
            self.assertEqual([score[m] for m in ("precision", "recall", "f1")], [0, 0, 0])

    def test_ap_preserves_order_and_full_ground_truth_denominator(self):
        self.assertEqual(ev.average_precision(["wrong", "a"], {"a", "b"}), 0.25)
        self.assertAlmostEqual(ev.average_precision(["wrong", "a", "b"], {"a", "b"}), 7 / 12)
        self.assertEqual(ev.average_precision(["a", "wrong"], {"a"}), 1)
        self.assertEqual(ev.average_precision(["wrong", "a"], {"a"}), 0.5)
        self.assertIsNone(ev.average_precision([], set()))
        with self.assertRaises(ValueError):
            ev.score_ranking(["a", "a"], {"a"}, True)
        unordered = ev.InstanceEvaluation("case", {g: set() for g in ev.GRANULARITIES}, truth(), False)
        with self.assertRaisesRegex(ValueError, "Ordered"):
            ev.instance_record(unordered)

    def test_ranking_eligibility_is_per_granularity_and_avg3_is_complete_case(self):
        rows = [
            ev.instance_record(item("a", hit=False)),
            ev.instance_record(item("b", gt=truth(classes=()))),
            ev.instance_record(item("c", gt=truth(functions=()))),
        ]
        result = ev.summarize_records(rows)
        self.assertEqual([result["metrics"][g]["ranking_n"] for g in ev.FEATURES], [3, 2, 2, 1])
        self.assertEqual(result["metrics"]["avg3"]["map"], 0)
        self.assertAlmostEqual(result["metrics"]["files"]["map"], 2 / 3)
        self.assertAlmostEqual(result["metrics"]["classes"]["map"], 0.5)
        self.assertAlmostEqual(result["metrics"]["avg3"]["f1"], 2 / 3)
        empty = ev.summarize_records([])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["metrics"]["avg3"]["f1"])
        self.assertIsNone(empty["valid_rate"])

    def test_macro_f1_is_not_harmonic_mean_of_macro_precision_recall(self):
        rows = [
            ev.instance_record(item("a", gt=truth(files=("a.py", "b.py")),
                                    ranked={"files": ["a.py"], "classes": [], "functions": []})),
            ev.instance_record(item("b", gt=truth(files=("a.py",)),
                                    ranked={"files": ["a.py", "wrong.py"], "classes": [], "functions": []})),
        ]
        files = ev.summarize_records(rows)["metrics"]["files"]
        self.assertAlmostEqual(files["precision"], 0.75)
        self.assertAlmostEqual(files["recall"], 0.75)
        self.assertAlmostEqual(files["f1"], 2 / 3)


class ParsingTests(unittest.TestCase):
    def test_schema_duplicates_and_path_rules(self):
        payload = {"results": {"files": ["b.py", "a.py"], "classes": [], "functions": []}}
        ranked, error = ev.validate_ranked_payload(payload)
        self.assertIsNone(error)
        self.assertEqual(ranked["files"], ["b.py", "a.py"])
        for path in ("./a.py", "a/../b.py", "a//b.py", "/a.py", "repo/a.py", "a\\b.py"):
            altered = copy.deepcopy(payload)
            altered["results"]["files"] = [path]
            self.assertIsNotNone(ev.validate_ranked_payload(altered)[1], path)
        altered = copy.deepcopy(payload)
        altered["results"]["files"] = ["a.py", "a.py"]
        self.assertIsNotNone(ev.validate_ranked_payload(altered)[1])
        altered = copy.deepcopy(payload)
        altered["results"]["classes"] = ["a.py"]
        self.assertIsNotNone(ev.validate_ranked_payload(altered)[1])
        self.assertIsNotNone(ev.validate_ranked_payload({"results": {}, "extra": 1})[1])

    def test_bad_json_is_failure_but_filesystem_errors_are_not_agent_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "results.txt"
            self.assertEqual(ev.read_ranked_results(path, root)[1], "results.txt is missing")
            for raw in (b"\xff", b"not json", b'{"results": {}, "results": {}}'):
                path.write_bytes(raw)
                ranked, reason, digest = ev.read_ranked_results(path, root)
                self.assertIsNone(ranked)
                self.assertIsNotNone(reason)
                self.assertEqual(len(digest), 64)
            with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    ev.read_ranked_results(path, root)

    def test_manifest_and_ground_truth_errors_do_not_silently_change_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "selected.toml"
            manifest.write_text('instance_ids = ["a", "a"]')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                ev.load_selected_instance_ids(manifest)
            path = root / "truth.jsonl"
            row = {"instance_id": "a", "ground_truth_certainty": "sure",
                   "ground_truth": {"added": {}, "edited": {}}}
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "lack sure"):
                ev.load_ground_truth_map(path, {"a", "b"})
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                ev.load_ground_truth_map(path, {"a"})
            row["ground_truth_certainty"] = "uncertain"
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "lack sure"):
                ev.load_ground_truth_map(path, {"a"})
        with self.assertRaisesRegex(ValueError, "outside"):
            ev.validate_benchmark_ids({"not-in-the-validated-cohort"})
        with self.assertRaises(ValueError):
            ev.normalize_ground_truth({"added": {"files": [123]}})


class ComparisonTests(unittest.TestCase):
    def compare(self, baseline, candidate, **kwargs):
        return ev.compare_run_reports(
            baseline, candidate, bootstrap_draws=200, permutation_draws=500,
            **kwargs,
        )

    def test_pairing_common_success_completion_and_determinism(self):
        baseline = report([item("a"), item("b", valid=False), item("c"), item("d", valid=False)])
        candidate = report([item("d", valid=False), item("c", valid=False), item("b"), item("a", hit=False)],
                           method="dalic")
        result = self.compare(baseline, candidate, tests="paired")
        self.assertEqual(result["completion"]["cells"],
                         dict(both_valid=1, baseline_only=1, candidate_only=1, both_invalid=1))
        self.assertEqual(result["completion"]["effect"]["estimate"], 0)
        self.assertEqual(result["tests"]["completion"]["raw_p"], 1)
        common = result["scopes"]["common_success"]
        self.assertEqual(common["baseline"]["instance_ids"], ["a"])
        self.assertEqual(common["effects"]["f1"]["avg3"]["estimate"], -1)
        self.assertEqual(result["scopes"]["all_instances"]["effects"]["f1"]["avg3"]["estimate"], -0.25)
        self.assertEqual(result, self.compare(baseline, candidate, tests="paired"))
        self.assertEqual(self.compare(baseline, candidate)["tests"], {})

    def test_empty_common_success_and_empty_ranking_are_undefined(self):
        gt = truth(classes=())
        result = self.compare(report([item("a", gt=gt, valid=False)]),
                              report([item("a", gt=gt)], method="dalic"), tests="paired")
        self.assertEqual(result["scopes"]["common_success"]["baseline"]["n"], 0)
        self.assertIsNone(result["scopes"]["common_success"]["effects"]["f1"]["avg3"]["ci_95"])
        self.assertIsNone(result["scopes"]["all_instances"]["effects"]["map"]["avg3"]["estimate"])
        self.assertIsNone(result["tests"]["map"]["raw_p"])

    def test_incompatible_cohorts_and_truth_are_rejected(self):
        baseline = report([item("a")])
        with self.assertRaisesRegex(ValueError, "identical"):
            self.compare(baseline, report([item("b")]))
        with self.assertRaisesRegex(ValueError, "Ground truth"):
            self.compare(baseline, report([item("a", gt=truth(files=("other.py",)))]))

    def test_exact_and_randomized_statistical_corner_cases(self):
        self.assertEqual(ev.exact_mcnemar(0, 0), 1)
        self.assertEqual(ev.exact_mcnemar(0, 9), 2 / 512)
        self.assertEqual(ev.sign_flip_test([0, 0], draws=100, seed=1)["raw_p"], 1)
        self.assertEqual(ev.sign_flip_test([1], draws=100, seed=1)["raw_p"], 1)
        self.assertIsNone(ev.sign_flip_test([], draws=100, seed=1)["raw_p"])
        self.assertGreater(ev.sign_flip_test([1] * 30, draws=100, seed=1)["raw_p"], 0)
        self.assertEqual(ev.holm_adjust({"a": 0.001, "b": 0.02, "c": 0.04}),
                         {"a": 0.003, "b": 0.04, "c": 0.04})
        with self.assertRaises(ValueError):
            ev.holm_adjust({"missing": None})
        interval = ev.bootstrap_differences([[0.0], [1.0]], ["x"], draws=1000, seed=1)["x"]
        self.assertEqual(interval["estimate"], 0.5)
        self.assertEqual(interval["ci_95"], [0, 1])

    def test_json_reload_rejects_tampered_scores(self):
        instance_id = sorted(ev.load_selected_instance_ids(ev.get_repo_root() / ev.DEFAULT_SELECTED_INSTANCES_PATH))[0]
        value = report([item(instance_id)])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation.json"
            path.write_text(json.dumps(value))
            self.assertEqual(ev.load_run_report(path), value)
            value["instances"][0]["metrics"]["avg3"]["f1"] = 0.123
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Saved scores"):
                ev.load_run_report(path)


class PaperTests(unittest.TestCase):
    def test_cutoffs_failure_fallback_and_complete_holm_families(self):
        ids = ["a", "b", "c"]
        gt = {i: truth() for i in ids}
        models = ["one", "two"]
        agents = {method: {} for method in ("baseline", "dalic")}
        for model in models:
            agents["baseline"][model] = report([item(i) for i in ids], model=model)
            agents["dalic"][model] = report(
                [item("a", hit=False), item("b", valid=False), item("c")], method="dalic", model=model,
            )
        rankings = {
            method: {i: {g: ["wrong.py" if g == "files" else "wrong.py::X", *sorted(gt[i]["total"][g])]
                         for g in ev.GRANULARITIES} for i in ids}
            for method in (*ev.STATIC_METHODS, "trace_bm25")
        }
        cutoffs = ev.matched_cutoffs(agents["dalic"]["one"]["instances"])
        self.assertEqual(cutoffs["a"]["files"], 0)
        self.assertEqual(cutoffs["b"]["files"], 3)
        self.assertEqual(cutoffs["c"]["files"], 1)
        rows = ev.retrieval_records(rankings["trace_bm25"], gt, cutoffs)
        self.assertEqual([r["metrics"]["files"]["prediction_count"] for r in rows], [0, 2, 1])
        result = ev.build_paper_report(
            agents, rankings, gt, models=models, bootstrap_draws=50, permutation_draws=100,
        )
        self.assertEqual(set(result["rq1_static_trace"]), {"3", "5", "10", "20"})
        self.assertEqual(set(result["rq2_static_dalic"]["one"]["fixed"]), set(ev.STATIC_METHODS))
        for pair in result["rq3_baseline_dalic"].values():
            for test in pair["tests"].values():
                self.assertEqual(test["family_size"], 2)
                self.assertGreaterEqual(test["holm_p"], test["raw_p"])
        with self.assertRaises(ValueError):
            ev.build_paper_report(agents, rankings, gt, models=["one", "one"])

    def test_missing_failed_and_duplicate_retrieval_artifacts_are_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in (*ev.STATIC_METHODS, "trace_bm25"):
                directory = root / method
                directory.mkdir()
                row = {"method": ev.RETRIEVAL_LABELS[method], "instance_id": "a", "failure_reason": None,
                       "rankings": {g: [] for g in ev.GRANULARITIES}}
                with gzip.open(directory / "a.json.gz", "wt") as f:
                    json.dump(row, f)
            rankings, sources = ev.load_retrieval_rankings(root, ["a"])
            self.assertEqual(rankings["static_vsm"]["a"]["files"], [])
            self.assertEqual(len(sources["static_vsm"]["a"]["sha256"]), 64)
            path = root / "static_vsm/a.json.gz"
            row.update(method="static_vsm", failure_reason="retrieval failed")
            with gzip.open(path, "wt") as f:
                json.dump(row, f)
            with self.assertRaisesRegex(ValueError, "failed retrieval"):
                ev.load_retrieval_rankings(root, ["a"])
            row.update(failure_reason=None)
            row["rankings"]["files"] = ["a.py", "a.py"]
            with gzip.open(path, "wt") as f:
                json.dump(row, f)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                ev.load_retrieval_rankings(root, ["a"])
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                ev.load_retrieval_rankings(root, ["a"])


class CommandTests(unittest.TestCase):
    def test_legacy_run_then_compare_cli(self):
        selected = sorted(ev.load_selected_instance_ids(ev.get_repo_root() / ev.DEFAULT_SELECTED_INSTANCES_PATH))[:2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, workbench = root / "baseline/model", root / "workbench"
            results.mkdir(parents=True)
            workbench.mkdir()
            # One valid run, one missing instance directory.
            (results / selected[0]).mkdir()
            (workbench / selected[0] / "repo/project").mkdir(parents=True)
            (results / selected[0] / "results.txt").write_text(json.dumps({
                "results": {"files": ["a.py"], "classes": [], "functions": []},
            }))
            manifest, truth_path = root / "selected.toml", root / "truth.jsonl"
            manifest.write_text("instance_ids = " + json.dumps(selected))
            truth_path.write_text("".join(json.dumps({
                "instance_id": i, "ground_truth_certainty": "sure",
                "ground_truth": {"added": {}, "edited": {"files": ["a.py"], "classes": [], "functions": []}},
            }) + "\n" for i in selected))
            script = ev.get_repo_root() / "scripts/evaluate.py"
            command = [sys.executable, "-B", str(script), "--result_folder", str(results),
                       "--workbench_path", str(workbench), "--input", str(truth_path),
                       "--selected_instances", str(manifest)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            output = results / "evaluation.json"
            data = ev.load_run_report(output)
            self.assertEqual(data["scopes"]["all_instances"]["valid_n"], 1)
            self.assertEqual(data["scopes"]["all_instances"]["metrics"]["avg3"]["f1"], 0.5)
            text = (results / "evaluation.txt").read_text()
            self.assertIn("sure_ground_truth_instances: 2", text)
            self.assertIn("successful_instances: 1", text)
            comparison = subprocess.run(
                [sys.executable, "-B", str(script), "compare", "--baseline", str(output),
                 "--candidate", str(output), "--bootstrap-draws", "50", "--tests", "paired",
                 "--permutation-draws", "100"], capture_output=True, text=True,
            )
            self.assertEqual(comparison.returncode, 0, comparison.stderr)
            payload = json.loads(comparison.stdout)
            self.assertEqual(payload["tests"]["f1"]["raw_p"], 1)
            self.assertIsNone(payload["tests"]["map"]["raw_p"])
            self.assertEqual(payload["scopes"]["common_success"]["baseline"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
