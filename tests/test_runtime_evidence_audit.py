import json
import unittest

from scripts import audit_runtime_evidence_usage as audit


class RuntimeAuditTests(unittest.TestCase):
    def test_runner_notice_and_nested_output_are_decoded(self):
        inner = {"returncode": 0, "output": "1. pkg/a.py\n"}
        content = json.dumps({"output": json.dumps(inner)}) + "\n<IMPORTANT>notice</IMPORTANT>"
        self.assertEqual(audit.visible_output(content), ("1. pkg/a.py\n", 0))

    def test_elided_output_does_not_restore_hidden_middle(self):
        content = json.dumps({"returncode": 0, "output_head": "pkg/a.py\n", "output_tail": "pkg/z.py\n", "elided_chars": 100})
        text, rc = audit.visible_output(content)
        self.assertEqual(audit.exposed_names(text), {"pkg/a.py", "pkg/z.py"})
        self.assertEqual(rc, 0)

    def test_quoted_semicolons_and_python_loops_preserve_readers(self):
        self.assertEqual(audit.reader_kinds("sed -n '2,11p;12q' dynamic_evidence/file.json"), {"file"})
        self.assertEqual(audit.reader_kinds("python -c \"import json; print(json.load(open('dynamic_evidence/class.json')))\""), {"class"})
        self.assertEqual(audit.reader_kinds("python - <<'PY'\nfrom pathlib import Path\nfor name in ['file','class','function']:\n data=Path('dynamic_evidence') / f'{name}.json'\n print(data.read_text())\nPY"), set(audit.KINDS))

    def test_counts_and_directory_listings_are_not_content(self):
        for command in ("wc -l dynamic_evidence/file.json", "ls dynamic_evidence/*.json", "grep -c qualified_name dynamic_evidence/file.json"):
            self.assertEqual(audit.reader_kinds(command), set(), command)

    def test_shell_loop_constructed_paths(self):
        command = 'for f in file class function; do grep qualified_name dynamic_evidence/$f.json | head -n 10; done'
        self.assertEqual(audit.reader_kinds(command), set(audit.KINDS))

    def test_numbered_names_and_related_metadata_are_separated(self):
        text = '1. pkg/a.py\n2. pkg/b.py::f (vsm=0.1)\n"qualified_name": "pkg/c.py",\n"functions": ["pkg/c.py::hidden"],\n"file": "pkg/metadata.py"\n'
        self.assertEqual(audit.exposed_names(text), {"pkg/a.py", "pkg/b.py::f", "pkg/c.py"})

    def test_scores_before_names_are_recognized(self):
        self.assertEqual(audit.exposed_names('0.04835736: pkg/a.py\n0.08 pkg/a.py::A.f\n'), {"pkg/a.py", "pkg/a.py::A.f"})

    def test_empty_granularity_access_in_a_combined_reader(self):
        call = audit.Call(1, 0, "x", "grep qualified_name dynamic_evidence/file.json; grep qualified_name dynamic_evidence/class.json", '"qualified_name": "pkg/a.py"\n', 0)
        read, _ = audit.classify_access(call, {"file": {"pkg/a.py": 1}, "class": {}, "function": {}})
        self.assertEqual(read, {"file", "class"})

    def test_combined_granularities_do_not_promote_metadata_to_rank(self):
        call = audit.Call(1, 0, "x", "cat dynamic_evidence/file.json dynamic_evidence/function.json", '[{"qualified_name":"pkg/a.py", "functions":["pkg/a.py::late"]}]\n[{"qualified_name":"pkg/a.py::early"}]', 0)
        read, ranks = audit.classify_access(call, {"file": {"pkg/a.py": 1}, "class": {}, "function": {"pkg/a.py::early": 1, "pkg/a.py::late": 25}})
        self.assertEqual(read, {"file", "function"})
        self.assertEqual(ranks["function"], {1})
        self.assertIn("pkg/a.py::late", audit.visible_entity_names(call.output))

    def test_failed_python_echoing_a_name_is_not_evidence(self):
        call = audit.Call(1, 0, "x", "python -c \"print(open('dynamic_evidence/file.json').read())\"", '  File "<string>", line 1\n    print(open("dynamic_evidence/file.json").read())\nSyntaxError: invalid syntax\n', 1)
        read, ranks = audit.classify_access(call, {"file": {"pkg/a.py": 1}, "class": {}, "function": {}})
        self.assertEqual(read, set())

    def test_partial_success_before_error_is_retained(self):
        call = audit.Call(1, 0, "x", "cat dynamic_evidence/file.json", '{"qualified_name": "pkg/a.py"}\ncat: missing: No such file\n', 1)
        read, _ = audit.classify_access(call, {"file": {"pkg/a.py": 11}, "class": {}, "function": {}})
        self.assertEqual(read, {"file"})

    def test_empty_evidence_is_not_unknown_rank(self):
        rows = [{"accessed": ["file"], "rank_status": "empty_evidence", "beyond_10": None}, {"accessed": ["file"], "rank_status": "observed", "beyond_10": False}]
        self.assertEqual(audit.aggregate_access(rows)["beyond_10"]["denominator"], 1)
        self.assertEqual(audit.aggregate_access(rows)["rank_status"]["empty_evidence"], 1)

    def test_wrapper_assignment_is_a_baseline_candidate(self):
        command = 'python -c "def traced(x):\n print(type(x).__name__)\n return original(x)\nprinter._print = traced\nprinter.doprint(A)"'
        flags = audit.baseline_flags(command)
        self.assertIn("attribute_assignment", flags)
        self.assertIn("function_definition", flags)
        self.assertEqual(audit.baseline_flags("grep -n sys.settrace repo/pkg/a.py"), set())


if __name__ == "__main__":
    unittest.main()
