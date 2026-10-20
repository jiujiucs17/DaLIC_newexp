#!/usr/bin/env python3
"""Audit current main DALIC-VSM/Baseline runs without executing their commands.

Run with the experiment venv. Writes only to --output-dir, never to raw runs.
Baseline candidate detection is a review aid; judgments are kept separately.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import analyze_trajectory_audit as legacy
from scripts import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("file", "class", "function")
CONFIGS = (
    ("Qwen3.5-4B", "Qwen35-4B-a(main)", "Qwen3.5-4B-a(main)"),
    ("Qwen3.5-9B", "Qwen35-9B", "Qwen35-9B"),
    ("Qwen3.5-27B", "Qwen35-27B", "Qwen3.5-27B"),
    ("GPT-5.4", "gpt54-0305-openai-a(main)", "gpt54-0305-openai-a(main)"),
    ("DeepSeek-v4-Flash", "DeepSeek-v4-flash-0423", "DeepSeek-v4-flash-0423"),
    ("DeepSeek-v4-Pro", "DeepSeek-v4-pro-0423", "DeepSeek-v4-pro-0423"),
    ("DeepSeek-v4-Pro-32K", "DeepSeek-v4-pro-0423-32K", "DeepSeek-v4-pro-0423-32K"),
)


def json_prefix(text):
    """Decode a JSON prefix even when the runner appends an IMPORTANT notice."""
    try:
        return json.JSONDecoder().raw_decode(text.lstrip())[0]
    except (ValueError, TypeError):
        return None


def visible_output(content):
    """Use message.content only. Hidden raw_output must never restore elisions."""
    rc = None
    value = content
    for _ in range(8):
        obj = json_prefix(value) if isinstance(value, str) else value
        if not isinstance(obj, dict) or not any(
            k in obj for k in ("output", "output_head", "output_tail")
        ):
            return str(value or ""), rc
        if rc is None:
            rc = obj.get("returncode")
        if "output_head" in obj or "output_tail" in obj:
            # A separator prevents a name being synthesized across an elision.
            return "\n<OUTPUT_ELIDED>\n".join(
                str(obj[k]) for k in ("output_head", "output_tail") if obj.get(k)
            ), rc
        value = obj.get("output", "")
    raise ValueError("Unexpected tool-output nesting depth")


@dataclass
class Call:
    command_index: int
    message_index: int
    tool_call_id: str
    command: str
    output: str | None
    returncode: int | None


def extract_calls(trajectory):
    outputs = {}
    for message in trajectory.get("messages", []):
        if message.get("role") == "tool" and message.get("tool_call_id"):
            key = message["tool_call_id"]
            if key in outputs:
                raise ValueError(f"Duplicate tool response ID: {key}")
            outputs[key] = visible_output(message.get("content"))
    calls = []
    for index, message in enumerate(trajectory.get("messages", [])):
        for tool in message.get("tool_calls") or []:
            function = tool.get("function") or {}
            if function.get("name", "").lower() not in {"bash", "shell"}:
                continue
            command = legacy.parse_command(function.get("arguments"))
            if command is None:
                raise ValueError("Unparseable shell arguments")
            output, rc = outputs.get(tool.get("id"), (None, None))
            calls.append(Call(len(calls) + 1, index, tool.get("id"), command, output, rc))
    return calls


def shell_parts(command):
    """Tokenize shell syntax while retaining quoted programs and heredoc bodies."""
    lines = command.splitlines(keepends=True)
    shell, bodies = [], []
    index = 0
    while index < len(lines):
        line = lines[index]
        shell.append(line)
        index += 1
        for match in re.finditer(r"<<-?\s*['\"]?([A-Za-z_]\w*)['\"]?", line):
            body = []
            while index < len(lines) and lines[index].strip() != match[1]:
                body.append(lines[index])
                index += 1
            index += index < len(lines)
            bodies.append("".join(body))
    lex = shlex.shlex("".join(shell), posix=True, punctuation_chars=";&|()<>\n")
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    lex.commenters = "#"
    try:
        tokens = list(lex)
    except ValueError:
        return [], bodies, True
    parts, part = [], []
    for token in tokens:
        if token and all(c in ";&|()\n" for c in token):
            if part:
                parts.append(part)
                part = []
        else:
            part.append(token)
    if part:
        parts.append(part)
    return parts, bodies, False


def mentioned_kinds(text, evidence_context=False):
    result = set()
    if "dynamic_evidence" not in text and not evidence_context:
        return result
    for kind in KINDS:
        if re.search(rf"(?<!\w){kind}\.json\b", text):
            result.add(kind)
    if re.search(r"dynamic_evidence/(?:\*|\*\.json|\{file,class,function\})", text):
        result.update(KINDS)
    # Both shell variables and Python/Path f-strings may construct the path.
    if ".json" in text and re.search(r"for\s+\w+\s+in", text):
        loop_values = re.findall(r"for\s+\w+\s+in\s+(\[[^]]*\]|[^;\n]+)", text)
        for values in loop_values:
            result.update(k for k in KINDS if re.search(rf"(?<!\w){k}(?!\w)", values))
    if re.search(r"dynamic_evidence/\$\w+\.json", text):
        result.update(k for k in KINDS if re.search(rf"(?<!\w){k}(?!\w)", text))
    return result


def reader_kinds(command):
    parts, bodies, error = shell_parts(command)
    if error:
        return set()
    whole = "\n".join(" ".join(p) for p in parts) + "\n" + "\n".join(bodies)
    context = bool(re.search(r"\bcd\s+(?:/workspace/instance/)?dynamic_evidence\b", whole))
    candidates = set()
    for part in parts:
        argv = list(part)
        while argv and (argv[0] in {"do", "then", "env", "command"} or re.match(r"^\w+=", argv[0])):
            argv.pop(0)
        if not argv:
            continue
        program = Path(argv[0]).name
        text = " ".join(argv)
        kinds = mentioned_kinds(text, context)
        if re.search(r"dynamic_evidence/\$\w+\.json", text):
            kinds.update(mentioned_kinds(whole, context))
        if re.fullmatch(r"python[\d.]*|pypy[\d.]*", program):
            # -c programs and heredocs are inspected, never executed.
            code = argv[argv.index("-c") + 1] if "-c" in argv and argv.index("-c") + 1 < len(argv) else "\n".join(bodies)
            if re.search(r"\bopen\s*\(|\.read_text\s*\(|json\.load", code):
                candidates.update(mentioned_kinds(whole, context))
        elif program in {"cat", "head", "tail", "sed", "grep", "rg", "jq", "awk", "perl", "nl", "od", "xxd", "less", "more", "strings"}:
            if program in {"grep", "rg"} and any(
                a in {"--count", "--files-with-matches", "--files-without-match", "--quiet"}
                or re.fullmatch(r"-[A-Za-z]*[clLq][A-Za-z]*", a) for a in argv[1:]
            ):
                continue
            candidates.update(kinds)
    # Recursive searches can expose filenames even without an explicit glob.
    if not candidates and re.search(r"\b(?:grep|rg)\b[^\n]*dynamic_evidence/", whole):
        if re.search(r"(?:grep\s+-[A-Za-z]*[rR]|rg\s)", whole):
            candidates.update(KINDS)
    return candidates


QN = re.compile(r"[\"']qualified_name[\"']\s*:\s*([\"'])(.*?)\1|\bqualified_name:\s*(\S+)")
ENTITY = re.compile(r"[A-Za-z0-9_./+-]+\.py(?:\:\:[A-Za-z0-9_.$<>+-]+)?")


def exposed_names(output):
    """Recognize candidate fields/projections, excluding related-entity metadata."""
    names = set()
    for line in output.splitlines():
        if line.startswith(("Traceback", "  File ", "<OUTPUT_ELIDED>")):
            continue
        qns = list(QN.finditer(line))
        if qns:
            names.update(m[2] or m[3] for m in qns)
            continue
        text = re.sub(r"^\s*(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[.):]?\s+)+", "", line).strip()
        # A flat Python/JSON list of candidate names is an explicit projection.
        if text.startswith(("[", "(")):
            try:
                obj = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                obj = None
            if isinstance(obj, (list, tuple)):
                if all(isinstance(x, str) for x in obj):
                    names.update(obj)
                elif obj and isinstance(obj[0], str):
                    names.add(obj[0])
            continue
        # Prefix match accepts numbered/tabular output and score suffixes.
        # JSON strings inside classes/functions/test_cases arrays are excluded.
        match = ENTITY.match(text)
        if match and (match.end() == len(text) or text[match.end()] in " \t|,(:"):
            names.add(match[0])
    return names


def visible_entity_names(output):
    """Paper's visible-entity measure includes names in related-entity metadata.

    Keep this separate from explicitly viewing a ranked record. Neither measure
    uses hidden raw output or assumes unseen entries were returned.
    """
    return (exposed_names(output) | legacy.output_string_tokens(output)
            | {m[0] for m in ENTITY.finditer(output)})


def classify_access(call, evidence):
    kinds = reader_kinds(call.command)
    ranks = {k: set() for k in KINDS}
    if not kinds or call.output is None or call.returncode in {126, 127}:
        return set(), ranks
    output = call.output
    names = exposed_names(output)
    for kind in kinds:
        ranks[kind].update(evidence[kind][name] for name in names if name in evidence[kind])
    supported = {k for k in kinds if ranks[k]}
    # Metadata content is still access, but it never supplies ranked candidates.
    metadata = re.search(r"[\"'](?:vsm_score|bm25_score|test_cases|test_case_count|total_test_case_count)[\"']\s*:", output)
    no_content_error = re.search(r"Traceback|SyntaxError|Syntax error|cannot open|can't open|No such file|Permission denied|command not found|sed:.*(?:error|unknown)|cut:.*delimiter", output, re.I)
    if metadata and not no_content_error:
        supported.update(kinds)
    if not no_content_error:
        if re.search(r"(?:^|\n)\s*\[\]\s*(?:\n|$)", output):
            supported.update(k for k in kinds if not evidence[k])
        if call.returncode == 0:
            # Successful content-reader on an empty file, e.g. grep -> no matches.
            supported.update(k for k in kinds if not evidence[k])
        # od/xxd can show the literal bytes of an empty JSON list.
        if re.search(r"5b\s+5d", output, re.I):
            supported.update(k for k in kinds if not evidence[k])
        # Test-case/related-entity projections are content even without a QN.
        if re.search(r"[A-Za-z0-9_./-]+\.py(?:::|['\"\s])", output) and not supported:
            supported.update(kinds)
    return supported, ranks


def baseline_flags(command):
    parts, bodies, error = shell_parts(command)
    flags = set()
    if error:
        return {"shell_parse_error"}
    payloads = list(bodies)
    for part in parts:
        if not part:
            continue
        program = Path(part[0]).name
        if (program in {"bash", "sh"} and len(part) > 1) or program.endswith(".sh"):
            flags.add("shell_script")
        if re.fullmatch(r"python[\d.]*|pypy[\d.]*", program):
            if "-c" in part and part.index("-c") + 1 < len(part):
                payloads.append(part[part.index("-c") + 1])
            elif not any(x in part for x in ("-m", "<<", "<<-", "-")) and len(part) > 1 and not part[1].startswith("-"):
                if Path(part[1]).name not in {"setup.py", "runtests.py", "manage.py"}:
                    flags.add("script_execution")
            if "-m" in part and part.index("-m") + 1 < len(part) and part[part.index("-m") + 1] in {"trace", "coverage", "cProfile", "profile", "pdb"}:
                flags.add("instrumentation_cli")
            if "-" in part and not bodies:
                flags.add("python_stdin")
        if re.search(r"\bcoverage\s+(?:run|report)|--cov\b|\b(?:cProfile|viztracer|pyinstrument)\b", " ".join(part)):
            flags.add("instrumentation_cli")
    for code in payloads:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            flags.add("python_syntax_error")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                flags.add("function_definition")
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in {"exec", "eval", "__import__", "setattr"} or any(x in name for x in ("settrace", "setprofile", "exec_module", "runpy.", "patch", "wrap")):
                    flags.add("dynamic_call")
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Attribute) for t in targets):
                    flags.add("attribute_assignment")
    return flags


def fraction(numerator, denominator):
    return {"count": numerator, "denominator": denominator,
            "percentage": 100 * numerator / denominator if denominator else None}


def aggregate_access(rows):
    applicable = [r for r in rows if r["rank_status"] == "observed"]
    result = {"trajectories": len(rows), "rank_status": dict(collections.Counter(r["rank_status"] for r in rows))}
    for kind in ("any", *KINDS):
        result[kind] = fraction(sum(bool(r["accessed"]) if kind == "any" else kind in r["accessed"] for r in rows), len(rows))
    result["beyond_10"] = fraction(sum(r["beyond_10"] is True for r in applicable), len(applicable))
    if rows and "ranked_entry_beyond_10" in rows[0]:
        explicit = [r for r in rows if any(r["ranked_entry_ranks"].values())]
        result["ranked_entry_beyond_10"] = fraction(sum(r["ranked_entry_beyond_10"] is True for r in explicit), len(explicit))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-judgments", type=Path,
                        default=Path(__file__).with_name("audit_runtime_evidence_usage_baseline_judgments.json"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = legacy.load_ids()
    truth = ev.load_ground_truth_map(ROOT / "dataset/swebench_feature_with_truth.jsonl", set(ids))
    wb = ROOT / "dataset/agent_workbench/dalic_vsm"
    access, calls_out, candidates, predictions, configs, snapshots = [], [], [], [], [], []
    def snapshot(path):
        snapshots.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    snapshot(ROOT / "dataset/trace_completed_instance_validated.toml")
    snapshot(ROOT / "dataset/swebench_feature_with_truth.jsonl")
    evidence = {}
    for i in ids:
        evidence[i] = {}
        for k in KINDS:
            path = wb / i / "dynamic_evidence" / f"{k}.json"
            snapshot(path)
            entries = json.loads(path.read_text())
            names = [x["qualified_name"] for x in entries]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate evidence entity: {path}")
            evidence[i][k] = {name: rank for rank, name in enumerate(names, 1)}
    judgments = json.loads(args.baseline_judgments.read_text()) if args.baseline_judgments else {}
    baseline_positive = []
    for model, vsm, base in CONFIGS:
        result_dir = ROOT / "dataset/agent_results/dalic_vsm" / vsm
        evaluations, _ = ev.load_instance_evaluations(result_dir, truth, wb)
        valid = {e.instance_id: e for e in evaluations if not e.is_failure}
        for e in evaluations:
            result_path = result_dir / e.instance_id / "results.txt"
            if result_path.exists():
                snapshot(result_path)
                if e.result_sha256 and snapshots[-1]["sha256"] != e.result_sha256:
                    raise ValueError(f"Result changed during evaluation: {result_path}")
        configs.append({"model": model, "dalic_vsm_run": vsm, "baseline_run": base, "valid_results": len(valid)})
        for instance in ids:
            path = result_dir / instance / "last_mini_run.traj.json"
            raw = path.read_bytes()
            snapshots.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()})
            kinds, ranks = set(), {k: set() for k in KINDS}
            visible_ranks = {k: set() for k in KINDS}
            for call in extract_calls(json.loads(raw)):
                if "dynamic_evidence" not in call.command:
                    continue
                read, one = classify_access(call, evidence[instance])
                kinds.update(read)
                names = visible_entity_names(call.output or "")
                visible = {k: sorted({evidence[instance][k][name] for name in names if name in evidence[instance][k]}) if k in read else [] for k in KINDS}
                for k in KINDS:
                    if k in read:
                        ranks[k].update(one[k])
                        visible_ranks[k].update(visible[k])
                calls_out.append({"model": model, "instance": instance, "trajectory": str(path.relative_to(ROOT)), **call.__dict__, "accessed": sorted(read), "ranked_entry_ranks": {k: sorted(one[k]) for k in KINDS}, "ranks": visible})
            empty = all(not evidence[instance][k] for k in KINDS)
            observed = any(visible_ranks.values())
            access.append({"model": model, "instance": instance, "accessed": sorted(kinds), "ranks": {k: sorted(visible_ranks[k]) for k in KINDS}, "ranked_entry_ranks": {k: sorted(ranks[k]) for k in KINDS}, "rank_status": "empty_evidence" if empty else "observed" if observed else "unresolved", "beyond_10": any(r > 10 for rs in visible_ranks.values() for r in rs) if observed else None, "ranked_entry_beyond_10": any(r > 10 for rs in ranks.values() for r in rs) if any(ranks.values()) else None})
            if instance in valid:
                for k, g in zip(KINDS, ev.GRANULARITIES):
                    p, t, r = valid[instance].predictions[g], truth[instance]["total"][g], set(evidence[instance][k])
                    outside, inside = p - r, p & r
                    correct_outside, correct_inside = outside & t, inside & t
                    missing = t - r
                    assert correct_outside == p & missing
                    assert len(p) == len(inside) + len(outside)
                    assert len(p & t) == len(correct_inside) + len(correct_outside)
                    predictions.append({"model": model, "instance": instance, "kind": k, "predictions": len(p), "outside": len(outside), "inside": len(inside), "correct": len(p & t), "correct_outside": len(correct_outside), "correct_inside": len(correct_inside), "missing_gt": len(missing), "recovered_missing_gt": len(p & missing)})
            bp = ROOT / "dataset/agent_results/baseline" / base / instance / "last_mini_run.traj.json"
            raw = bp.read_bytes()
            snapshots.append({"path": str(bp.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()})
            for call in extract_calls(json.loads(raw)):
                flags = baseline_flags(call.command)
                if flags:
                    key = f"{base}/{instance}/{call.tool_call_id}"
                    digest = hashlib.sha256(call.command.encode()).hexdigest()
                    output_digest = hashlib.sha256((call.output or "").encode()).hexdigest()
                    judgment = judgments.get(key)
                    if judgment and (judgment["command_sha256"] != digest or judgment["output_sha256"] != output_digest):
                        raise ValueError(f"Stale Baseline judgment: {key}")
                    candidates.append({"key": key, "command_sha256": digest, "output_sha256": output_digest, "flags": sorted(flags), "model": model, "instance": instance, "trajectory": str(bp.relative_to(ROOT)), **call.__dict__, "judgment": judgment})
                    if judgment and judgment["runtime_observation"]:
                        baseline_positive.append(candidates[-1])
        print(f"Audited {model}", flush=True)
    pooled = {}
    for model, _, _ in CONFIGS:
        rr = [r for r in predictions if r["model"] == model]
        counts = {key: sum(r[key] for r in rr) for key in ("predictions", "outside", "inside", "correct", "correct_outside", "correct_inside", "missing_gt", "recovered_missing_gt")}
        counts.update({"outside_share": fraction(counts["outside"], counts["predictions"]), "correct_outside_share": fraction(counts["correct_outside"], counts["correct"]), "missing_gt_recovery": fraction(counts["recovered_missing_gt"], counts["missing_gt"]), "precision_inside": fraction(counts["correct_inside"], counts["inside"]), "precision_outside": fraction(counts["correct_outside"], counts["outside"])})
        pooled[model] = counts
    coverage = {k: fraction(sum(len(set(evidence[i][k]) & truth[i]["total"][g]) for i in ids), sum(len(truth[i]["total"][g]) for i in ids)) for k, g in zip(KINDS, ev.GRANULARITIES)}
    pending = [r["key"] for r in candidates if r["judgment"] is None]
    for row in snapshots:
        if hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Trajectory changed during audit: {row['path']}")
    stale = set(judgments) - {r['key'] for r in candidates}
    if stale:
        raise ValueError(f"Judgments no longer correspond to candidates: {sorted(stale)}")
    summary = {"verification_status": "PENDING_BASELINE_REVIEW" if pending else "VERIFIED", "cohort": "dataset/trace_completed_instance_validated.toml", "instances": len(ids), "configs": configs, "access": aggregate_access(access), "access_by_model": {m: aggregate_access([r for r in access if r["model"] == m]) for m, _, _ in CONFIGS}, "coverage": coverage, "predictions": pooled, "baseline": {"candidate_commands": len(candidates), "pending_judgments": pending, "runtime_observation_trajectories": None if pending else fraction(len({r['trajectory'] for r in baseline_positive}), len(ids) * len(CONFIGS)), "positive_commands": [{k: r[k] for k in ("key", "model", "instance", "trajectory", "command_index", "message_index", "judgment")} for r in baseline_positive]}, "input_snapshots": snapshots}
    for filename, data in (("summary", summary), ("access", access), ("evidence_calls", calls_out), ("baseline_candidates", candidates), ("predictions", predictions)):
        (args.output_dir / f"{filename}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"access": summary["access"], "baseline_pending": len(pending)}, ensure_ascii=False))
    if pending:
        print("Baseline observations remain provisional until every candidate is adjudicated.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
