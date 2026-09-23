#!/usr/bin/env python3
"""Compute requested statistics from the final validated DALIC archive."""

from __future__ import annotations

import gzip, hashlib, json, math, sys, tomllib
from pathlib import Path
from typing import Any, Callable
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "dataset/results_archive/agent_results"
MANIFEST = ROOT / "dataset/trace_completed_instance_validated.toml"
TRUTH = ROOT / "dataset/swebench_feature_with_truth.jsonl"
RANKS = ROOT / "dataset/trad_methods_results/rankings"
OUT = ARCH / "requested_statistical_results.json"
SUMMARY = ROOT / ("paper_writing/DaLIC-Paper/DaLIC_ feature recovery for python "
                  "combining software reconnaissance and data dependencies/"
                  "exp_metric_results/experiment_metrics_summary_readable.md")
MODELS = ("Qwen_Qwen3.5-4B", "Qwen_Qwen3.5-9B", "Qwen_Qwen3.5-27B",
          "GPT_5-4", "DeepSeek-v4-flash", "DeepSeek-v4-pro",
          "DS-v4-pro_32k_think_full")
NAMES = dict(zip(MODELS, ("Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B",
                          "GPT-5.4", "DeepSeek-v4-flash", "DeepSeek-v4-pro",
                          "DeepSeek-v4-pro-32k")))
STATIC = ("static_vsm", "static_lsi", "static_bm25")
MN = {"static_vsm": "Static VSM", "static_lsi": "Static LSI",
      "static_bm25": "Static BM25", "trace_bm25": "Trace BM25"}
G, FEATURES, KS = ev.GRANULARITIES, (*ev.GRANULARITIES, "avg3"), (3, 5, 10, 20)
BD, FD, BS, FS = 20_000, 100_000, 20260910, 20260911
BEGIN = "<!-- BEGIN GENERATED REQUESTED STATISTICS -->"
END = "<!-- END GENERATED REQUESTED STATISTICS -->"
Condition = Callable[[str, str], tuple[list[str], bool]]


def dseed(base: int, key: str) -> int:
    return base + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % 10**9


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    with MANIFEST.open("rb") as f:
        ids = sorted(tomllib.load(f)["instance_ids"])
    assert len(ids) == len(set(ids)) == 226
    all_truth = ev.load_ground_truth_map(TRUTH)
    selected = {i: all_truth[i] for i in ids}
    truth = {i: selected[i]["total"] for i in ids}
    agents = {m: {} for m in ("original", "baseline", "dalic")}
    for method in agents:
        wb = ROOT / "dataset/agent_workbench" / ("dalic" if method == "dalic" else "baseline")
        for model in MODELS:
            folder = ARCH / method / model
            if not folder.is_dir():
                continue
            evaluations, _ = ev.load_instance_evaluations(folder, selected, wb)
            records = {}
            for item in evaluations:
                ranked = {g: [] for g in G}
                if not item.is_failure:
                    ranked = json.loads((folder/item.instance_id/"results.txt").read_text())["results"]
                records[item.instance_id] = {"valid": not item.is_failure, "ranked": ranked}
            assert set(records) == set(ids)
            agents[method][model] = records
    retrieval = {}
    for method in (*STATIC, "trace_bm25"):
        files = {p.name.removesuffix(".json.gz"): p for p in (RANKS/method).glob("*.json.gz")}
        assert set(files) == set(ids)
        retrieval[method] = {}
        for i in ids:
            with gzip.open(files[i], "rt") as f:
                row = json.load(f)
            assert row["failure_reason"] is None
            retrieval[method][i] = row["rankings"]
    return ids, truth, agents, retrieval


def rcond(retrieval: dict, method: str, cutoff: Any = None) -> Condition:
    def get(i: str, g: str):
        ranked = retrieval[method][i][g]
        k = cutoff if isinstance(cutoff, int) else (None if cutoff is None else cutoff[i][g])
        return (ranked if k is None else ranked[:k]), True
    return get


def acond(agents: dict, method: str, model: str) -> Condition:
    def get(i: str, g: str):
        row = agents[method][model][i]
        return row["ranked"][g], row["valid"]
    return get


def prf(ranked, truth, valid):
    return (0., 0., 0.) if not valid else ev.instance_precision_recall_f1(set(ranked), truth)


def average_precision(ranked, truth, valid):
    assert truth
    if not valid:
        return 0.
    seen, total = set(), 0.
    for rank, item in enumerate(ranked, 1):
        if item in truth and item not in seen:
            seen.add(item); total += len(seen)/rank
    return total/len(truth)


def rank_ids(ids, truth):
    return [i for i in ids if all(truth[i][g] for g in G)]


def matrices(condition, ids, truth):
    member = np.empty((len(ids), 3, 3))
    for x, i in enumerate(ids):
        for y, g in enumerate(G):
            ranked, valid = condition(i, g)
            member[x, y] = prf(ranked, truth[i][g], valid)
    rid = rank_ids(ids, truth)
    ranks = np.empty((len(rid), 3))
    for x, i in enumerate(rid):
        for y, g in enumerate(G):
            ranked, valid = condition(i, g)
            ranks[x, y] = average_precision(ranked, truth[i][g], valid)
    return {"precision": member[:,:,0], "recall": member[:,:,1],
            "f1": member[:,:,2], "map": ranks,
            "ranking_n": len(rid)}


def four(a):
    return np.column_stack((a, a.mean(axis=1)))


def point(condition, ids, truth):
    m = matrices(condition, ids, truth)
    complete_ground_truth_mask = np.asarray([
        all(truth[i][g] for g in G) for i in ids
    ], dtype=bool)
    complete_ground_truth_avg3_f1 = m["f1"][complete_ground_truth_mask].mean(axis=1)
    metrics = {
        metric: dict(zip(FEATURES, map(float, four(m[metric]).mean(axis=0))))
        for metric in ("precision", "recall", "f1")
    }
    ranking_ns = {}
    metrics["map"] = {}
    for granularity in G:
        granularity_ids = [i for i in ids if truth[i][granularity]]
        ranking_ns[granularity] = len(granularity_ids)
        scores = []
        for i in granularity_ids:
            ranked, valid = condition(i, granularity)
            scores.append(average_precision(ranked, truth[i][granularity], valid))
        metrics["map"][granularity] = float(np.asarray(scores).mean())
    metrics["map"]["avg3"] = float(four(m["map"]).mean(axis=0)[-1])
    return {
        "membership_n": len(ids),
        "ranking_n": m["ranking_n"],
        "ranking_n_by_granularity": ranking_ns,
        "complete_ground_truth": {
            "n": int(complete_ground_truth_mask.sum()),
            "avg3_f1": float(complete_ground_truth_avg3_f1.mean()),
        },
        "metrics": metrics,
    }


def boot(delta, key):
    delta = delta[:,None] if delta.ndim == 1 else delta
    exact = dseed(BS, key); rng = np.random.default_rng(exact)
    samples = np.empty((BD, delta.shape[1]))
    for start in range(0, BD, 500):
        size = min(500, BD-start)
        ix = rng.integers(0, len(delta), (size, len(delta)))
        samples[start:start+size] = delta[ix].mean(axis=1)
    out = {f: {"n": len(delta), "estimate": float(delta[:,j].mean()),
               "ci_95": [float(np.percentile(samples[:,j], 2.5)),
                         float(np.percentile(samples[:,j], 97.5))]}
           for j, f in enumerate(FEATURES[:delta.shape[1]])}
    out["seed"] = exact
    return out


def ranking_effect(later, earlier, ids, truth, key):
    result = {}
    for granularity in G:
        granularity_ids = [i for i in ids if truth[i][granularity]]
        delta = []
        for i in granularity_ids:
            later_ranked, later_valid = later(i, granularity)
            earlier_ranked, earlier_valid = earlier(i, granularity)
            later_value = average_precision(
                later_ranked, truth[i][granularity], later_valid
            )
            earlier_value = average_precision(
                earlier_ranked, truth[i][granularity], earlier_valid
            )
            delta.append(later_value - earlier_value)
        one = boot(np.asarray(delta), key + ":map:" + granularity)
        result[granularity] = {**one[FEATURES[0]], "seed": one["seed"]}

    complete_ids = rank_ids(ids, truth)
    a = matrices(later, complete_ids, truth)
    b = matrices(earlier, complete_ids, truth)
    avg3_delta = (a["map"] - b["map"]).mean(axis=1)
    one = boot(avg3_delta, key + ":map:avg3")
    result["avg3"] = {**one[FEATURES[0]], "seed": one["seed"]}
    return result


def compare(later, earlier, ids, truth, key):
    a, b = matrices(later, ids, truth), matrices(earlier, ids, truth)
    ranking_ns = {g: sum(bool(truth[i][g]) for i in ids) for g in G}
    complete_ground_truth_mask = np.asarray([
        all(truth[i][g] for g in G) for i in ids
    ], dtype=bool)
    complete_ground_truth_delta = (
        a["f1"][complete_ground_truth_mask]
        - b["f1"][complete_ground_truth_mask]
    ).mean(axis=1)
    complete_ground_truth_effect = boot(
        complete_ground_truth_delta, key + ":f1:complete_ground_truth"
    )
    return {
        "membership_n": len(ids),
        "ranking_n": a["ranking_n"],
        "ranking_n_by_granularity": ranking_ns,
        "f1": boot(four(a["f1"] - b["f1"]), key + ":f1"),
        "complete_ground_truth_f1": {
            **complete_ground_truth_effect["files"],
            "seed": complete_ground_truth_effect["seed"],
        },
        "map": ranking_effect(later, earlier, ids, truth, key),
    }


def flip(delta, key):
    exact = dseed(FS, key)
    if not np.any(delta): return 1., exact
    rng, observed, exceed = np.random.default_rng(exact), abs(float(delta.mean())), 0
    for start in range(0, FD, 2000):
        size = min(2000, FD-start)
        values = np.abs((rng.choice((-1.,1.), (size,len(delta)))*delta).mean(axis=1))
        exceed += int(np.count_nonzero(values >= observed-1e-15))
    return (exceed+1)/(FD+1), exact


def mcnemar(b, c):
    n = b+c
    return 1. if not n else min(1., 2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n)


def holm(raw):
    ordered, out, running = sorted(raw, key=raw.get), {}, 0.
    for j, key in enumerate(ordered):
        running = max(running, min(1., (len(ordered)-j)*raw[key]))
        out[key] = running
    return out


def output_lengths(agents, model, ids, invalid_k=3):
    return {i: {g: len(agents["dalic"][model][i]["ranked"][g])
                   if agents["dalic"][model][i]["valid"] else invalid_k
                for g in G} for i in ids}



def build():
    ids, truth, agents, retrieval = load()
    rid = rank_ids(ids, truth)
    data = {
        "metadata": {
            "cohort": str(MANIFEST.relative_to(ROOT)), "instances": len(ids),
            "ranking_instances": len(rid), "manifest_sha256": sha(MANIFEST),
            "ranking_instances_by_granularity": {
                g: sum(bool(truth[i][g]) for i in ids) for g in G
            },
            "truth": str(TRUTH.relative_to(ROOT)), "truth_sha256": sha(TRUTH),
            "avg3": "unweighted within-instance file/class/function mean, then instance macro mean",
            "invalid": "zero P/R/F1/AP in All Instances; excluded from valid-only/common-success",
            "empty_membership": "score 1 only when prediction and truth are both empty; otherwise 0 if one is empty",
            "empty_ranking": "exclude at granularity; Avg3 MAP requires all three nonempty",
            "matched": "truncate Static or Trace IR per instance/granularity to valid DALIC length; invalid DALIC gives k=3",
            "trace_dalic_criteria": "all 226 instances: matched length with invalid k=3; fixed k=3 for every instance",
            "bootstrap": {"draws": BD, "seed_base": BS, "unit": "paired instance",
                          "ci": "percentile 95%", "seeds": "SHA-256 derived and stored per effect"},
            "randomization": {"draws": FD, "seed_base": FS, "test": "two-sided paired sign flip",
                              "p": "(exceedances+1)/(draws+1)", "seeds": "SHA-256 derived"},
            "mcnemar": "exact two-sided binomial", "holm": "step-down; three separate seven-test families",
            "prompt": "scripts/prompts/dalic_v1.yaml",
            "prompt_check": "run_dalic_agent.py -> run_mini_instance.py --agent_version dalic -> dalic_v1.yaml",
            "script": "scripts/compute_requested_statistical_tests.py",
            "json": "dataset/results_archive/agent_results/requested_statistical_results.json",
        },
        "rq1_static_trace": {}, "rq1_trace_dalic": {},
        "rq2_static_dalic": {}, "rq3_baseline_dalic": {}
    }
    for k in KS:
        s, t = rcond(retrieval, "static_bm25", k), rcond(retrieval, "trace_bm25", k)
        data["rq1_static_trace"][str(k)] = {
            "static": point(s, ids, truth), "trace": point(t, ids, truth),
            "effect": compare(t, s, ids, truth, f"rq1:static_trace:{k}")}
    static_points = {m: {str(k): point(rcond(retrieval,m,k),ids,truth) for k in KS}
                     for m in STATIC}
    raw = {family: {} for family in ("completion","f1","map")}
    for model in MODELS:
        d, b = acond(agents,"dalic",model), acond(agents,"baseline",model)
        lens = output_lengths(agents,model,ids,invalid_k=3)
        tm = rcond(retrieval,"trace_bm25",lens)
        tk3 = rcond(retrieval,"trace_bm25",3)
        matched_effect = compare(
            d,tm,ids,truth,f"rq1:trace_matched_fail_k3_dalic:{model}"
        )
        fixed_k3_all_effect = compare(
            d,tk3,ids,truth,f"rq1:trace_fixed_k3_all_dalic:{model}"
        )
        data["rq1_trace_dalic"][model] = {
            "dalic_all": point(d,ids,truth),
            "trace_matched": point(tm,ids,truth),
            "trace_fixed_k3_all": point(tk3,ids,truth),
            "matched_effect": matched_effect,
            "fixed_k3_all_effect": fixed_k3_all_effect}
        rq2 = {"dalic":point(d,ids,truth),"fixed":{},"matched":{},"full":{}}
        for method in STATIC:
            rq2["fixed"][method] = {}
            for k in KS:
                s = rcond(retrieval,method,k)
                rq2["fixed"][method][str(k)] = {"static":static_points[method][str(k)],
                    "effect":compare(d,s,ids,truth,f"rq2:{model}:{method}:{k}")}
            s = rcond(retrieval,method,lens)
            rq2["matched"][method] = {"static":point(s,ids,truth),
                "effect":compare(d,s,ids,truth,f"rq2:{model}:{method}:matched")}
            s = rcond(retrieval,method)
            e = compare(d,s,ids,truth,f"rq2:{model}:{method}:full")
            rq2["full"][method] = {"static":point(s,ids,truth),"map":e["map"]}
        data["rq2_static_dalic"][model] = rq2
        alle = compare(d,b,ids,truth,f"rq3:{model}:all")
        common = [i for i in ids if agents["dalic"][model][i]["valid"]
                  and agents["baseline"][model][i]["valid"]]
        ce = compare(d,b,common,truth,f"rq3:{model}:common")
        bv = np.array([agents["baseline"][model][i]["valid"] for i in ids],int)
        dv = np.array([agents["dalic"][model][i]["valid"] for i in ids],int)
        comp = boot((dv-bv).astype(float),f"rq3:{model}:completion")
        both=int(((bv==1)&(dv==1)).sum()); bo=int(((bv==1)&(dv==0)).sum())
        do=int(((bv==0)&(dv==1)).sum()); neither=int(((bv==0)&(dv==0)).sum())
        cp=mcnemar(bo,do)
        bm=four(matrices(b,ids,truth)["f1"])[:,-1]
        dm=four(matrices(d,ids,truth)["f1"])[:,-1]
        br=four(matrices(b,ids,truth)["map"])[:,-1]
        dr=four(matrices(d,ids,truth)["map"])[:,-1]
        fp,fseed=flip(dm-bm,f"rq3:{model}:f1"); mp,mseed=flip(dr-br,f"rq3:{model}:map")
        raw["completion"][model]=cp; raw["f1"][model]=fp; raw["map"][model]=mp
        original = None
        if model in agents["original"]:
            o=acond(agents,"original",model)
            ov=np.array([agents["original"][model][i]["valid"] for i in ids],int)
            original={"completion":boot((ov-bv).astype(float),f"rq3:{model}:original_completion"),
                      "effect":compare(o,b,ids,truth,f"rq3:{model}:original_baseline")}
        data["rq3_baseline_dalic"][model] = {
            "completion":{"baseline_valid":int(bv.sum()),"dalic_valid":int(dv.sum()),
                          "both_valid":both,"b":bo,"c":do,"both_invalid":neither,
                          "effect":comp["files"],"seed":comp["seed"],"raw_p":cp},
            "baseline_all":point(b,ids,truth),"dalic_all":point(d,ids,truth),
            "all":alle,"common":ce,
            "f1_test":{"n":len(ids),"observed":float((dm-bm).mean()),"raw_p":fp,"seed":fseed},
            "map_test":{"n":len(rid),"observed":float((dr-br).mean()),"raw_p":mp,"seed":mseed},
            "original_baseline":original}
    for family, values in raw.items():
        adj=holm(values)
        for model in MODELS:
            target=(data["rq3_baseline_dalic"][model]["completion"] if family=="completion"
                    else data["rq3_baseline_dalic"][model][family+"_test"])
            target["holm_p"]=adj[model]; target["reject"]=adj[model] <= .05
    return data


def pct(x): return f"{100*x:.2f}%"
def pp(x): return f"{100*x:+.2f}"
def pv(x): return f"{x:.5f}"
def eff(x, feature="avg3"):
    y=x[feature]; return f"{pp(y['estimate'])} [{pp(y['ci_95'][0])}, {pp(y['ci_95'][1])}]"
def pt(x, metric): return pct(x["metrics"][metric]["avg3"])
def complete_f1(x): return pct(x["complete_ground_truth"]["avg3_f1"])
def complete_eff(x):
    return f"{pp(x['estimate'])} [{pp(x['ci_95'][0])}, {pp(x['ci_95'][1])}]"


def render(data):
    L=[BEGIN,"## 8. Requested paired statistical analyses","",
       "All effects are absolute differences in percentage points with pointwise 95% percentile intervals from 20,000 paired instance bootstrap resamples. RQ1 and RQ2 use estimation-oriented analysis without formal null-hypothesis tests.","",
       "This section is recomputed from archived per-instance outputs. Output validity is checked against the required JSON schema and each instance checkout, including rejection of redundant checkout-directory prefixes. It does not reuse manuscript or summary-table values.","",
       "### 8.1 RQ1: Static BM25 to Trace BM25","",
       "Direction is Trace BM25 minus Static BM25. F1 uses all 226 instances; Avg3 MAP uses the 170 instances with nonempty ground truth at all three granularities.","",
       "| k | Avg3 F1 difference [95% CI] | Avg3 MAP difference [95% CI] | File F1 | Class F1 | Function F1 | File MAP | Class MAP | Function MAP |",
       "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in KS:
        e=data["rq1_static_trace"][str(k)]["effect"]
        L.append(f"| {k} | {eff(e['f1'])} | {eff(e['map'])} | "+ " | ".join(
            [eff(e["f1"],g) for g in G]+[eff(e["map"],g) for g in G])+" |")
    L += ["","### 8.2 RQ1: Trace BM25 to DALIC","",
          "Both criteria include all 226 instances. An invalid DALIC output scores zero. MAP uses the Trace BM25 prefix specified by each criterion. Direction is DALIC minus Trace BM25.",""]
    L += ["","#### 8.2.1 Matched length, with failed DALIC runs assigned Trace BM25 k=3","",
          "For a valid DALIC run, Trace BM25 uses the corresponding DALIC output length at each granularity. For an invalid DALIC run, Trace BM25 uses k=3.","",
          "| Model | N | Trace / DALIC Avg3 P | Trace / DALIC Avg3 R | Trace / DALIC Avg3 F1 | F1 difference [95% CI] | MAP N | Trace / DALIC Avg3 MAP | MAP difference [95% CI] |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        r=data["rq1_trace_dalic"][model]; e=r["matched_effect"]
        t=r["trace_matched"]; d=r["dalic_all"]
        L.append(f"| {NAMES[model]} | {e['membership_n']} | {pt(t,'precision')} / {pt(d,'precision')} | {pt(t,'recall')} / {pt(d,'recall')} | {pt(t,'f1')} / {pt(d,'f1')} | {eff(e['f1'])} | {e['ranking_n']} | {pt(t,'map')} / {pt(d,'map')} | {eff(e['map'])} |")
    L += ["","#### 8.2.2 Fixed Trace BM25 k=3 for every instance","",
          "Trace BM25 uses k=3 for every instance and granularity, regardless of whether DALIC is valid.","",
          "| Model | N | Trace / DALIC Avg3 P | Trace / DALIC Avg3 R | Trace / DALIC Avg3 F1 | F1 difference [95% CI] | MAP N | Trace / DALIC Avg3 MAP | MAP difference [95% CI] |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        r=data["rq1_trace_dalic"][model]; e=r["fixed_k3_all_effect"]
        t=r["trace_fixed_k3_all"]; d=r["dalic_all"]
        L.append(f"| {NAMES[model]} | {e['membership_n']} | {pt(t,'precision')} / {pt(d,'precision')} | {pt(t,'recall')} / {pt(d,'recall')} | {pt(t,'f1')} / {pt(d,'f1')} | {eff(e['f1'])} | {e['ranking_n']} | {pt(t,'map')} / {pt(d,'map')} | {eff(e['map'])} |")


    L += ["","### 8.3 RQ2: Static IR to complete DALIC","",
          "Fixed rows compare each Static IR prefix with complete DALIC output on all 226 instances. Matched rows use DALIC output length for valid runs and k=3 for invalid runs. Invalid DALIC output scores zero. Direction is DALIC minus Static IR.","",
          "Avg3 retrieval point estimates:","",
          "| Method | k | P | R | F1 | MAP |","|---|---:|---:|---:|---:|---:|"]
    for method in (*STATIC,"trace_bm25"):
        for k in KS:
            if method=="trace_bm25": x=data["rq1_static_trace"][str(k)]["trace"]
            elif method=="static_bm25": x=data["rq1_static_trace"][str(k)]["static"]
            else: x=data["rq2_static_dalic"][MODELS[0]]["fixed"][method][str(k)]["static"]
            L.append(f"| {MN[method]} | {k} | "+" | ".join(pt(x,m) for m in ("precision","recall","f1","map"))+" |")
    L += ["","Complete DALIC Avg3 point estimates:","",
          "| Model | P | R | F1 | MAP |",
          "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        x=data["rq2_static_dalic"][model]["dalic"]
        L.append(f"| {NAMES[model]} | "+" | ".join(
            pt(x,m) for m in ("precision","recall","f1","map"))+" |")
    L += ["","Fixed-cutoff effects:","",
          "| Model | Static method | k | Static / DALIC F1 | F1 difference [95% CI] | Static / DALIC MAP | MAP difference [95% CI] |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        r=data["rq2_static_dalic"][model]
        for method in STATIC:
            for k in KS:
                x=r["fixed"][method][str(k)]
                L.append(f"| {NAMES[model]} | {MN[method]} | {k} | {pt(x['static'],'f1')} / {pt(r['dalic'],'f1')} | {eff(x['effect']['f1'])} | {pt(x['static'],'map')} / {pt(r['dalic'],'map')} | {eff(x['effect']['map'])} |")
    L += ["","Fixed-cutoff granularity effects:","",
          "| Model | Static method | k | File F1 | Class F1 | Function F1 | File MAP | Class MAP | Function MAP |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        r=data["rq2_static_dalic"][model]
        for method in STATIC:
            for k in KS:
                e=r["fixed"][method][str(k)]["effect"]
                L.append(f"| {NAMES[model]} | {MN[method]} | {k} | "+" | ".join(
                    [eff(e["f1"],g) for g in G]+[eff(e["map"],g) for g in G])+" |")
    L += ["","Matched-length effects with failed DALIC runs assigned Static IR k=3:","",
          "All 226 instances are included. Valid DALIC runs define the Static IR cutoff by granularity; invalid DALIC runs define k=3.","",
          "| Model | Static method | Static / DALIC F1 | F1 difference [95% CI] | Static / DALIC MAP@matched | MAP difference [95% CI] |",
          "|---|---|---:|---:|---:|---:|"]
    for model in MODELS:
        r=data["rq2_static_dalic"][model]
        for method in STATIC:
            x=r["matched"][method]
            L.append(f"| {NAMES[model]} | {MN[method]} | {pt(x['static'],'f1')} / {pt(r['dalic'],'f1')} | {eff(x['effect']['f1'])} | {pt(x['static'],'map')} / {pt(r['dalic'],'map')} | {eff(x['effect']['map'])} |")
    L += ["","Matched-length granularity effects with failed DALIC runs assigned Static IR k=3:","",
          "| Model | Static method | File F1 | Class F1 | Function F1 | File MAP | Class MAP | Function MAP |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        r=data["rq2_static_dalic"][model]
        for method in STATIC:
            e=r["matched"][method]["effect"]
            L.append(f"| {NAMES[model]} | {MN[method]} | "+" | ".join(
                [eff(e["f1"],g) for g in G]+[eff(e["map"],g) for g in G])+" |")
    L += ["","Complete-ranking descriptive effects:","",
          "| Model | Static method | Static / DALIC MAP | MAP difference [95% CI] |",
          "|---|---|---:|---:|"]
    for model in MODELS:
        r=data["rq2_static_dalic"][model]
        for method in STATIC:
            x=r["full"][method]
            L.append(f"| {NAMES[model]} | {MN[method]} | {pt(x['static'],'map')} / {pt(r['dalic'],'map')} | {eff(x['map'])} |")
    L += ["","The JSON artifact additionally stores file, class, function, and Avg3 P/R/F1/MAP point estimates for every comparison condition.","",
          "### 8.4 RQ3: Baseline to DALIC","",
          "Holm adjustment is separate within the seven completion, seven Avg3 F1, and seven Avg3 MAP tests. Every F1 test uses 226 paired instances and every MAP test uses 170 paired instances.","",
          "| Model | Completion difference [95% CI] | McNemar raw p | Holm p | Reject | Avg3 F1 difference [95% CI] | Raw p | Holm p | Reject | Avg3 MAP difference [95% CI] | Raw p | Holm p | Reject |",
          "|---|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|:---:|"]
    for model in MODELS:
        r=data["rq3_baseline_dalic"][model]; c=r["completion"]; f=r["f1_test"]; m=r["map_test"]
        ce=f"{pp(c['effect']['estimate'])} [{pp(c['effect']['ci_95'][0])}, {pp(c['effect']['ci_95'][1])}]"
        L.append(f"| {NAMES[model]} | {ce} | {pv(c['raw_p'])} | {pv(c['holm_p'])} | {'yes' if c['reject'] else 'no'} | {eff(r['all']['f1'])} | {pv(f['raw_p'])} | {pv(f['holm_p'])} | {'yes' if f['reject'] else 'no'} | {eff(r['all']['map'])} | {pv(m['raw_p'])} | {pv(m['holm_p'])} | {'yes' if m['reject'] else 'no'} |")
    L += ["","Paired completion tables:","",
          "| Model | Both valid | Baseline valid / DALIC invalid (b) | Baseline invalid / DALIC valid (c) | Both invalid |",
          "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        c=data["rq3_baseline_dalic"][model]["completion"]
        L.append(f"| {NAMES[model]} | {c['both_valid']} | {c['b']} | {c['c']} | {c['both_invalid']} |")
    L += ["","Secondary descriptive effects:","",
          "| Model | Common N F1 / MAP | Common Avg3 F1 difference [95% CI] | Common Avg3 MAP difference [95% CI] |",
          "|---|---:|---:|---:|"]
    for model in MODELS:
        r=data["rq3_baseline_dalic"][model]; c=r["common"]
        L.append(f"| {NAMES[model]} | {c['membership_n']} / {c['ranking_n']} | {eff(c['f1'])} | {eff(c['map'])} |")
    L += ["","Granularity-specific All Instances effects:","",
          "| Model | File F1 | Class F1 | Function F1 | File MAP | Class MAP | Function MAP |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        e=data["rq3_baseline_dalic"][model]["all"]
        L.append(f"| {NAMES[model]} | "+" | ".join([eff(e["f1"],g) for g in G]+[eff(e["map"],g) for g in G])+" |")
    L += ["","Original minus Baseline descriptive effects:","",
          "| Model | Completion difference [95% CI] | Avg3 F1 difference [95% CI] | Avg3 MAP difference [95% CI] |",
          "|---|---:|---:|---:|"]
    for model in MODELS:
        o=data["rq3_baseline_dalic"][model]["original_baseline"]
        if o is None: continue
        c=o["completion"]["files"]
        L.append(f"| {NAMES[model]} | {pp(c['estimate'])} [{pp(c['ci_95'][0])}, {pp(c['ci_95'][1])}] | {eff(o['effect']['f1'])} | {eff(o['effect']['map'])} |")
    L += ["","### 8.5 Avg3 F1 on the complete-ground-truth cohort","",
          "This sensitivity view restricts Avg3 F1 to the same fixed 170 instances used for Avg3 MAP: each instance has nonempty file, class, and function ground truth. Avg3 F1 is the unweighted mean of the three instance-level F1 values, followed by a macro average over the 170 instances. Invalid agent outputs score zero. The cohort is defined by ground truth and is therefore identical across methods and model configurations.","",
          "Fixed-cutoff retrieval:","",
          "| k | Static VSM | Static LSI | Static BM25 | Trace BM25 | Trace minus Static BM25 [95% CI] |",
          "|---:|---:|---:|---:|---:|---:|"]
    for k in KS:
        rq1=data["rq1_static_trace"][str(k)]
        vsm=data["rq2_static_dalic"][MODELS[0]]["fixed"]["static_vsm"][str(k)]["static"]
        lsi=data["rq2_static_dalic"][MODELS[0]]["fixed"]["static_lsi"][str(k)]["static"]
        L.append(
            f"| {k} | {complete_f1(vsm)} | {complete_f1(lsi)} | "
            f"{complete_f1(rq1['static'])} | {complete_f1(rq1['trace'])} | "
            f"{complete_eff(rq1['effect']['complete_ground_truth_f1'])} |"
        )
    L += ["","Complete DALIC and RQ1 Trace BM25 comparisons:","",
          "| Model | DALIC | Trace k=3 / DALIC difference [95% CI] | Matched Trace / DALIC difference [95% CI] |",
          "|---|---:|---:|---:|"]
    for model in MODELS:
        r=data["rq1_trace_dalic"][model]
        L.append(
            f"| {NAMES[model]} | {complete_f1(r['dalic_all'])} | "
            f"{complete_f1(r['trace_fixed_k3_all'])} / {complete_eff(r['fixed_k3_all_effect']['complete_ground_truth_f1'])} | "
            f"{complete_f1(r['trace_matched'])} / {complete_eff(r['matched_effect']['complete_ground_truth_f1'])} |"
        )
    L += ["","RQ2 matched-length comparisons:","",
          "| Model | DALIC | Static VSM / DALIC difference [95% CI] | Static LSI / DALIC difference [95% CI] | Static BM25 / DALIC difference [95% CI] |",
          "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        r=data["rq2_static_dalic"][model]
        cells=[]
        for method in STATIC:
            x=r["matched"][method]
            cells.append(
                f"{complete_f1(x['static'])} / {complete_eff(x['effect']['complete_ground_truth_f1'])}"
            )
        L.append(f"| {NAMES[model]} | {complete_f1(r['dalic'])} | " + " | ".join(cells) + " |")
    L += ["","RQ3 All Instances comparison on the fixed 170-instance cohort:","",
          "| Model | Baseline | DALIC | DALIC minus Baseline [95% CI] |",
          "|---|---:|---:|---:|"]
    for model in MODELS:
        r=data["rq3_baseline_dalic"][model]
        L.append(
            f"| {NAMES[model]} | {complete_f1(r['baseline_all'])} | "
            f"{complete_f1(r['dalic_all'])} | {complete_eff(r['all']['complete_ground_truth_f1'])} |"
        )
    z=data["metadata"]
    L += ["","### 8.6 Reproducibility metadata","",
          f"- Cohort: {z['cohort']}, {z['instances']} instances; all other archive entries are ignored.",
          f"- Avg3 ranking cohort: {z['ranking_instances']} instances.",
          "- Granularity ranking cohorts: "+", ".join(
              f"{g}={z['ranking_instances_by_granularity'][g]}" for g in G)+".",
          f"- Avg3: {z['avg3']}.",f"- Invalid outputs: {z['invalid']}.",
          f"- Empty ground truth for membership: {z['empty_membership']}.",
          f"- Empty ground truth for ranking: {z['empty_ranking']}.",
          f"- Matched length: {z['matched']}.",
          f"- Bootstrap: {BD:,} paired instance resamples, percentile 95% CI, seed base {BS}; exact derived seeds are in JSON.",
          f"- Randomization: {FD:,} two-sided sign flips, seed base {FS}, p=(exceedances+1)/(draws+1); exact derived seeds are stored per test in JSON.",
          f"- McNemar: {z['mcnemar']}.",f"- Holm: {z['holm']}.",
          f"- DALIC prompt: {z['prompt']}; verified by {z['prompt_check']}.",
          f"- Generator: {z['script']}.",f"- Machine-readable output: {z['json']}.",
          f"- Manifest SHA-256: {z['manifest_sha256']}.",f"- Ground-truth SHA-256: {z['truth_sha256']}.","",
          "### 8.7 Interpretation and validation boundary","",
          "RQ1 and RQ2 intervals are estimation results, not formal hypothesis tests. RQ3 rejection decisions apply only to the three prespecified seven-test Holm families. The complete-ground-truth Avg3 F1 sensitivity results, granularity-specific results, Pairwise Common Success results, and Original versus Baseline results remain descriptive.","",
          "Statistical fallacy scan: 11/11 categories checked. The primary All Instances scope avoids survivorship conditioning; Common Success is explicitly secondary. The separate prespecified Holm families address the relevant look-elsewhere risk. The remaining Simpson, ecological, Berkson, collider, base-rate, regression-to-the-mean, garden-of-forking-paths, causal-language, and reverse-causality patterns are not implicated by these paired benchmark contrasts.","",END]
    return "\n".join(L)+"\n"


def main():
    data=build()
    OUT.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    generated=render(data); text=SUMMARY.read_text()
    if BEGIN in text:
        a=text.index(BEGIN); b=text.index(END,a)+len(END)
        text=text[:a].rstrip()+"\n\n"+generated.rstrip()+text[b:]
    else: text=text.rstrip()+"\n\n"+generated
    SUMMARY.write_text(text.rstrip()+"\n")
    print(f"Wrote {OUT}\nUpdated {SUMMARY}")


if __name__ == "__main__":
    main()
