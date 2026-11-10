# REAL-FL (DaLIC)

Experiment code and benchmark data for locating feature implementations in Python repositories. REAL-FL was developed under the working name DaLIC; scripts, configuration files, and result directories retain the `dalic` identifiers.

Given a feature request and a repository containing its implementation, the method traces associated test scenarios, ranks the executed code entities by textual relevance, and supplies this evidence to an LLM agent. The agent returns ranked locations at file, class, and function granularities.

The task is retrospective feature location. Repository workbenches contain the implementation patch and test patch applied to the base commit. The agent inspects this code without editing it; it does not implement the requested feature.

## Repository layout

| Path | Purpose |
| --- | --- |
| `dataset/` | SWE-bench records, classification annotations, ground truth, and instance manifests |
| `scripts/obtain_dataset.py`, `scripts/classify_instance.py` | Dataset acquisition and initial feature request classification |
| `scripts/extract_ground_truth_from_patch.py` | Recover added and edited code locations from implementation patches |
| `scripts/trace_test_cases.py` | Recover test scenarios and collect execution traces with VarTracer |
| `scripts/prepare_agent_workbench*.py` | Restore patched repositories and prepare candidate evidence |
| `scripts/run_*agent*.py` | Execute agent experiments in Docker |
| `scripts/prompts/` | Agent instructions and task configurations |
| `scripts/trad_methods/` | Static and trace-based information retrieval baselines |
| `scripts/evaluate.py` | Evaluate individual runs, compare conditions, and reproduce paper statistics |
| `docker/` | Agent image definition and model connection template |
| `tools/vartracer/` | VarTracer wheel used inside tracing containers |
| `tests/` | Tests for evaluation, retry merging, and evidence auditing |

Generated workbenches, traces, and agent results are ignored by Git. A fresh checkout must regenerate them or obtain the corresponding experiment artifacts separately. Local `analysis/`, `paper_writing/`, and `replication_package/` directories are also ignored by Git.

## Benchmark and experiment scope

The authoritative benchmark manifest is [`dataset/trace_completed_instance_validated.toml`](dataset/trace_completed_instance_validated.toml): **226 instances from 11 Python projects**. Instance metadata, implementation and test patches, and recovered ground truth are stored in [`dataset/swebench_feature_with_truth.jsonl`](dataset/swebench_feature_with_truth.jsonl).

Use the validated manifest explicitly for preparation, execution, retrieval, and evaluation. The older `trace_completed_instance.toml` contains 228 instances and is not the final benchmark. Other dataset files contain intermediate records and should not be treated as the evaluation cohort.

When rebuilding the benchmark, exclude the five false positives identified during manual feature request validation immediately after human validation, before ground-truth recovery and all subsequent stages. Existing traces or results do not make these instances eligible. The final manifest determines membership in the reported benchmark.

The authoritative DaLIC prompt is [`scripts/prompts/dalic_v1.yaml`](scripts/prompts/dalic_v1.yaml). `run_dalic_agent.py` selects it by default, and `run_mini_instance.py` also selects it for `--agent_version dalic`. The commands below specify it explicitly.

## Environment

Run commands from the repository root. You need Linux, Python 3.11 or newer, Git, Docker with access to its daemon, and network access for upstream repositories, container images, and the configured model endpoint. Tracing downloads SWE-bench instance images and can require substantial disk space.

Use the repository's dedicated virtual environment. If it does not exist, create it first:

```bash
python3.11 -m venv python_venv/dalic-nexp
```

Activate it and install the dependencies needed for workbench preparation, tracing, retrieval, and evaluation:

```bash
source python_venv/dalic-nexp/bin/activate
python -m pip install numpy scikit-learn tqdm unidiff bm25s==0.3.9
```

For dataset acquisition and classifier execution, install the broader dependencies in `requirement.txt`. That file also expects a local SWE-bench checkout; clone it only if `tools/SWE-bench` is absent:

```bash
git clone https://github.com/SWE-bench/SWE-bench.git tools/SWE-bench
python -m pip install -r requirement.txt
```

The agent runs inside Docker. Its Dockerfile downloads mini-swe-agent at commit `2caffc565474b8856a323ff163ffb7ab98d1ef02`; a host installation of mini-swe-agent is not required.

## Prepare repositories and runtime evidence

First restore the repositories for the validated benchmark:

```bash
python scripts/prepare_agent_workbench.py \
  --selected_instances dataset/trace_completed_instance_validated.toml
```

This creates `dataset/agent_workbench/baseline/<instance_id>/` with `problem_statement.txt` and `repo/<repo_name>/`. Both patches are applied, and Git metadata and patch files are removed from the agent checkout. Existing instance directories are skipped.

The tracing script accepts a JSONL input rather than an instance manifest. Create a filtered input so it schedules only benchmark instances:

```bash
python - <<'PY'
import json
import tomllib
from pathlib import Path

manifest = Path("dataset/trace_completed_instance_validated.toml")
ids = set(tomllib.loads(manifest.read_text())["instance_ids"])
source = Path("dataset/swebench_feature_with_truth.jsonl")
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
rows = [row for row in rows if row["instance_id"] in ids]
assert len(rows) == len(ids) == 226
assert {row["instance_id"] for row in rows} == ids
output = Path("dataset/agent_workbench/benchmark_226.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("".join(json.dumps(row) + "\n" for row in rows))
PY

python scripts/trace_test_cases.py \
  --input dataset/agent_workbench/benchmark_226.jsonl \
  --resume

python scripts/prepare_agent_workbench_dalic_vsm.py \
  --trace-completed-instances dataset/trace_completed_instance_validated.toml
```

Traces are written under `dataset/vartracer_results/`. The VSM workbench is written under `dataset/agent_workbench/dalic_vsm/<instance_id>/`, including:

```text
problem_statement.txt
repo/<repo_name>/
dynamic_evidence/
  file.json
  class.json
  function.json
```

Evidence records include qualified entity names, VSM scores, and the test cases that executed them. Evidence is a candidate set, not ground truth. The v1 prompt directs the agent to inspect ranked evidence, check source code, expand its search when needed, and exclude test locations from its final predictions.

## Run agent experiments

Build the image using the tag expected by the runners, then create a local connection file:

```bash
docker build -f docker/mini-swe-agent.dockerfile -t mini-swe-agent:2caffc56 .
cp docker/mini-swe-agent.env.template docker/mini-swe-agent.local.env
```

Edit `docker/mini-swe-agent.local.env` with your endpoint details:

```dotenv
MSWEA_MODEL_NAME=openai/<served-model-name>
OPENAI_API_KEY=<YOUR_API_KEY>
OPENAI_BASE_URL=<YOUR_API_BASE_URL>
MSWEA_CONFIGURED=true
MSWEA_COST_TRACKING=ignore_errors
```

`MSWEA_MODEL_NAME` selects the model used for inference. The runner's `--model_name` argument labels output directories and containers; it does not select the inference model. Local `mini-swe-agent-*.env` files are ignored by Git.

Run DaLIC with the validated cohort:

```bash
python scripts/run_dalic_agent.py \
  --selected_instances dataset/trace_completed_instance_validated.toml \
  --env_file docker/mini-swe-agent.local.env \
  --config_yaml dalic_v1.yaml \
  --workbench_path dataset/agent_workbench/dalic_vsm \
  --results_base dataset/agent_results/dalic_vsm \
  --model_name my-model --resume
```

The same runner supports the following conditions by changing these arguments. Run conditions sequentially when reusing the same model label, because this runner uses the same container naming scheme for them.

| Condition | `--config_yaml` | `--workbench_path` | `--results_base` |
| --- | --- | --- | --- |
| Baseline | `baseline.yaml` | `dataset/agent_workbench/baseline` | `dataset/agent_results/baseline` |
| DaLIC | `dalic_v1.yaml` | `dataset/agent_workbench/dalic_vsm` | `dataset/agent_results/dalic_vsm` |
| Without runtime evidence | `dalic_no_runtime.yaml` | `dataset/agent_workbench/baseline` | `dataset/agent_results/dalic_no_runtime` |

`run_mini_instance.py` applies a limit of 50 agent steps. Other model parameters depend on the selected YAML configuration and endpoint. The default v1 configuration does not itself declare a 32K context limit; the context wrapper is enabled only under the conditions checked by `run_mini_instance.py`.

Always use a new output label for an independent run. Without `--resume`, the runner deletes an existing result directory for the same label. With `--resume`, it preserves results and skips processed instances, including failures recorded in `summary.txt`; this flag does not automatically retry those failures. `--test` randomly samples at most 10 available instances and is not a fixed benchmark subset.

Each instance writes `results.txt`, `last_mini_run.traj.json`, and `docker_run.log` under `<results_base>/<model_name>/<instance_id>/`. The run root also contains `dalic.log` and `summary.txt`.

Despite its extension, `results.txt` contains JSON. Lists are ranked, contain no duplicates, and use repository-relative paths:

```json
{
  "results": {
    "files": ["pkg/module.py"],
    "classes": ["pkg/module.py::Example"],
    "functions": ["pkg/module.py::Example.method"]
  }
}
```

## Evaluate results

Score one run against the validated benchmark and restored repositories:

```bash
python scripts/evaluate.py run \
  --selected-instances dataset/trace_completed_instance_validated.toml \
  --workbench-path dataset/agent_workbench/dalic_vsm \
  --result-folder dataset/agent_results/dalic_vsm/my-model
```

For a custom run, this writes `evaluation.json` and `evaluation.txt` into the result directory. Compare two evaluated runs with:

```bash
python scripts/evaluate.py compare \
  --baseline dataset/agent_results/baseline/my-model/evaluation.json \
  --candidate dataset/agent_results/dalic_vsm/my-model/evaluation.json \
  --output-dir dataset/evaluation/my-model
```

Comparisons require identical instance cohorts. Reported differences are candidate minus baseline. Evaluation includes precision, recall, F1, and MAP. Invalid or missing predictions score zero in All Instances; Common Success uses the intersection of valid outputs. Ranking metrics exclude instances with empty ground truth at the corresponding granularity. Avg3 MAP requires nonempty ground truth at all three granularities.

`python scripts/evaluate.py paper` recomputes paper statistics from the archived runs named in `PAPER_RUNS` and cached IR rankings. It requires those artifacts and restored baseline repositories; a fresh checkout or a single custom run is insufficient. Outputs are written under `dataset/evaluation/paper/`. Use `python scripts/evaluate.py --help` and the subcommands' `--help` for additional options.

## Information retrieval baselines

Run static VSM, BM25, and LSI over the restored repositories:

```bash
for method in vsm bm25 lsi; do
  python "scripts/trad_methods/$method.py" \
    --truth dataset/swebench_feature_with_truth.jsonl \
    --workbench dataset/agent_workbench/baseline \
    --instances dataset/trace_completed_instance_validated.toml \
    --output-dir dataset/trad_methods_results
done
```

Run VSM over the runtime candidate set:

```bash
python scripts/trad_methods/trace_ir.py --ranker vsm \
  --truth dataset/swebench_feature_with_truth.jsonl \
  --workbench dataset/agent_workbench/dalic_vsm \
  --instances dataset/trace_completed_instance_validated.toml \
  --output-dir dataset/trad_methods_results
```

These commands write Markdown summaries under `dataset/trad_methods_results/`; they do not generate the cached ranking archives required by `evaluate.py paper`. The repository also includes BM25 evidence preparation in `scripts/prepare_agent_workbench_dalic.py` and Trace BM25 evaluation in `scripts/trad_methods/trace_bm25.py`.

## Tests

With the virtual environment active, run the repository tests:

```bash
python -m unittest discover -s tests
```

These tests check local evaluation and result-processing behavior. Running the Docker experiments additionally requires the repositories, traces, and model connection described above.
