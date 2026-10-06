# Repository Instructions

## DaLIC experiment source of truth

- Use `scripts/prompts/dalic_v1.yaml` as the authoritative DaLIC agent prompt for experiment analysis, replication checks, and paper revisions. The final reported experiments use the v1 data.
- Do not inspect, use, reconstruct, or cite `dalic_v2.yaml` or `dalic_v3.yaml` unless the user explicitly requests them.
- Verify the active prompt path through `scripts/run_dalic_agent.py` and `scripts/run_mini_instance.py` before describing the agent configuration.

## Benchmark cohort exclusion

- Exclude the five instances judged to be false positives during manual feature request classification immediately after human validation and before ground-truth recovery or any later methodology stage.
- Existing experimental runs or generated artifacts for these five instances do not make them eligible for the benchmark. Do not include them in downstream processing, evaluation, aggregation, statistical analysis, or reported results.

## Academic writing

- Avoid the em dash character in papers, documentation, and articles.
- Avoid compound modifiers with multiple hyphenated components. Rewrite them as plain phrases when possible.
- Keep responses and manuscript edits focused on the requested scope.


## Python venv
如需使用python，请使用“python_venv/dalic-nexp”中专为本实验仓库准备的python venv。如果它缺少依赖，可以为它安装所需依赖。