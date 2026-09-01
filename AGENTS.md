# Repository Instructions

## DaLIC experiment source of truth

- Use `scripts/prompts/dalic_v1.yaml` as the authoritative DaLIC agent prompt for experiment analysis, replication checks, and paper revisions. The final reported experiments use the v1 data. But you don't have to mention this in the paper writing. you just assume v2/v3 related files never existed.
- Do not inspect, use, reconstruct, or cite `dalic_v2.yaml` or `dalic_v3.yaml` unless the user explicitly requests them.
- Verify the active prompt path through `scripts/run_dalic_agent.py` and `scripts/run_mini_instance.py` before describing the agent configuration.

## Final benchmark population

- The final benchmark and every final paper analysis use 226 instances, not the 228 instances in the raw trace-complete set.
- Exclude all five feature-classifier false positives identified by the human audit. Three of them are already absent after scenario recovery; the two trace-complete false positives, `sympy__sympy-17176` and `sympy__sympy-17770`, must also be excluded from the final benchmark and all reported statistics.
- Treat 228 only as the count of raw trace-complete execution results. Do not report it as the final benchmark size or analysis population.

## Academic writing

- Avoid the em dash character in papers, documentation, and articles.
- Avoid compound modifiers with multiple hyphenated components. Rewrite them as plain phrases when possible.
- Keep responses and manuscript edits focused on the requested scope.
