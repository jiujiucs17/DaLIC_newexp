# DaLIC CLI Functional Model

## Goal

DaLIC CLI should help an agent use large trace artifacts for feature localization without reading whole files into context.

The CLI should be optimized for:

- `mini-swe-agent` style usage through a single `bash` command
- short, deterministic, non-interactive outputs
- stateless calls that work well across fresh subshells
- progressive disclosure: summary first, drill-down second
- direct access to raw trace data when the agent explicitly needs it


## Why This Shape

The current setup suggests a few strong design constraints:

- `mini-swe-agent` effectively sees the world through `bash(command)`.
- Its tool observations are truncated when output gets too long.
- The current trace corpus is dominated by `exec_stack.json`, while `dep_tree.json` is much smaller and more index-like.

So the CLI should not assume an agent can safely read a raw JSON artifact directly. It should instead expose:

1. small summary commands
2. ranking commands
3. targeted slice commands
4. explicit raw-data escape hatches


## Agent-Facing Design Principles

### 1. Prefer one command per question

Each subcommand should answer one concrete question:

- "Which files are most relevant?"
- "Which test cases hit this file?"
- "Show me 20 events around this line."

Avoid commands that require the agent to assemble several unrelated flags before getting a useful answer.

### 2. Default output must stay short

Default behavior should fit comfortably inside one tool observation.

Recommended defaults:

- default `--limit 20`
- truncate long strings
- omit raw source/code bodies unless explicitly requested
- print hints like `use --offset 20 --limit 20 for more`

### 3. Human-readable first, machine-readable on demand

Default format:

- compact tables for ranking/listing commands

Optional formats:

- `--json`
- `--jsonl`
- `--tsv`

This matches agent behavior well: it can skim a table, and switch to JSON only when it needs exact field names.

### 4. Stateless by default

Each command should work without hidden session state.

Good:

- `dalic locate files --instance django__django-15180`
- `dalic trace slice --instance ... --tc ... --file ... --line 127`

Avoid making the agent do:

- `dalic open-session ...`
- `dalic set-instance ...`
- `dalic cd ...`

If caching is useful, it should be explicit and optional.

### 5. Stable identifiers everywhere

Every output should use stable IDs the agent can feed into the next command:

- `instance_id`
- `test_case_id`
- `file_id`
- `symbol_id`
- `event_id`
- `path_id`

This reduces quoting issues and avoids re-parsing long strings.


## Object Model

DaLIC CLI should expose the corpus through the following hierarchy:

- corpus
- project
- instance
- test case
- artifact

Core artifact types:

- `summary`
- `metadata`
- `dep_tree`
- `exec_stack`
- `stdout`
- derived indexes and summaries


## Top-Level Command Model

Recommended top-level shape:

```text
dalic stats ...
dalic instance ...
dalic testcase ...
dalic locate ...
dalic dep ...
dalic trace ...
dalic raw ...
dalic index ...
dalic doctor ...
```

This keeps the surface area understandable while separating "high-level localization" from "low-level artifact access".


## Command Families

## 1. `dalic stats`

Purpose: quick discovery and health checks.

Suggested subcommands:

- `dalic stats corpus`
- `dalic stats projects`
- `dalic stats instances --project django__django`
- `dalic stats sizes --instance <id>`
- `dalic stats failures`

Typical questions answered:

- How much data exists?
- Which projects have the largest traces?
- Which instances are incomplete or failed?
- Which artifact dominates size?

Example:

```bash
dalic stats sizes --instance django__django-15180
```

Default output:

- one compact table
- total bytes by artifact
- test case counts
- success/failure counts


## 2. `dalic instance`

Purpose: instance-level discovery.

Suggested subcommands:

- `dalic instance list`
- `dalic instance show --instance <id>`
- `dalic instance files --instance <id>`
- `dalic instance tests --instance <id>`
- `dalic instance patch --instance <id>`

Typical questions answered:

- What artifacts exist for this instance?
- Which test cases succeeded?
- What files are touched by the patch?
- Which files appear most in the traces?

Example:

```bash
dalic instance show --instance pydata__xarray-5233
```

Recommended default fields:

- status
- finalized
- project
- test case counts
- successful / failed / timed out
- total trace size
- top touched files


## 3. `dalic testcase`

Purpose: test-case-level discovery.

Suggested subcommands:

- `dalic testcase list --instance <id>`
- `dalic testcase show --instance <id> --tc <tc_id>`
- `dalic testcase artifacts --instance <id> --tc <tc_id>`
- `dalic testcase stdout --instance <id> --tc <tc_id>`

Typical questions answered:

- Which test cases are available?
- Which ones are huge?
- Which ones failed?
- Which test case is best to inspect first?

Example:

```bash
dalic testcase list --instance django__django-12343 --sort trace_bytes
```

Recommended default fields:

- `test_case_id`
- test name
- exit status
- timeout flag
- `exec_stack_bytes`
- `dep_tree_bytes`
- `trace_total_bytes`


## 4. `dalic locate`

Purpose: feature localization. This should be the main agent-facing layer.

Suggested subcommands:

- `dalic locate files --instance <id>`
- `dalic locate symbols --instance <id>`
- `dalic locate functions --instance <id>`
- `dalic locate paths --instance <id>`
- `dalic locate explain --instance <id> --file <path>`

This family should combine evidence from:

- patch files / changed hunks
- successful test cases
- `dep_tree`
- `exec_stack` summaries

Recommended outputs:

- ranked candidates
- compact evidence columns
- short explanation strings

### `dalic locate files`

Return ranked files for feature localization.

Good default columns:

- `rank`
- `file`
- `tests_hit`
- `hit_lines`
- `dep_in`
- `dep_out`
- `cross_tc_support`
- `score`
- `why`

### `dalic locate symbols`

Return ranked symbols within one instance or one file.

Good default columns:

- `rank`
- `symbol_id`
- `kind`
- `name`
- `scope`
- `file`
- `line`
- `score`
- `why`

### `dalic locate paths`

Return a small number of interpretable dependency chains.

Good default columns:

- `path_id`
- `source`
- `sink`
- `length`
- `support`
- `files_crossed`

### `dalic locate explain`

Give a compact explanation for one candidate.

Example:

```bash
dalic locate explain --instance django__django-15180 --file django/urls/conf.py
```

Default output should be a short report:

- why this file ranked highly
- which test cases hit it
- most important symbols
- top dependency paths
- suggested follow-up commands


## 5. `dalic dep`

Purpose: direct exploration of `dep_tree.json`.

This is the best low-level interface for the agent after `locate`.

Suggested subcommands:

- `dalic dep files --instance <id> --tc <tc_id>`
- `dalic dep symbols --instance <id> --tc <tc_id>`
- `dalic dep edges --instance <id> --tc <tc_id>`
- `dalic dep sinks --instance <id> --tc <tc_id>`
- `dalic dep paths --instance <id> --tc <tc_id> --sink <symbol_id>`
- `dalic dep neighbors --instance <id> --tc <tc_id> --symbol <symbol_id>`

Typical questions answered:

- What symbols exist in this test case?
- Which symbols depend on this one?
- What path leads into this sink?
- Which files are connected by data flow?

Recommended behavior:

- resolve IDs to human-readable labels by default
- keep raw IDs visible
- support `--ids-only` for compact chaining

Example:

```bash
dalic dep paths --instance django__django-15180 --tc tc_72a4c738 --sink s18
```


## 6. `dalic trace`

Purpose: controlled access to execution-trace views derived from `exec_stack.json`.

This family should not dump raw nested JSON by default. It should expose indexed views first.

Suggested subcommands:

- `dalic trace files --instance <id> [--tc <tc_id>]`
- `dalic trace funcs --instance <id> [--file <path>]`
- `dalic trace lines --instance <id> [--file <path>]`
- `dalic trace calls --instance <id> [--func <name>]`
- `dalic trace slice --instance <id> --tc <tc_id> --file <path> --line <n>`
- `dalic trace around-event --instance <id> --tc <tc_id> --event <event_id>`
- `dalic trace grep --instance <id> --pattern \"assertRaisesMessage\"`

### `dalic trace files`

Aggregate dynamic evidence per file.

Recommended columns:

- `file`
- `events`
- `line_hits`
- `call_hits`
- `exception_hits`
- `tests_hit`

### `dalic trace funcs`

Aggregate dynamic evidence per function.

Recommended columns:

- `func`
- `file`
- `calls`
- `line_hits`
- `max_depth`
- `tests_hit`

### `dalic trace lines`

Aggregate dynamic evidence per line.

Recommended columns:

- `file`
- `line`
- `hits`
- `assigned`
- `depends_on`

### `dalic trace slice`

This is the main "show me the local raw behavior" tool.

Example:

```bash
dalic trace slice \
  --instance django__django-15180 \
  --tc tc_72a4c738 \
  --file tests/urlpatterns/tests.py \
  --line 127 \
  --before 8 \
  --after 12
```

Default output:

- numbered event list
- event type
- file / line / func
- short line content
- compact dependency / assignment columns


## 7. `dalic raw`

Purpose: explicit direct access to original artifacts.

This family is important because the agent sometimes does need the raw source data, not just summaries.

But raw access should still be safe and slice-oriented.

Suggested subcommands:

- `dalic raw cat --artifact summary --instance <id>`
- `dalic raw cat --artifact metadata --instance <id> --tc <tc_id>`
- `dalic raw events --instance <id> --tc <tc_id> --offset 0 --limit 20`
- `dalic raw event --instance <id> --tc <tc_id> --event <event_id>`
- `dalic raw subtree --instance <id> --tc <tc_id> --event <event_id>`
- `dalic raw dep-field --instance <id> --tc <tc_id> --field symbols.s18`
- `dalic raw stdout --instance <id> --tc <tc_id> --head 80`

### Raw-access rules

- no full-file dump by default for large artifacts
- full dump requires an explicit `--all` or `--output <file>`
- raw JSON should support `--json`
- nested selection should use a simple field-path syntax instead of full jq-style expressions

Recommended field-path examples:

- `symbols.s18`
- `paths.s21`
- `execution_stack[120]`
- `execution_stack[120].details.daughter_stack`

This keeps direct raw access possible without forcing the agent to shell out into custom Python for common cases.


## 8. `dalic index`

Purpose: build and manage derived indexes that make the rest of the CLI fast and compact.

Suggested subcommands:

- `dalic index build`
- `dalic index status`
- `dalic index rebuild --instance <id>`
- `dalic index verify`

Derived index ideas:

- per-test-case file/function/line coverage summaries
- per-instance aggregated file and symbol rankings
- inverted index from file -> test cases
- inverted index from symbol -> paths / neighbors
- event offsets for fast `trace slice`

This is where large `exec_stack.json` should be turned into something agent-friendly.


## 9. `dalic doctor`

Purpose: operational checks.

Suggested subcommands:

- `dalic doctor corpus`
- `dalic doctor instance --instance <id>`
- `dalic doctor artifacts --instance <id>`

Typical checks:

- missing summary
- missing metadata
- missing `dep_tree`
- missing `exec_stack`
- inconsistent counts
- unreadable JSON


## Global Options

Recommended global flags:

- `--root <path>`
- `--instance <id>`
- `--tc <test_case_id>`
- `--limit <n>`
- `--offset <n>`
- `--sort <field>`
- `--json`
- `--jsonl`
- `--tsv`
- `--no-color`
- `--max-chars <n>`
- `--verbose`

Important defaults:

- `--root` defaults to `dataset/vartracer_results`
- stable sort order when not specified
- `--limit 20` unless the command naturally returns one object


## Output Contract

To work well with `mini-swe-agent`, outputs should obey these rules:

### Table mode

- compact columns
- fixed column names
- no wrapped giant cells
- elide long strings with `...`

### JSON mode

- one stable schema per subcommand
- avoid mixing human-only text with machine fields
- include IDs alongside resolved labels

### Long result handling

When the result is truncated, the CLI should print:

- how many rows were shown
- total available rows if known
- exact follow-up hint

Example:

```text
Showing 20 of 143 rows. Use --offset 20 --limit 20 for the next page.
```


## Recommended V1 Command Set

If you want a narrow and useful first version, I would build these first:

- `dalic stats corpus`
- `dalic instance show`
- `dalic testcase list`
- `dalic locate files`
- `dalic locate symbols`
- `dalic dep symbols`
- `dalic dep paths`
- `dalic trace files`
- `dalic trace slice`
- `dalic raw events`
- `dalic raw dep-field`

This is enough to support a realistic feature-localization workflow without exposing the whole system at once.


## Recommended Agent Workflow

A good `mini-swe-agent` flow should look like this:

1. Start from ranked candidates

```bash
dalic locate files --instance django__django-15180
```

2. Drill into one file

```bash
dalic locate explain --instance django__django-15180 --file django/urls/conf.py
```

3. Inspect supporting dependency paths

```bash
dalic dep paths --instance django__django-15180 --tc tc_72a4c738 --sink s18
```

4. Inspect local execution around a suspicious line

```bash
dalic trace slice --instance django__django-15180 --tc tc_72a4c738 --file django/urls/conf.py --line 57
```

5. If necessary, fetch a raw JSON fragment

```bash
dalic raw events --instance django__django-15180 --tc tc_72a4c738 --offset 120 --limit 20 --json
```

This is exactly the kind of progressive drill-down that fits a bash-only agent with short tool observations.


## What The CLI Should Not Do

- It should not require loading a full `exec_stack.json` into output by default.
- It should not require the agent to use complex quoting or embedded Python for common lookups.
- It should not hide important IDs.
- It should not rely on interactive prompts.
- It should not require persistent shell state.


## Bottom Line

The right DaLIC CLI is not "a thin wrapper around the artifact files".

It is a three-layer interface:

1. localization commands for agent decision-making
2. indexed trace/dep exploration commands for targeted inspection
3. raw slice commands for exact evidence retrieval

That gives `mini-swe-agent` both:

- small, decision-ready answers
- direct access to original evidence when needed

without forcing it to read giant raw trace files into context.
