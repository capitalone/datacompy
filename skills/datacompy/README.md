# datacompy-skills

Official DataComPy agent skill for comparing two DataFrames or tables with DataComPy's v1 API. Built on the SKILL.md open standard and compatible with Claude Code, OpenAI Codex, GitHub Copilot, and Cursor.

## What this skill does

When installed, this skill activates automatically when an agent encounters a comparison, diff, reconciliation, or validation task on tabular data (including PROC COMPARE-style requests), or any Python dataframe comparison request where no library has been specified. It teaches agents to:

- Use the v1 API only (`PandasCompare`, `PolarsCompare`, `SparkSQLCompare`, `SnowflakeCompare`) and never generate the removed `Compare`, `SparkCompare`, or `datacompy.fugue` interfaces
- Pick the comparison class by the dataframe type already in hand, without converting Spark or Snowpark frames to Pandas
- Choose a unique join key and set numeric tolerances (`abs_tol`, `rel_tol`) deliberately instead of relying on exact-match defaults
- Answer questions with the built-in accessors (`matches()`, `all_mismatch()`, `df1_unq_rows`, `build_report_data()`) instead of parsing the text report or hand-rolling merges
- Avoid the silent-failure traps of the library, such as column-name lowercasing, `rel_tol` measured against df2, duplicate join keys paired by row order, and partially effective column masking

## Installation

Copy the `skills/datacompy/` directory into the skills folder for your agent tool. No build step required. Start a session and the skill loads automatically when a task involves comparing two datasets.

### Claude Code

```bash
# Personal (all projects)
git clone https://github.com/capitalone/datacompy
cp -r skills/datacompy ~/.claude/skills/

# Project-level (checked into git)
cp -r skills/datacompy .claude/skills/
```

After a manual copy the skill command is `/datacompy` (no plugin namespace).

### OpenAI Codex

```bash
# User-level
cp -r skills/datacompy ~/.codex/skills/

# Repo-level
cp -r skills/datacompy .codex/skills/
```

### Cursor

```bash
cp -r skills/datacompy .cursor/skills/
```

### GitHub Copilot (VS Code)

Place the `skills/datacompy/` directory in your VS Code agent skills folder. See the GitHub Copilot Agent Skills documentation for the correct path.

## Compatibility

| Tool | Supported |
|---|---|
| Claude Code | Yes |
| OpenAI Codex | Yes |
| GitHub Copilot | Yes |
| Cursor | Yes |
| Any assistant with system instructions | Yes |

The skill targets Python 3.10+ and DataComPy 1.0 or later. If your environment still runs a 0.x version, the skill detects it, tells you, and guides you through the migration while still writing v1 code. It covers all four DataComPy backends (Pandas, Polars, Spark, Snowflake). It does not apply to SAS `PROC COMPARE` itself, or to the removed v0 `Compare`, `SparkCompare`, and `fugue` APIs.

## Repository structure

```
skills/datacompy/
├── .claude-plugin/
│   └── plugin.json             # Claude Code plugin manifest
├── SKILL.md                    # Core skill: workflow, accessors, rules, gotchas
└── references/
    ├── backends.md             # Constructors, extras, and backend-specific options
    └── migration-v0-to-v1.md   # v0 to v1 API mapping and rename table
```

Reference files are loaded on demand. The agent reads them only when the task requires detail beyond what is in `SKILL.md`.

## Documentation

Full library documentation lives at [https://capitalone.github.io/datacompy/](https://capitalone.github.io/datacompy/). The skill prefers its own reference files for accuracy, and falls back to the online docs for anything not covered.

## License

Apache-2.0
