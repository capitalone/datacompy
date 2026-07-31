# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataComPy is a Python library for comparing two DataFrames/tables across multiple backends: Pandas, Polars, Spark, and Snowflake. It originated as a replacement for SAS's `PROC COMPARE`. v1 is GA; the version lives in `datacompy/__version__` and `pyproject.toml` derives the distribution version from it.

This file is the single AI agent guide for the repository. `.github/copilot-instructions.md` used to duplicate it and was removed; put new guidance here rather than starting a second copy.

## Common Commands

### Setup
```bash
pip install -e ".[dev]"
pre-commit install
```

### Testing
```bash
pytest                                              # all tests
pytest tests/test_pandas.py                         # single backend
pytest tests/test_pandas.py::test_numeric_columns_equal_abs   # single test
pytest -k "tolerance and not spark"                 # by expression
pytest --cov=datacompy --cov-report=term-missing    # with coverage
pytest -c pytest-ansi.ini                           # Spark ANSI mode
```

CI runs the suite twice, once with the default `pytest.ini` and once with `-c pytest-ansi.ini`, which only differs by `spark.sql.ansi.enabled`. A change touching Spark casting or null handling needs both.

**Coverage:** always target the top-level package (`--cov=datacompy`). Passing a dotted submodule such as `--cov=datacompy.cli` triggers a numpy double-import in some environments and fails ~100 otherwise-passing tests with a confusing `_NoValueType` `TypeError`.

**Spark** needs `pyspark` and **Java 17** (newer JDKs fail with `py4j.protocol` errors). If the JDK came from conda (`conda install openjdk=17`, as `[edgetest.envs.core]` does), it is at `$CONDA_PREFIX/lib/jvm` and `JAVA_HOME` must point there. Activating the env normally sets this; a non-interactive shell will not inherit it:
```bash
export JAVA_HOME=$CONDA_PREFIX/lib/jvm
```

**Snowflake** tests need a live session, or `--snowflake-session local` for Snowpark's local testing mode. Local mode is an emulator, not Snowflake: `eqNullSafe` returns `True` for every row and high-precision decimals are truncated on DataFrame creation. Tests that depend on either must request the `requires_live_snowflake_session` fixture (`tests/conftest.py`), which skips them in local mode.

### Linting & Formatting
```bash
ruff check                 # lint
ruff check --fix           # lint with auto-fix
ruff format --check        # format check
ruff format                # apply formatting
mypy .                     # type-check (strict mode)
```

**Use the `ruff` version pinned in `.pre-commit-config.yaml`.** The config uses recent selectors, and an older ruff fails to parse `pyproject.toml` at all rather than degrading gracefully. `pre-commit run --all-files` fetches the right version itself.

**`mypy .` has a large pre-existing error baseline** (~185, mostly in `snowflake.py`, `polars.py`, and `pandas.py`) and is enforced by neither CI nor pre-commit; CI lint runs only `ruff check` and `ruff format --check`. New code is still expected to be clean, so check that your diff introduces no *new* errors rather than that the run is empty, and do not refactor unrelated modules to chase the baseline. Missing-stub errors for `pyspark` and `snowflake.snowpark` mean those extras are not installed in the current environment, not a code defect.

### Documentation
```bash
make sphinx                # build docs (runs in docs/ subdirectory)
```

## Architecture

### Strategy Pattern for Backend Comparisons

The core design uses the **Strategy pattern** with two abstraction layers:

1. **`BaseCompare`** (`datacompy/base.py`), the ABC defining the comparison interface. All backends implement: `_compare`, `_dataframe_merge`, `_intersect_compare`, `report`, `matches`, `subset`, `sample_mismatch`, `all_mismatch`, etc.

2. **Backend implementations**, each in its own module:
   - `datacompy/pandas.py` → `PandasCompare`
   - `datacompy/polars.py` → `PolarsCompare`
   - `datacompy/spark.py` → `SparkSQLCompare`
   - `datacompy/snowflake.py` → `SnowflakeCompare`

Spark and Snowflake are optional imports (try/except in `__init__.py`), so `datacompy.SparkSQLCompare` simply does not exist when the extra is missing. Because that try/except runs at package import time, importing *any* datacompy submodule pulls in pyspark when it is installed; there is no lazy path around it.

Beyond the report, each backend exposes `df1_unq_rows`, `df2_unq_rows`, and `intersect_rows` for programmatic analysis, and `build_report_data()` returns a typed `ReportData` (`datacompy/report.py`) with `render()`, `to_html()`, `save()`, and `to_dict()`. Prefer these over parsing the string report.

### Comparator Subpackage

`datacompy/comparator/` provides type-specific column comparison logic, also using a strategy pattern:

- `base.py` → `BaseComparator` ABC with `compare(col1, col2)` method
- `numeric.py` → Numeric comparators per backend (handles tolerances)
- `string.py` → String comparators per backend
- `array.py` → Array-like comparators per backend

Each type has backend-specific implementations: `Pandas*Comparator`, `Polars*Comparator`, `Spark*Comparator`, `Snowflake*Comparator`.

### Reporting

Reports use Jinja2 templates from `datacompy/templates/report_template.j2`. The `render()` function in `base.py` handles template resolution. Custom templates can be passed via `report(template_path=...)`.

### Tolerance Handling

Tolerances (`abs_tol`, `rel_tol`) can be a single float (applied globally) or a dict mapping column names to per-column values. Validated by `validate_tolerance_parameter()` in `base.py`.

### Command Line Interface

`datacompy/cli/` implements the `datacompy` console script (entry point `datacompy.cli:main`, also reachable as `python -m datacompy`).

- `parser.py` holds the `OPTIONS` tuple, the **single source of truth** for the argument surface. Each `Opt` row records the argparse flags *and* the `*Compare` constructor keyword it maps to, so `build_parser()` and `backends.compare_kwargs()` are both generated from it. **Adding a library kwarg to the CLI is one new row.** Do not hand-write it in two places. `tests/cli/test_parser.py` asserts every `Opt.kwarg` against the real constructor signature via `inspect.signature`, so drift fails the build.
- Options are registered with `default=argparse.SUPPRESS` and defaults live on `Opt.default`. That is what makes `Opt.was_given()` meaningful. `validate_arguments()` must run **before** `fill_defaults()`.
- `backends.py` holds the `CLIBackend` ABC plus one implementation per backend, mirroring the `BaseCompare` strategy pattern. A backend owns its session, its loaders, and its constructor call. `pyspark` and `snowflake.snowpark` imports stay inside methods.
- Backend applicability is data (`Opt.backends`), not an `if` chain. An option passed with a backend that does not accept it is rejected generically.
- With `--backend snowflake`, `--left` / `--right` are **always** table references. There is deliberately no file-versus-table heuristic.
- Exit codes are the contract: `0` match, `1` mismatch, `2` error, `130` interrupt. Expected failures raise `CLIError` subclasses; anything else propagates as a traceback.

## Code Conventions

- **Typing**: All code must be fully type-hinted and pass `mypy --strict`
- **Docstrings**: NumPy style
- **Imports**: Only absolute imports (relative imports banned via ruff TID252)
- **Pre-commit hooks**: ruff (lint + format), trailing whitespace, debug statements, end-of-file fixer, pyproject-fmt

## Testing Conventions

- Write plain pytest functions, not class-based suites. Use `def test_*()` at module level.
- Do not group tests into `class Test*` unless the file already does so.

## Documentation Conventions

- Do not use em dashes in documentation or docstrings; rewrite the sentence instead.
- Do not use emojis in documentation, docstrings, or commit messages.

## Branching

- `main` is the release branch and currently the most advanced one. Recent release commits land here, so branch from `main` unless told otherwise.
- `develop` predates the v1 GA and lags `main`. Do not assume it is the integration branch without checking `git log origin/main origin/develop`.
- `support/0.19.x` is maintained for v0 users (bug fixes only).

CI (`.github/workflows/test-package.yml`) runs on `develop`, `main`, `release/*`, `release-*`, and `support/*`.
