# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataComPy is a Python library for comparing two DataFrames/tables across multiple backends: Pandas, Polars, Spark, and Snowflake. It originated as a replacement for SAS's `PROC COMPARE`. v1 is released; `datacompy/__init__.py`'s `__version__` is the source of truth for the current version.

## Common Commands

### Setup
```bash
pip install -e ".[dev]"
pre-commit install
```

### Testing

```bash
pytest                                          # all tests
pytest tests/test_pandas.py                     # single backend
pytest tests/test_pandas.py::TestPandasCompare::test_method  # single test
pytest --cov=datacompy --cov-report=term-missing  # with coverage

pytest -c pytest-connect.ini tests/test_spark.py tests/comparator/  # existing Spark suite, against Spark Connect
pytest -m spark_connect tests/test_spark_connect.py                # Spark Connect regression suite
```

The Makefile wraps each of these: `make test`, `test-ansi`, `test-connect`, `test-connect-regression`, `test-cov`, `test-all`, plus `test-no-snowflake` / `test-all-no-snowflake`.

There are three pytest configs, and `-c` **replaces** the config file rather than layering on it — anything a run needs (markers, `testpaths`, `spark_options`) must be present in the file it names:

- `pytest.ini` — classic Spark session, ANSI off
- `pytest-ansi.ini` — same, `spark.sql.ansi.enabled=true`
- `pytest-connect.ini` — Spark Connect session (`spark_connect_url`); must not set `spark.master`, since PySpark rejects a session with both

All three set `testpaths = tests`. Without it a bare `pytest` walks the whole repo, and a git worktree checked out under the repo root contributes a second `tests/conftest.py` that collides with the real one — pytest then aborts the entire run with `ImportPathMismatchError`. Keep PR-review worktrees outside the repo.

Spark tests require Java 17 and `pyspark` installed. Snowflake tests require a live Snowflake session; `--snowflake-session local` is *not* a substitute, as Snowpark's local testing mode is an emulator that most of the Snowflake suite fails against.

`benchmarks/` holds pytest-benchmark suites (`benchmark.py`, `benchmark_spark.py`). They sit outside `testpaths`, so no ordinary run collects them, and they read parquet fixtures that `python benchmarks/generate_data.py` has to write first. Results are written up in `docs/source/benchmark.rst`.

The two Spark Connect commands must each run in their own pytest process, and are excluded from the default run via `addopts`. Starting a local Spark Connect server sets `SPARK_LOCAL_REMOTE`, after which every later `SparkSession.builder.getOrCreate()` in that process returns the Connect session — so a classic and a Connect session cannot coexist in one run.

### Local CI matrix (tox)

`tox.ini` mirrors `.github/workflows/test-package.yml` job for job, using `tox-conda` so Java and PySpark come from conda-forge rather than a system JDK. It needs `pip install "tox<4" tox-conda` — tox-conda 0.10.x only supports tox 3.

```bash
tox                            # the whole envlist
tox -e lint                    # ruff check + ruff format --check
tox -e py312-spark4-pandas3    # the one env covering ANSI + Spark Connect
tox -e typecheck               # mypy (not in envlist; not run by CI)
```

Both files deliberately cover the test axes independently rather than as a cross product, because the Spark jobs are the entire cost of a run (~20 min each) while the four no-Spark jobs finish in under a minute. Python breadth comes from the cheap no-Spark envs; ANSI mode and Spark Connect are Spark-side semantics and run once on the 3.12 baseline. Read `tox.ini`'s header before widening the matrix — it records the known gap and what closing it costs.

Two traps that cost real time to rediscover, both documented in `tox.ini`:

- **Never loosen the `openjdk>=17.0.8,<18` floor** in the Spark envs. conda-forge ships GraalVM builds of openjdk 17 below that floor; selecting one pulls in `graalpy-graalvm`, which silently replaces CPython with GraalPy, and pip then dies with a Truffle `Unable to load native posix support library` error. It only affects Python 3.10 envs, since GraalPy implements 3.10.
- **`{envpython}` substitutes to the string `None` in tox 3.28** — use `{envbindir}/python`. The absolute path also avoids tox 3's silent fallback to a `$PATH` interpreter when a command is missing from a freshly-created env, which otherwise runs the suite against the wrong Python and only warns.

### Linting & Formatting
```bash
ruff check                 # lint
ruff check --fix           # lint with auto-fix
ruff format --check        # format check
ruff format                # apply formatting
mypy .                     # type-check (strict mode)
```

`pyproject.toml` uses selectors (`noqa-comments`, `rule-codes-in-selectors`) that only exist in recent ruff, hence the `ruff>=0.16` floor on the `qa` extra. An older ruff fails while *parsing* the config with `Unknown rule selector`, before linting anything. Note that ruff formats Python code blocks inside Markdown, so `CLAUDE.md` and other docs are in scope for `ruff format --check`.

`mypy` is **not** a pre-commit hook and is not run by CI — the hooks are ruff, ruff-format, trailing-whitespace, debug-statements, end-of-file-fixer, and pyproject-fmt. Run it explicitly or via `tox -e typecheck`.

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

- `base.py` → `BaseComparator` ABC with `compare(col1, col2, **kwargs)` method
- `numeric.py` → Numeric comparators per backend (handles tolerances)
- `string.py` → String comparators per backend
- `boolean.py` → Boolean comparators per backend
- `array.py` → Array-like comparators per backend
- `utility.py` → Shared Spark/Snowflake helpers, including `get_spark_functions` / `get_spark_window`

Each type has backend-specific implementations: `Pandas*Comparator`, `Polars*Comparator`, `Spark*Comparator`, `Snowflake*Comparator`.

**Dispatch protocol** — the column-compare helper in each backend module (`columns_equal` and friends) walks a comparator list in order and takes the first result that isn't `None`:

- **`compare()` returning `None` means "not my type, try the next one."** This is how type dispatch happens — there is no `can_compare()`. A comparator that raises instead of returning `None` breaks the chain.
- Order matters. Each backend defines `_<BACKEND>_DEFAULT_COMPARATORS` (array-like → boolean → numeric → string) at module scope.
- All four backends accept `custom_comparators=[...]` on the constructor; `_get_comparators()` puts them **before** the defaults, so a custom comparator can pre-empt a built-in for a column type.
- The dispatch loop passes the built-ins their type-specific kwargs via `isinstance` branches (tolerances to numeric, `ignore_spaces`/`ignore_case` to string), and passes custom comparators **everything** as `**kwargs`. Custom comparators must therefore accept and ignore kwargs they don't use.
- If every comparator returns `None`, the column compares as all-`False` rather than erroring.

### Spark Connect

Never import `pyspark.sql.functions` or `pyspark.sql.Window` at module scope in Spark code paths. Those dispatch to the Spark Connect implementations only when the process-global `SPARK_CONNECT_MODE_ENABLED` environment variable is set, which a Connect session from a notebook or serverless runtime does not necessarily set. Instead resolve them from the DataFrame or Column being operated on:

```python
F = get_spark_functions(dataframe)  # datacompy/spark.py
psf = get_spark_functions(dataframe)  # datacompy/comparator/*.py
Window = get_spark_window(dataframe)
```

Because there is no module-level binding, ruff's `F821` flags any call site that forgets the local. For the same reason, never import `pyspark.sql.connect.*` at module scope — that package requires the optional `grpcio` dependency, and `__init__.py`'s `except ImportError` would silently drop `SparkSQLCompare` from the package. Use `is_spark_connect_object()` instead.

### Reporting

Reports use Jinja2 templates from `datacompy/templates/report_template.j2`. The `render()` function in `base.py` handles template resolution. Custom templates can be passed via `report(template_path=...)`.

`build_report_data()` (defined once on `BaseCompare`) is the structured counterpart to `report()`, returning a typed `ReportData` object for dashboards or JSON export without parsing the string report. `ReportData` and its member dataclasses (`RowSummary`, `ColumnSummary`, `ColumnComparison`, `MismatchStat`, `MismatchStats`, `UniqueRowsData`) live in `datacompy/report.py` and are re-exported from `datacompy/__init__.py` — unlike the backends, they are unconditional top-level exports. It is public API — changes to its shape are breaking. Snapshot tests in `tests/test_report_snapshots.py` compare rendered output against fixtures in `tests/snapshots/`, which are excluded from the trailing-whitespace and end-of-file pre-commit hooks because their exact bytes are the assertion.

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

- `main` is the default branch and the target for all active development (this changed with the v1 release — `develop` still exists and CI still builds it, but it is no longer where work lands)
- `support/0.19.x` is archived: critical security fixes only, best-effort, no features or regular maintenance
- CI runs on pushes and PRs to `develop`, `main`, `release/*`, `release-*`, and `support/*`
