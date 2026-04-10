# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataComPy is a Python library for comparing two DataFrames/tables across multiple backends: Pandas, Polars, Spark, and Snowflake. It originated as a replacement for SAS's `PROC COMPARE`. Currently at v1.0.0 beta (pre-release on the `develop` branch).

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
```

Spark tests require Java 17 and `pyspark` installed. Snowflake tests require a live Snowflake session (or `--snowflake-session local` for local testing).

### Linting & Formatting
```bash
ruff check                 # lint
ruff check --fix           # lint with auto-fix
ruff format --check        # format check
ruff format                # apply formatting
mypy .                     # type-check (strict mode)
```

### Documentation
```bash
make sphinx                # build docs (runs in docs/ subdirectory)
```

## Architecture

### Strategy Pattern for Backend Comparisons

The core design uses the **Strategy pattern** with two abstraction layers:

1. **`BaseCompare`** (`datacompy/base.py`) — ABC defining the comparison interface. All backends implement: `_compare`, `_dataframe_merge`, `_intersect_compare`, `report`, `matches`, `subset`, `sample_mismatch`, `all_mismatch`, etc.

2. **Backend implementations** — Each in its own module:
   - `datacompy/pandas.py` → `PandasCompare`
   - `datacompy/polars.py` → `PolarsCompare`
   - `datacompy/spark.py` → `SparkSQLCompare`
   - `datacompy/snowflake.py` → `SnowflakeCompare`

Spark and Snowflake are optional imports (try/except in `__init__.py`).

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

## Code Conventions

- **Typing**: All code must be fully type-hinted and pass `mypy --strict`
- **Docstrings**: NumPy style
- **Imports**: Only absolute imports (relative imports banned via ruff TID252)
- **Pre-commit hooks**: ruff (lint + format), trailing whitespace, debug statements, end-of-file fixer, pyproject-fmt

## Branching

- `develop` is the active development branch for v1
- `main` is the release branch
- `support/0.19.x` maintained for v0 users (bug fixes only)
