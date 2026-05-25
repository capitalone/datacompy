# DataComPy AI Assistant Guide

This guide provides essential context for AI coding agents to be productive in the DataComPy codebase.

## Architecture Overview

- **Strategy Pattern**: The core logic uses the Strategy design pattern. The abstract base class `datacompy.base.BaseCompare` defines the interface for all comparison operations. Each backend (Pandas, Spark, Polars, Snowflake) has a concrete implementation (`datacompy.pandas.PandasCompare`, etc.) that inherits from `BaseCompare` and implements its methods.
- **Comparison Reports**: Reports are generated via Jinja2 templates in `datacompy/templates/`, with the main template being `report_template.j2`. All backends use a similar reporting interface.
- **Extensibility**: To add or modify comparison logic for a backend, update the corresponding class. For changes affecting all backends, start with `BaseCompare`.

## Developer Workflow

- **Environment Setup**:
  ```bash
  pip install -e ".[dev]"
  pre-commit install
  ```
- **Testing**:
  - Run all tests: `pytest`
  - Backend-specific tests: see `tests/test_pandas.py`, `tests/test_spark.py`, etc.
- **Linting & Formatting**:
  - Lint: `ruff check`
  - Format: `ruff format --check`
  - Type-check: `mypy .` (strict mode)
  - All are enforced via pre-commit hooks.
- **Documentation**:
  - Build docs: `make -C docs html`
  - Docs source: `docs/source/`, output: `docs/build/html/`

## Code Conventions

- **Typing**: All code must be fully type-hinted and pass `mypy --strict`.
- **Docstrings**: Use [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).
- **Imports**: Only absolute imports are allowed (see `pyproject.toml`).
- **Templates**: All reporting uses Jinja2 templates in `datacompy/templates/`.
- **Backend-specific logic**: Each backend file (`pandas.py`, `spark.py`, `polars.py`, `snowflake.py`) implements the same interface and reporting pattern.

## Patterns & Examples

- **Comparison Usage**:
  ```python
  from datacompy import PandasCompare
  compare = PandasCompare(df1, df2, join_columns=[...])
  print(compare.report())
  ```
- **Custom Templates**:
  ```python
  compare.report(template_path="custom_report.j2")
  ```
- **Tolerance Handling**: Tolerances can be set globally or per-column (see `validate_tolerance_parameter`).
- **Unique/Intersect Rows**: Each backend exposes `df1_unq_rows`, `df2_unq_rows`, and `intersect_rows` for advanced analysis.

## Integration Points

- **Dependencies**: Core dependencies are in `pyproject.toml` (Jinja2, pandas, polars, pyspark, snowflake-snowpark-python, etc.).
- **Pre-commit**: Linting, formatting, and type-checking are enforced via pre-commit hooks.
- **Builds**: Use the Makefile for docs (`make sphinx`).
- **CI**: See `.github/workflows/` for test and lint automation.

---

If any section is unclear or missing, please provide feedback to iterate and improve these instructions.
