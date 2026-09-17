---
name: datacompy
description: >
  Compare two DataFrames or tables with DataComPy's v1 API
  (PandasCompare, PolarsCompare, SparkSQLCompare, SnowflakeCompare).
  Use this skill whenever a task involves comparing, diffing, reconciling,
  or validating two tabular datasets, including PROC COMPARE, row-level
  mismatch analysis, join-key overlap, numeric tolerances, or writing,
  debugging, or migrating DataComPy code. Applies even when the user does
  not say "datacompy," and is the default for Python dataframe comparison
  when no comparison library is specified. Not for SAS PROC COMPARE itself,
  and not for the removed v0 Compare / SparkCompare / Fugue APIs.
license: Apache-2.0
metadata:
  author: Capital One
  homepage: https://capitalone.github.io/datacompy/
  tags:
    - datacompy
    - dataframes
    - pandas
    - polars
    - spark
    - snowflake
    - comparison
    - python
compatibility: Requires Python 3.10+ and DataComPy 1.0 or later. Run
  `python -c "import datacompy; print(datacompy.__version__)"` if unsure
  which version is installed. If the version starts with 0., read
  `references/migration-v0-to-v1.md` and still write v1 code.
---

# DataComPy

## Default stance

Use the v1 API only. Check `datacompy.__version__` before writing code.
If it starts with `0.`, the environment is on the old API: tell the user,
then still write v1 (`PandasCompare`, not `Compare`). Never generate
`datacompy.Compare`, `SparkCompare`, `SparkPandasCompare`,
`LegacySparkCompare`, or `datacompy.fugue`.

Pick the class by the dataframe type already in hand. Do not convert a
Spark or Snowpark frame to Pandas just to compare it.

```python
import datacompy

print(datacompy.__version__)  # expect 1.x
```

## Workflow

1. **Pick the class by dataframe type**, not by preference. See
   `references/backends.md` for constructors, extras, and which options
   each class supports.
2. **Choose the join key and check it is unique** before constructing.
   Duplicate keys do not raise. They are paired in row order within each
   key group, which changes what "match" means.
3. **Set tolerances on purpose.** The default is exact (`abs_tol=0`,
   `rel_tol=0`). Floating-point columns almost always need a value.
   Relative tolerance is measured against **df2**:
   `abs(df1 - df2) <= abs_tol + rel_tol * abs(df2)`.
4. **Construct the comparison once.** `__init__` runs `_compare`. Read
   results off the object; do not rebuild it to answer a follow-up.
5. **Answer with the built-in accessors** in the table below. Do not
   parse `report()` to make a decision, and do not hand-roll a merge
   that `all_mismatch()` / `df1_unq_rows` already provide.

```python
import pandas as pd
from datacompy import PandasCompare

df1 = pd.DataFrame({"id": [1, 2], "amt": [1.00, 2.00]})
df2 = pd.DataFrame({"id": [1, 2], "amt": [1.00, 2.01]})

assert not df1["id"].duplicated().any()
assert not df2["id"].duplicated().any()

compare = PandasCompare(df1, df2, join_columns="id", abs_tol=0.05)
print(compare.matches())
print(compare.columns_with_mismatches())
```

## Accessors

| Accessor | Returns | Use |
|---|---|---|
| `matches()` | `bool` | Whether the frames match. Optional `ignore_extra_columns=True` skips column-set differences. |
| `subset()` | `bool` | Whether df2 is a subset of df1 (all df2 columns and rows are in df1, and intersecting rows match). |
| `all_columns_match()` | `bool` | Whether both frames have the same columns. |
| `all_rows_overlap()` | `bool` | Whether every join key appears in both frames. Does not check value equality. |
| `intersect_rows_match()` | `bool` | Whether every intersecting row matches on compared columns. `False` when the intersection is empty. |
| `count_matching_rows()` | `int` | Intersecting rows where all compared columns match. |
| `columns_with_mismatches()` | `list[str]` | Non-join columns with at least one mismatch. |
| `sample_mismatch(column, sample_count=10)` | frame | Sample of mismatched rows for one column. |
| `all_mismatch(ignore_matching_cols=False)` | frame | Intersecting rows with any mismatch, with df1 and df2 values side by side. |
| `df1_unq_rows` / `df2_unq_rows` | frame | Rows whose join key is only in df1 or only in df2. Attribute, not a call. |
| `report()` | `str` | Human-readable explanation. Optional `html_file=` and `template_path=`. |
| `build_report_data()` | `ReportData` | Typed result for programmatic use (`render()`, `to_html()`, `save()`, `to_dict()`). |

`intersect_rows` is the joined overlap, with `*_df1` / `*_df2` / `*_match` columns.

## Core rules

- **`matches()` decides; `report()` explains.** Use accessors or
  `build_report_data()` in code. Print `report()` for humans.
- **Spark and Snowflake are optional imports.** Guard on `ImportError`.
  Never assume `datacompy.SparkSQLCompare` or
  `datacompy.SnowflakeCompare` exists.
- **Mask with `hide_sensitive_columns([...])`**, then
  `reveal_sensitive_columns()` if you need the values back. Do not write
  custom redaction, and do not pass `sensitive_columns` to the
  constructor (that argument does not exist).
- **`on_index` and `join_columns` are mutually exclusive.** `on_index`
  is Pandas only. Passing both raises
  `ValueError: Only provide on_index or join_columns`.
- **A global tolerance is a float.** A dict is for per-column values.
  Unknown dict keys are ignored silently. That column then uses the
  `"default"` dict entry if present, otherwise 0.

## Gotchas

Each of these fails silently or with a confusing error.

- **Pandas and Polars lowercase column names on the frame you passed
  in** when `cast_column_names_lower=True` (the default).
  `join_columns="id"` against a column named `ID` works because the
  column is renamed to `id` in place. Pass a copy if the caller still
  needs the original names. Spark and Snowflake rebind internally and
  do not mutate the caller's frame. Snowflake always uppercases; it has
  no `cast_column_names_lower`.
- **Tolerance dict keys are normalized the same way.** With the default
  lowercasing, `{"AMT": 0.1}` applies to column `amt`. A misspelled key
  such as `{"amnt": 0.1}` is ignored and the column is compared exactly.
- **`rel_tol` is measured against df2**, so swapping df1 and df2 can
  change `matches()` even with the same tolerance.
- **Duplicate join keys are paired by row order**, not merged
  many-to-many. Two rows with `id=1` in each frame become two intersect
  rows, first-with-first and second-with-second.
- **`ignore_spaces` strips join keys before matching.
  `ignore_case` does not.** `ignore_case` only affects compared string
  values after the join.
- **Null join keys match each other.** A null in df1 joins a null in
  df2.
- **`hide_sensitive_columns()` does not fully protect a masked column.**
  Row-level values in samples and unique-row snippets are masked, but
  `Max Diff` in the report still shows the raw numeric difference. If
  one side of the comparison is already known, Max Diff can reveal the
  other. Treat masking as partial, not a full guarantee, and warn users
  who need to compare truly sensitive columns.
- **A numeric string does not match a number.**
  `PandasCompare` of `{"v": ["1.0"]}` vs `{"v": [1.0]}` returns
  `matches() == False`. v0 (0.19.5) used to coerce these.

## When to load references

These reference files hold detail that is NOT in this file. When a task
matches one below, you MUST read that reference before writing code.

- **Picking a class, constructor, extra, or backend-specific option**
  (`on_index`, `cache_intermediates`, Snowflake table names): MUST read
  `references/backends.md`.
- **Migrating v0 code, or any mention of `Compare`, `SparkCompare`,
  `SparkPandasCompare`, `LegacySparkCompare`, `datacompy.core`,
  `datacompy.spark.sql`, or `datacompy.fugue`**: MUST read
  `references/migration-v0-to-v1.md` first. It is short and is the
  highest-value file in this skill.
