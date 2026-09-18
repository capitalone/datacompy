# Migrating v0 to v1

## Contents

Why this file exists; class and import map; Spark name history;
removed Fugue / Dask / DuckDB / Ray; v0 wheels and branches; behavior
that changed in v1; a worked rewrite.

## Why this file exists

Models trained on pre-1.0 material default to `datacompy.Compare`,
`SparkCompare`, and `datacompy.fugue`. Those names are gone. Write v1
even if the user's snippet is v0.

The v0 API lives on the `support/0.19.x` branch and on older PyPI
wheels (`0.19.5`, `0.13.3`, `0.10.5`). v1 is DataComPy 1.0 and later.

## Class and import map

| v0 | v1 |
|---|---|
| `from datacompy.core import Compare` or `from datacompy import Compare` | `from datacompy import PandasCompare` |
| `from datacompy.spark.sql import SparkSQLCompare` | `from datacompy import SparkSQLCompare` |
| `SparkPandasCompare` (0.12-0.13; named `SparkCompare` in 0.12) | removed. Call `psdf.to_spark()`, then use `SparkSQLCompare` |
| `SparkCompare` / `LegacySparkCompare` (native Spark, `base_df` / `compare_df`) | `SparkSQLCompare`. `base_df` / `compare_df` become `df1` / `df2`; `rows_only_base` becomes `df1_unq_rows`; `rows_only_compare` becomes `df2_unq_rows`; `rows_both_mismatch` becomes `all_mismatch()`; `report(file=...)` becomes `report()`, which returns a string |
| `datacompy.fugue` (`is_match`, `report`, `all_columns_match`, `all_rows_overlap`, `count_matching_rows`, ...) | removed. Use the native backend class. There is no v1 path for Dask, DuckDB, or Ray |
| `from datacompy.polars import PolarsCompare` | `from datacompy import PolarsCompare` (name unchanged) |
| `from datacompy.snowflake import SnowflakeCompare` | `from datacompy import SnowflakeCompare` (optional import) |

`from datacompy import Compare` worked through 0.18.x. On 0.19.x the
package `__init__.py` no longer re-exports `Compare`; `from
datacompy.core import Compare` still works on that line. Neither import
exists in v1.

`PolarsCompare` was already the v0 name. Do not rename it to `Compare`.

## Spark name history (check the wheel, not memory)

The name `SparkCompare` meant different things:

- **0.10.x:** `datacompy.spark.SparkCompare` is native Spark. Constructor
  is `(spark_session, base_df, compare_df, join_columns=...)`.
  `report(file=sys.stdout)` writes to a file handle and returns `None`.
  Unique rows are `rows_only_base` / `rows_only_compare`; mismatches are
  `rows_both_mismatch`.
- **0.12.0:** that native class became `LegacySparkCompare`. `SparkCompare`
  was rewritten on pandas-on-Spark with `df1` / `df2`, like Pandas.
- **0.13.x:** pandas-on-Spark was renamed `SparkPandasCompare`. Native SQL
  is `SparkSQLCompare` in `datacompy.spark.sql`. `LegacySparkCompare`
  remains in `datacompy.spark.legacy`.
- **0.17+:** `SparkPandasCompare` and `LegacySparkCompare` are removed.
  `SparkSQLCompare` is the only Spark class.
- **v1:** `SparkSQLCompare` lives in `datacompy.spark` (a module, not a
  package). `from datacompy.spark.sql import SparkSQLCompare` fails.

Pandas-on-Spark frames are not a v1 input type. Convert with
`psdf.to_spark()` and pass the result to `SparkSQLCompare`.

## Fugue, Dask, DuckDB, Ray

v0 extras included `datacompy[fugue]`, which compared Pandas, Polars,
Spark, Dask, DuckDB, and Ray through Fugue wrappers around the Pandas
logic. That module is deleted in v1.

Replacement:

- Pandas, Polars, Spark, Snowflake: the native v1 class.
- Dask, DuckDB, Ray: no v1 backend. Convert to Pandas or Polars first
  only if the data fits, or tell the user there is no native path.

Do not suggest `pip install datacompy[fugue]`.

## Behavior that changed in v1

- **Class names** as in the table above. Constructor kwargs `df1` /
  `df2` / `join_columns` are the v1 names for every backend.
- **A numeric string no longer matches a number.** In 0.19.5,
  `Compare` of `{"v": ["1.0"]}` vs `{"v": [1.0]}` could succeed after
  coercion. `PandasCompare` in v1 returns `matches() == False`.
- **`report()` returns a `str` on every v1 class.** The 0.10
  `SparkCompare.report(file=...)` side-effect API is gone. Save HTML
  with `report(html_file="out.html")` or
  `compare.build_report_data().save("out.html")`.
- **Sensitive columns** are `hide_sensitive_columns(["col"])` after
  construct, not a constructor argument.
- **Spark and Snowflake are optional.** `import datacompy` succeeds
  without those extras; the class names are missing until you install
  them.

## Worked rewrite

v0 (0.18 / `datacompy.core`):

```python
from datacompy.core import Compare

compare = Compare(df1, df2, join_columns="id", abs_tol=0.01)
print(compare.matches())
print(compare.report())
```

v1:

```python
import pandas as pd
from datacompy import PandasCompare

df1 = pd.DataFrame({"id": [1, 2], "amt": [1.00, 2.00]})
df2 = pd.DataFrame({"id": [1, 2], "amt": [1.00, 2.009]})
compare = PandasCompare(df1, df2, join_columns="id", abs_tol=0.01)
print(compare.matches())
print(compare.report())
```

v0 native Spark (0.10.x / `LegacySparkCompare`):

```python
from datacompy.spark import SparkCompare

compare = SparkCompare(spark, base_df, compare_df, join_columns=["id"])
compare.rows_only_base.show()
compare.rows_both_mismatch.show()
compare.report()
```

v1:

```python
from datacompy import SparkSQLCompare

compare = SparkSQLCompare(spark, base_df, compare_df, join_columns="id")
compare.df1_unq_rows.show()
compare.all_mismatch().show()
print(compare.report())
```
