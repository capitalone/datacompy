# Backends

## Contents

Class table; constructor signatures; option support; install extras;
optional-import guard; one example per backend.

## Class table

Pick the class by the type you already have. Result frames
(`df1_unq_rows`, `df2_unq_rows`, `intersect_rows`, `all_mismatch()`,
`sample_mismatch()`) come back as the same type.

| Class | Import | Input | Result frames |
|---|---|---|---|
| `PandasCompare` | `from datacompy import PandasCompare` | `pandas.DataFrame` | `pandas.DataFrame` |
| `PolarsCompare` | `from datacompy import PolarsCompare` | `polars.DataFrame` | `polars.DataFrame` |
| `SparkSQLCompare` | `from datacompy import SparkSQLCompare` | `pyspark.sql.DataFrame` (including Spark Connect) | `pyspark.sql.DataFrame` |
| `SnowflakeCompare` | `from datacompy import SnowflakeCompare` | `snowflake.snowpark.DataFrame` or a `DB.SCHEMA.TABLE` string | `snowflake.snowpark.DataFrame` |

Pandas and Polars ship with the base install. Spark and Snowflake require
extras and are optional imports (see below).

## Constructor signatures

Spark and Snowflake take the session as the first argument. Pandas and
Polars do not.

```python
PandasCompare(df1, df2, join_columns=..., on_index=False, abs_tol=0, rel_tol=0, ...)
PolarsCompare(df1, df2, join_columns=..., abs_tol=0, rel_tol=0, ...)
SparkSQLCompare(spark_session, df1, df2, join_columns=..., abs_tol=0, rel_tol=0, ..., cache_intermediates=True)
SnowflakeCompare(session, df1, df2, join_columns=..., abs_tol=0, rel_tol=0, ...)
```

`join_columns` is a string or a list of strings. Pandas also accepts
`on_index=True` instead (not in addition). Snowflake rejects `None` or
an empty list.

## Option support

| Option | Pandas | Polars | Spark | Snowflake |
|---|---|---|---|---|
| `join_columns` | yes | required | required | required |
| `on_index` | yes | no | no | no |
| `cast_column_names_lower` | yes, default `True` | yes, default `True` | yes, default `True` | no; always uppercases |
| `cache_intermediates` | no | no | yes, default `True` | no |
| table-name strings | no | no | no | yes; must be `DB.SCHEMA.TABLE` |
| `ignore_spaces` / `ignore_case` | yes | yes | yes | yes |
| `custom_comparators` | yes | yes | yes | yes |

Notes that bite:

- `on_index` is Pandas only. Polars has no index. Passing `on_index` and
  `join_columns` together raises `ValueError: Only provide on_index or
  join_columns`.
- Snowflake has no `cast_column_names_lower`. Column names, join columns,
  df names, and tolerance dict keys are uppercased.
- Set `cache_intermediates=False` on Databricks Serverless (and any
  other Spark environment that cannot cache). The CLI flag is
  `--no-cache-intermediates`.
- A Snowflake table-name string is split on `.` and must have exactly
  three parts. `"SCHEMA.TABLE"` raises `ValueError`. Pass a Snowpark
  DataFrame if you already loaded the table.

Pandas and Polars lowercase column names **on the caller's frame** when
`cast_column_names_lower=True`. Spark uses `toDF(...)` and Snowflake
uses `rename(...)`, so they rebind internally and leave the caller's
object alone.

## Tolerances

`abs_tol` and `rel_tol` accept a float (applied to every column) or a
`dict[str, float]` (per-column). A value matches when
`abs(df1 - df2) <= abs_tol + rel_tol * abs(df2)`.

Prefer a float for a global tolerance. Use a dict only when columns need
different values. Dict keys are lowercased when
`cast_column_names_lower=True`, preserved when it is `False`, and
uppercased on Snowflake. Keys that do not match a column after that
normalization are ignored; that column is then compared with 0 (or with
the `"default"` dict entry if you included one).

```python
import pandas as pd
from datacompy import PandasCompare

df1 = pd.DataFrame({"id": [1], "amt": [1.00], "qty": [10]})
df2 = pd.DataFrame({"id": [1], "amt": [1.009], "qty": [10]})
print(PandasCompare(df1.copy(), df2.copy(), join_columns="id", abs_tol=0.01).matches())
print(
    PandasCompare(
        df1.copy(), df2.copy(), join_columns="id", abs_tol={"amt": 0.01, "qty": 0}
    ).matches()
)
```

## Install extras

```bash
pip install datacompy              # Pandas and Polars
pip install datacompy[spark]
pip install datacompy[snowflake]
```

On Python 3.12+ the `spark` extra resolves to PySpark 4.

## Optional-import guard

`datacompy/__init__.py` wraps Spark and Snowflake in `try/except
ImportError`. If the extra is missing, the name is simply absent.

```python
try:
    from datacompy import SparkSQLCompare
except ImportError:
    SparkSQLCompare = None

try:
    from datacompy import SnowflakeCompare
except ImportError:
    SnowflakeCompare = None
```

Do not import `datacompy.spark` or `datacompy.snowflake` unless the extra
is installed; those modules import `pyspark` / `snowflake.snowpark` at
load time. Importing `datacompy` itself will also pull in PySpark when
it is installed.

## Examples

Pandas (runnable as-is):

```python
import pandas as pd
from datacompy import PandasCompare

df1 = pd.DataFrame({"id": [1, 2, 3], "amt": [1.00, 2.00, 3.00]})
df2 = pd.DataFrame({"id": [1, 2, 4], "amt": [1.00, 2.20, 4.00]})
compare = PandasCompare(df1, df2, join_columns="id", abs_tol=0.05)
print(compare.matches())
print(compare.all_rows_overlap())
print(compare.df1_unq_rows)
print(compare.all_mismatch())
```

Polars (runnable as-is):

```python
import polars as pl
from datacompy import PolarsCompare

df1 = pl.DataFrame({"id": [1, 2, 3], "amt": [1.00, 2.00, 3.00]})
df2 = pl.DataFrame({"id": [1, 2, 4], "amt": [1.00, 2.20, 4.00]})
compare = PolarsCompare(df1, df2, join_columns="id", abs_tol=0.05)
print(compare.matches())
print(compare.all_rows_overlap())
print(compare.df1_unq_rows)
print(compare.all_mismatch())
```

Spark (needs `datacompy[spark]` and a session):

```python
from pyspark.sql import SparkSession
from datacompy import SparkSQLCompare

spark = SparkSession.builder.getOrCreate()
df1 = spark.createDataFrame([(1, 1.00), (2, 2.00)], ["id", "amt"])
df2 = spark.createDataFrame([(1, 1.00), (2, 2.02)], ["id", "amt"])
compare = SparkSQLCompare(
    spark,
    df1,
    df2,
    join_columns="id",
    abs_tol=0.05,
    cache_intermediates=False,  # Databricks Serverless
)
print(compare.matches())
```

Snowflake (needs `datacompy[snowflake]` and a session). Table names are
always `DB.SCHEMA.TABLE`:

```python
from datacompy import SnowflakeCompare

compare = SnowflakeCompare(
    session,
    "ANALYTICS.PUBLIC.BASE_TABLE",
    "ANALYTICS.PUBLIC.COMPARE_TABLE",
    join_columns="ID",
    abs_tol=0.05,
)
print(compare.matches())
```

Or pass Snowpark DataFrames as `df1` / `df2` instead of strings.
