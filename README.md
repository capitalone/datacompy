# DataComPy

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/datacompy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://badge.fury.io/py/datacompy.svg)](https://badge.fury.io/py/datacompy)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/datacompy/badges/version.svg)](https://anaconda.org/conda-forge/datacompy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/datacompy)


DataComPy is a package to compare two DataFrames (or tables) such as Pandas, Spark, Polars, and
even Snowflake. Originally it was created to be something of a replacement
for SAS's ``PROC COMPARE`` for Pandas DataFrames with some more functionality than
just ``Pandas.DataFrame.equals(Pandas.DataFrame)`` (in that it prints out some stats,
and lets you tweak how accurate matches have to be). Supported types include:

- Pandas
- Polars
- Spark (classic and Spark Connect)
- Snowflake

> [!IMPORTANT]
> datacompy has released `v1`. The `v0.19.x` line is no longer supported — users should upgrade to `v1` going forward.
> The `support/0.19.x` branch is archived and will only receive critical security fixes on a best-effort basis; no new features or regular maintenance will be provided.
> All active development targets `main`.


## Quick Installation

```shell
pip install datacompy
```

or

```shell
conda install datacompy
```

### Installing extras

If you would like to use Spark or any other backends please make sure you install via extras:

```shell
pip install datacompy[spark]
pip install datacompy[snowflake]

```


## Supported backends

- Pandas: ([See documentation](https://capitalone.github.io/datacompy/pandas_usage.html))
- Spark: ([See documentation](https://capitalone.github.io/datacompy/spark_usage.html))
- Polars: ([See documentation](https://capitalone.github.io/datacompy/polars_usage.html))
- Snowflake/Snowpark: ([See documentation](https://capitalone.github.io/datacompy/snowflake_usage.html))


## Command Line Interface

DataComPy ships a `datacompy` command, so ad hoc checks and CI pipelines do not
need a throw-away script ([see documentation](https://capitalone.github.io/datacompy/cli.html)):

```bash
# Compare two files and print a report
datacompy compare --left before.csv --right after.csv --on id

# Fail a build on any difference, with a JSON report saved as an artifact
datacompy compare \
    --left before.parquet --right after.parquet \
    --on account_id,as_of_date \
    --abs-tol balance=0.01 \
    --max-unequal-rows 0 \
    --report-format json --output reports/diff.json --quiet
```

It exits `0` when the datasets match, `1` when they differ, and `2` on error.
CSV, Parquet, and JSON inputs are supported on the pandas, polars, and Spark
backends, and Snowflake tables can be compared in place.

## Programmatic Report Access

Every compare object exposes `build_report_data()` which returns a typed
[`ReportData`](https://capitalone.github.io/datacompy/report_api.html) object
— useful for dashboards, JSON export, or custom rendering without relying on
the string report:

```python
import pandas as pd
from datacompy import PandasCompare

df1 = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
df2 = pd.DataFrame({"id": [1, 2, 3], "val": [10, 99, 30]})

compare = PandasCompare(df1, df2, join_columns="id")

# Access structured data directly
data = compare.build_report_data()
print(data.row_summary.unequal_rows)  # 1
print(data.mismatch_stats.stats[0].column)  # 'val'

# Render / export — methods live on ReportData itself
print(data.render())  # same text as compare.report()
data.save("report.html")  # HTML file
data.to_dict()  # JSON-serializable dict
```

See the [Report API documentation](https://capitalone.github.io/datacompy/report_api.html) for the full reference.


## Contributors

We welcome and appreciate your contributions! Before we can accept any contributions, we ask that you please be sure to
sign the [Contributor License Agreement (CLA)](https://cla-assistant.io/capitalone/datacompy).

This project adheres to the [Open Source Code of Conduct](https://developer.capitalone.com/resources/code-of-conduct/).
By participating, you are expected to honor this code.
