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
- Spark
- Snowflake

> [!IMPORTANT]
> datacompy is progressing towards a `v1` release. During this transition, a `support/0.19.x` branch will be maintained solely for `v0.19.x` users.
> This branch will only receive dependency updates and critical bug fixes; no new features will be added.
> All new feature development should target the `v1` branches (`develop` and eventually `main`).


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


## Contributors

We welcome and appreciate your contributions! Before we can accept any contributions, we ask that you please be sure to
sign the [Contributor License Agreement (CLA)](https://cla-assistant.io/capitalone/datacompy).

This project adheres to the [Open Source Code of Conduct](https://developer.capitalone.com/resources/code-of-conduct/).
By participating, you are expected to honor this code.


## Roadmap

Roadmap details can be found [here](https://github.com/capitalone/datacompy/blob/develop/ROADMAP.rst)
