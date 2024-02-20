# DataComPy

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/datacompy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPI version](https://badge.fury.io/py/datacompy.svg)](https://badge.fury.io/py/datacompy)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/datacompy/badges/version.svg)](https://anaconda.org/conda-forge/datacompy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/datacompy)


DataComPy is a package to compare two Pandas DataFrames. Originally started to
be something of a replacement for SAS's ``PROC COMPARE`` for Pandas DataFrames
with some more functionality than just ``Pandas.DataFrame.equals(Pandas.DataFrame)``
(in that it prints out some stats, and lets you tweak how accurate matches have to be).
Then extended to carry that functionality over to Spark Dataframes.

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
pip install datacompy[dask]
pip install datacompy[duckdb]
pip install datacompy[polars]
pip install datacompy[ray]

```

### In-scope Spark versions
Different versions of Spark play nicely with only certain versions of Python below is a matrix of what we test with

|             | Spark 3.1.3  | Spark 3.2.3 | Spark 3.3.4 | Spark 3.4.2 | Spark 3.5.0 |
|-------------|--------------|-------------|-------------|-------------|-------------|
| Python 3.8  | ✅           | ✅           | ✅           | ✅          | ✅          |
| Python 3.9  | ✅           | ✅           | ✅           | ✅          | ✅          |
| Python 3.10 | ✅           | ✅           | ✅           | ✅          | ✅          |
| Python 3.11 | ❌           | ❌           | ❌           | ✅          | ✅          |
| Python 3.12 | ❌           | ❌           | ❌           | ❌          | ❌          |


:::{note}
At the current time Python ``3.12`` is not supported by Spark and also Ray within Fugue.
:::

## Supported backends

- Pandas: ([See documentation](https://capitalone.github.io/datacompy/pandas_usage.html))
- Spark: ([See documentation](https://capitalone.github.io/datacompy/spark_usage.html))
- Polars (Experimental): ([See documentation](https://capitalone.github.io/datacompy/polars_usage.html))
- Fugue is a Python library that provides a unified interface for data processing on Pandas, DuckDB, Polars, Arrow, 
  Spark, Dask, Ray, and many other backends. DataComPy integrates with Fugue to provide a simple way to compare data 
  across these backends. Please note that Fugue will use the Pandas (Native) logic at its lowest level 
  ([See documentation](https://capitalone.github.io/datacompy/fugue_usage.html))

## Contributors

We welcome and appreciate your contributions! Before we can accept any contributions, we ask that you please be sure to
sign the [Contributor License Agreement (CLA)](https://cla-assistant.io/capitalone/datacompy).

This project adheres to the [Open Source Code of Conduct](https://developer.capitalone.com/resources/code-of-conduct/).
By participating, you are expected to honor this code.


## Roadmap

Roadmap details can be found [here](https://github.com/capitalone/datacompy/blob/develop/ROADMAP.rst)
