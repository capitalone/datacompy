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
pip install datacompy[fugue]
pip install datacompy[snowflake]

```

### Legacy Spark Deprecation

With version ``v0.12.0`` the original ``SparkCompare`` was replaced with a
Pandas on Spark implementation. The original ``SparkCompare`` implementation differs
from all the other native implementations. To align the API better,  and keep behaviour
consistent we are deprecating the original ``SparkCompare`` into a new module ``LegacySparkCompare``

Subsequently in ``v0.13.0`` a PySpark DataFrame class has been introduced (``SparkSQLCompare``)
which accepts ``pyspark.sql.DataFrame`` and should provide better performance. With this version
the Pandas on Spark implementation has been renamed to ``SparkPandasCompare`` and all the spark
logic is now under the ``spark`` submodule.

If you wish to use the old SparkCompare moving forward you can import it like so:

```python
from datacompy.spark.legacy import LegacySparkCompare
```

### SparkPandasCompare Deprecation

Starting with ``v0.14.1``, ``SparkPandasCompare`` is slated for deprecation. ``SparkSQLCompare`` is the prefered and much more performant.
It should be noted that if you continue to use ``SparkPandasCompare`` that ``numpy`` 2+ is not supported due to dependency issues.


#### Supported versions and dependncies

Different versions of Spark, Pandas, and Python interact differently. Below is a matrix of what we test with.
With the move to Pandas on Spark API and compatability issues with Pandas 2+ we will for the mean time note support Pandas 2
with the Pandas on Spark implementation. Spark plans to support Pandas 2 in [Spark 4](https://issues.apache.org/jira/browse/SPARK-44101)


|             | Spark 3.2.4 | Spark 3.3.4 | Spark 3.4.2 | Spark 3.5.1 |
|-------------|-------------|-------------|-------------|-------------|
| Python 3.10 | ✅           | ✅           | ✅           | ✅           |
| Python 3.11 | ❌           | ❌           | ✅           | ✅           |
| Python 3.12 | ❌           | ❌           | ❌           | ❌           |


|                        | Pandas < 1.5.3 | Pandas >=2.0.0 |
|------------------------|----------------|----------------|
| ``Compare``            | ✅              | ✅              |
| ``SparkPandasCompare`` | ✅              | ❌              |
| ``SparkSQLCompare``    | ✅              | ✅              |
| Fugue                  | ✅              | ✅              |



> [!NOTE]
> At the current time Python `3.12` is not supported by Spark and also Ray within Fugue.
> If you are using Python `3.12` and above, please note that not all functioanlity will be supported.
> Pandas and Polars support should work fine and are tested.

## Supported backends

- Pandas: ([See documentation](https://capitalone.github.io/datacompy/pandas_usage.html))
- Spark: ([See documentation](https://capitalone.github.io/datacompy/spark_usage.html))
- Polars: ([See documentation](https://capitalone.github.io/datacompy/polars_usage.html))
- Snowflake/Snowpark: ([See documentation](https://capitalone.github.io/datacompy/snowflake_usage.html))
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
