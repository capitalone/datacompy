Fugue Detail
============

`Fugue <https://github.com/fugue-project/fugue>`_ is a Python library that provides a unified interface
for data processing on Pandas, DuckDB, Polars, Arrow, Spark, Dask, Ray, and many other backends.
DataComPy integrates with Fugue to provide a simple way to compare data across these backends.

Basic Usage
-----------

The Fugue implementation can be accessed via the:

- ``datacompy.unq_columns``
- ``datacompy.intersect_columns``
- ``datacompy.all_columns_match``
- ``datacompy.all_rows_overlap``
- ``datacompy.is_match``
- and ``datacompy.report`` functions

Please note this is different than the native Pandas implementation which can be accessed via the ``Compare`` class,
the Fugue implementation is using the ``Compare`` class in the background though.

The following usage example compares two Pandas dataframes, it is equivalent to the Pandas usage example.

.. code-block:: python

    from io import StringIO
    import pandas as pd
    import datacompy

    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    """

    data2 = """acct_id,dollar_amt,name,float_fld
    10000001234,123.4,George Michael Bluth,14530.155
    10000001235,0.45,Michael Bluth,
    10000001236,1345,George Bluth,1
    10000001237,123456,Robert Loblaw,345.12
    10000001238,1.05,Loose Seal Bluth,111
    """

    df1 = pd.read_csv(StringIO(data1))
    df2 = pd.read_csv(StringIO(data2))

    datacompy.is_match(
        df1,
        df2,
        join_columns='acct_id',  #You can also specify a list of columns
        abs_tol=0, #Optional, defaults to 0
        rel_tol=0, #Optional, defaults to 0
        df1_name='Original', #Optional, defaults to 'df1'
        df2_name='New' #Optional, defaults to 'df2'
    )
    # False

    # This method prints out a human-readable report summarizing and sampling differences
    print(datacompy.report(
        df1,
        df2,
        join_columns='acct_id',  #You can also specify a list of columns
        abs_tol=0, #Optional, defaults to 0
        rel_tol=0, #Optional, defaults to 0
        df1_name='Original', #Optional, defaults to 'df1'
        df2_name='New' #Optional, defaults to 'df2'
    ))


Cross Comparing
---------------

In order to compare dataframes of different backends, you just need to replace ``df1`` and ``df2`` with
dataframes of different backends. Just pass in Dataframes such as Pandas dataframes, DuckDB relations,
Polars dataframes, Arrow tables, Spark dataframes, Dask dataframes or Ray datasets. For example,
to compare a Pandas dataframe with a Spark dataframe:

.. code-block:: python

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    spark_df2 = spark.createDataFrame(df2)
    datacompy.is_match(
        df1,
        spark_df2,
        join_columns='acct_id',
    )

Notice that in order to use a specific backend, you need to have the corresponding library installed.
For example, if you want compare Ray datasets, you must do

::

    pip install datacompy[ray]


How it works
------------

DataComPy uses Fugue to partition the two dataframes into chunks, and then compare each chunk in parallel
using the Pandas-based ``Compare``. The comparison results are then aggregated to produce the final result.
Different from the join operation used in ``SparkCompare``, the Fugue version uses the ``cogroup -> map``
like semantic (not exactly the same, Fugue adopts a coarse version to achieve great performance), which
guarantees full data comparison with consistent result compared to Pandas-based ``Compare``.


Future releases
---------------

We are hoping to pilot Fugue for the community in future releases (0.10+) and gather feedback. With Fugue we get the
benefits of not having to maintain Framework specific code, and also having cross-framework compatibility. We may in
future depending on feedback deprecate ``SparkCompare`` in favour of just using Fugue to manage non-Pandas use cases.
