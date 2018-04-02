.. image:: https://travis-ci.org/capitalone/datacompy.svg?branch=master
    :target: https://travis-ci.org/capitalone/datacompy

=========
DataComPy
=========

DataComPy is a package to compare two Pandas DataFrames. Originally started to
be something of a replacement for SAS's ``PROC COMPARE`` for Pandas DataFrames
with some more functionality than just ``Pandas.DataFrame.equals(Pandas.DataFrame)``
(in that it prints out some stats, and lets you tweak how accurate matches have to be).
Then extended to carry that functionality over to Spark Dataframes.

Quick Installation
==================

::

    pip install datacompy

Pandas Detail
=============

DataComPy will try to join two dataframes either on a list of join columns, or
on indexes.  If the two dataframes have duplicates based on join values, the
match process sorts by the remaining fields and joins based on that row number.

Column-wise comparisons attempt to match values even when dtypes don't match.
So if, for example, you have a column with ``decimal.Decimal`` values in one
dataframe and an identically-named column with ``float64`` dtype in another,
it will tell you that the dtypes are different but will still try to compare the
values.

Basic Usage
-----------

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

    compare = datacompy.Compare(
        df1,
        df2,
        join_columns='acct_id',  #You can also specify a list of columns
        abs_tol=0, #Optional, defaults to 0
        rel_tol=0, #Optional, defaults to 0
        df1_name='Original', #Optional, defaults to 'df1'
        df2_name='New' #Optional, defaults to 'df2'
        )
    compare.matches(ignore_extra_columns=False)
    # False

    # This method prints out a human-readable report summarizing and sampling differences
    print(compare.report())

See docs for more detailed usage instructions and an example of the report output.

Things that are happening behind the scenes
-------------------------------------------

- You pass in two dataframes (``df1``, ``df2``) to ``datacompy.Compare`` and a
  column to join on (or list of columns) to ``join_columns``.  By default the
  comparison needs to match values exactly, but you can pass in ``abs_tol``
  and/or ``rel_tol`` to apply absolute and/or relative tolerances for numeric columns.

  - You can pass in ``on_index=True`` instead of ``join_columns`` to join on
    the index instead.

- The class validates that you passed dataframes, that they contain all of the
  columns in `join_columns` and have unique column names other than that.  The
  class also lowercases all column names to disambiguate.
- On initialization the class validates inputs, and runs the comparison.
- ``Compare.matches()`` will return ``True`` if the dataframes match, ``False``
  otherwise.

  - You can pass in ``ignore_extra_columns=True`` to not return ``False`` just
    because there are non-overlapping column names (will still check on
    overlapping columns)
  - NOTE: if you only want to validate whether a dataframe matches exactly or
    not, you should look at ``pandas.testing.assert_frame_equal``.  The main
    use case for ``datacompy`` is when you need to interpret the difference
    between two dataframes.

- Compare also has some shortcuts like

  - ``intersect_rows``, ``df1_unq_rows``, ``df2_unq_rows`` for getting
    intersection, just df1 and just df2 records (DataFrames)
  - ``intersect_columns()``, ``df1_unq_columns()``, ``df2_unq_columns()`` for
    getting intersection, just df1 and just df2 columns (Sets)

- You can turn on logging to see more detailed logs.

.. _spark-detail:

Spark Detail
============

DataComPy's ``SparkCompare`` class will join two dataframes either on a list of join
columns. It has the capability to map column names that may be different in each
dataframe, including in the join columns. You are responsible for creating the
dataframes from any source which Spark can handle and specifying a unique join
key. If there are duplicates in either dataframe by join key, the match process
will remove the duplicates before joining (and tell you how many duplicates were
found).

As with the Pandas-based ``Compare`` class, comparisons will be attempted even
if dtypes don't match. Any schema differences will be reported in the output
as well as in any mismatch reports, so that you can assess whether or not a
type mismatch is a problem or not.

The main reasons why you would choose to use ``SparkCompare`` over ``Compare``
are that your data is too large to fit into memory, or you're comparing data
that works well in a Spark environment, like partitioned Parquet, CSV, or JSON
files, or Cerebro tables.

Performance Implications
------------------------

Spark scales incredibly well, so you can use ``SparkCompare`` to compare
billions of rows of data, provided you spin up a big enough cluster. Still,
joining billions of rows of data is an inherently large task, so there are a
couple of things you may want to take into consideration when getting into the
cliched realm of "big data":

* ``SparkCompare`` will compare all columns in common in the dataframes and
  report on the rest. If there are columns in the data that you don't care to
  compare, use a ``select`` statement/method on the dataframe(s) to filter
  those out. Particularly when reading from wide Parquet files, this can make
  a huge difference when the columns you don't care about don't have to be
  read into memory and included in the joined dataframe.
* For large datasets, adding ``cache_intermediates=True`` to the ``SparkCompare``
  call can help optimize performance by caching certain intermediate dataframes
  in memory, like the de-duped version of each input dataset, or the joined
  dataframe. Otherwise, Spark's lazy evaluation will recompute those each time
  it needs the data in a report or as you access instance attributes. This may
  be fine for smaller dataframes, but will be costly for larger ones. You do
  need to ensure that you have enough free cache memory before you do this, so
  this parameter is set to False by default.

Basic Usage
-----------

.. code-block:: python

    import datetime
    import datacompy
    from pyspark.sql import Row

    # This example assumes you have a SparkSession named "spark" in your environment, as you
    # do when running `pyspark` from the terminal or in a Databricks notebook (Spark v2.0 and higher)

    data1 = [
        Row(acct_id=10000001234, dollar_amt=123.45, name='George Maharis', float_fld=14530.1555,
            date_fld=datetime.date(2017, 1, 1)),
        Row(acct_id=10000001235, dollar_amt=0.45, name='Michael Bluth', float_fld=1.0,
            date_fld=datetime.date(2017, 1, 1)),
        Row(acct_id=10000001236, dollar_amt=1345.0, name='George Bluth', float_fld=None,
            date_fld=datetime.date(2017, 1, 1)),
        Row(acct_id=10000001237, dollar_amt=123456.0, name='Bob Loblaw', float_fld=345.12,
            date_fld=datetime.date(2017, 1, 1)),
        Row(acct_id=10000001239, dollar_amt=1.05, name='Lucille Bluth', float_fld=None,
            date_fld=datetime.date(2017, 1, 1))
    ]

    data2 = [
        Row(acct_id=10000001234, dollar_amt=123.4, name='George Michael Bluth', float_fld=14530.155),
        Row(acct_id=10000001235, dollar_amt=0.45, name='Michael Bluth', float_fld=None),
        Row(acct_id=10000001236, dollar_amt=1345.0, name='George Bluth', float_fld=1.0),
        Row(acct_id=10000001237, dollar_amt=123456.0, name='Robert Loblaw', float_fld=345.12),
        Row(acct_id=10000001238, dollar_amt=1.05, name='Loose Seal Bluth', float_fld=111.0)
    ]

    base_df = spark.createDataFrame(data1)
    compare_df = spark.createDataFrame(data2)

    comparison = datacompy.SparkCompare(spark, base_df, compare_df, join_columns=['acct_id'])

    # This prints out a human-readable report summarizing differences
    comparison.report()

Using SparkCompare on EMR or standalone Spark
---------------------------------------------

1. Set proxy variables
2. Create a virtual environment, if desired (``virtualenv venv; source venv/bin/activate``)
3. Pip install datacompy and requirements
4. Ensure your SPARK_HOME environment variable is set (this is probably ``/usr/lib/spark`` but may
   differ based on your installation)
5. Augment your PYTHONPATH environment variable with
   ``export PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.10.4-src.zip:$SPARK_HOME/python:$PYTHONPATH``
   (note that your version of py4j may differ depending on the version of Spark you're using)


Using SparkCompare on Databricks
--------------------------------

1. Clone this repository locally
2. Create a datacompy egg by running ``python setup.py bdist_egg`` from the repo root directory.
3. From the Databricks front page, click the "Library" link under the "New" section.
4. On the New library page:
    a. Change source to "Upload Python Egg or PyPi"
    b. Under "Upload Egg", Library Name should be "datacompy"
    c. Drag the egg file in datacompy/dist/ to the "Drop library egg here to upload" box
    d. Click the "Create Library" button
5. Once the library has been created, from the library page (which you can find in your /Users/{login} workspace),
   you can choose clusters to attach the library to.
6. ``import datacompy`` in a notebook attached to the cluster that the library is attached to and enjoy!

Contributors
------------

We welcome your interest in Capital One’s Open Source Projects (the "Project").
Any Contributor to the project must accept and sign a CLA indicating agreement to
the license terms. Except for the license granted in this CLA to Capital One and
to recipients of software distributed by Capital One, you reserve all right, title,
and interest in and to your contributions; this CLA does not impact your rights to
use your own contributions for any other purpose.

- `Link to Individual CLA <https://docs.google.com/forms/d/19LpBBjykHPox18vrZvBbZUcK6gQTj7qv1O5hCduAZFU/viewform>`_
- `Link to Corporate CLA <https://docs.google.com/forms/d/e/1FAIpQLSeAbobIPLCVZD_ccgtMWBDAcN68oqbAJBQyDTSAQ1AkYuCp_g/viewform>`_

This project adheres to the `Open Source Code of Conduct <https://developer.capitalone.com/single/code-of-conduct/>`_.
By participating, you are expected to honor this code.
