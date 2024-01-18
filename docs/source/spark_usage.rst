Spark Usage
===========

.. important::

    With version ``v0.9.0`` SparkCompare now uses Null Safe (``<=>``) comparisons


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


Known Differences
-----------------

For cases when two dataframes are expected to differ, it can be helpful to cluster detected
differences into three categories: matches, known differences, and true mismatches. Known
differences can be specified through an optional parameter:

.. code-block:: python

    SparkCompare(spark, base_df, compare_df, join_columns=[...], column_mapping=[...],
        known_differences = [
            {
             'name':  "My Known Difference Name",
             'types': ['int', 'bigint'],
             'flags': ['nullcheck'],
             'transformation': "case when {input}=0 then null else {input} end"
            },
            ...    
        ]
    )

The 'known_differences' parameter is a list of Python dicts with the following fields:

============== ========= ======================================================================
Field          Required? Description
============== ========= ======================================================================
name           yes       A user-readable title for this known difference
types          yes       A list of Spark data types on which this transformation can be applied
flags          no        Special flags used for computing known differences
transformation yes       Spark SQL function to apply, where {input} is a cell in the comparison
============== ========= ======================================================================

Valid flags are:

========= =============================================================
Flag      Description
========= =============================================================
nullcheck Must be set when the output of the transformation can be null
========= =============================================================

Transformations are applied to the compare side only. A known difference is found when transformation(compare.cell) equals base.cell. An example comparison is shown below.

.. code-block:: python

    import datetime
    import datacompy
    from pyspark.sql import Row
    
    base_data = [
        Row(acct_id=10000001234, acct_sfx_num=0, clsd_reas_cd='*2', open_dt=datetime.date(2017, 5, 1), tbal_cd='0001'),
        Row(acct_id=10000001235, acct_sfx_num=0, clsd_reas_cd='V1', open_dt=datetime.date(2017, 5, 2), tbal_cd='0002'),
        Row(acct_id=10000001236, acct_sfx_num=0, clsd_reas_cd='V2', open_dt=datetime.date(2017, 5, 3), tbal_cd='0003'),
        Row(acct_id=10000001237, acct_sfx_num=0, clsd_reas_cd='*2', open_dt=datetime.date(2017, 5, 4), tbal_cd='0004'),
        Row(acct_id=10000001238, acct_sfx_num=0, clsd_reas_cd='*2', open_dt=datetime.date(2017, 5, 5), tbal_cd='0005')
    ]
    base_df = spark.createDataFrame(base_data) 

    compare_data = [
        Row(ACCOUNT_IDENTIFIER=10000001234, SUFFIX_NUMBER=0, AM00_STATC_CLOSED=None, AM00_DATE_ACCOUNT_OPEN=2017121, AM0B_FC_TBAL=1.0),
        Row(ACCOUNT_IDENTIFIER=10000001235, SUFFIX_NUMBER=0, AM00_STATC_CLOSED='V1', AM00_DATE_ACCOUNT_OPEN=2017122, AM0B_FC_TBAL=2.0),
        Row(ACCOUNT_IDENTIFIER=10000001236, SUFFIX_NUMBER=0, AM00_STATC_CLOSED='V2', AM00_DATE_ACCOUNT_OPEN=2017123, AM0B_FC_TBAL=3.0),
        Row(ACCOUNT_IDENTIFIER=10000001237, SUFFIX_NUMBER=0, AM00_STATC_CLOSED='V3', AM00_DATE_ACCOUNT_OPEN=2017124, AM0B_FC_TBAL=4.0),
        Row(ACCOUNT_IDENTIFIER=10000001238, SUFFIX_NUMBER=0, AM00_STATC_CLOSED=None, AM00_DATE_ACCOUNT_OPEN=2017125, AM0B_FC_TBAL=5.0)
    ]
    compare_df = spark.createDataFrame(compare_data)

    comparison = datacompy.SparkCompare(spark, base_df, compare_df,
                        join_columns =   [('acct_id', 'ACCOUNT_IDENTIFIER'), ('acct_sfx_num', 'SUFFIX_NUMBER')],
                        column_mapping = [('clsd_reas_cd', 'AM00_STATC_CLOSED'),
                                          ('open_dt', 'AM00_DATE_ACCOUNT_OPEN'),
                                          ('tbal_cd', 'AM0B_FC_TBAL')],
                        known_differences= [
                            {'name': 'Left-padded, four-digit numeric code',
                             'types': ['tinyint', 'smallint', 'int', 'bigint', 'float', 'double', 'decimal'],
                             'transformation': "lpad(cast({input} AS bigint), 4, '0')"},
                            {'name': 'Null to *2',
                             'types': ['string'],
                             'transformation': "case when {input} is null then '*2' else {input} end"},
                            {'name': 'Julian date -> date',
                             'types': ['bigint'],
                             'transformation': "to_date(cast(unix_timestamp(cast({input} AS string), 'yyyyDDD') AS timestamp))"}
                        ])
    comparison.report()

Corresponding output::

    ****** Column Summary ******
    Number of columns in common with matching schemas: 3
    Number of columns in common with schema differences: 2
    Number of columns in base but not compare: 0
    Number of columns in compare but not base: 0
    
    ****** Schema Differences ******
    Base Column Name  Compare Column Name     Base Dtype     Compare Dtype
    ----------------  ----------------------  -------------  -------------
    open_dt           AM00_DATE_ACCOUNT_OPEN  date           bigint       
    tbal_cd           AM0B_FC_TBAL            string         double       
    
    ****** Row Summary ******
    Number of rows in common: 5
    Number of rows in base but not compare: 0
    Number of rows in compare but not base: 0
    Number of duplicate rows found in base: 0
    Number of duplicate rows found in compare: 0
    
    ****** Row Comparison ******
    Number of rows with some columns unequal: 5
    Number of rows with all columns equal: 0
    
    ****** Column Comparison ******
    Number of columns compared with unexpected differences in some values: 1
    Number of columns compared with all values equal but known differences found: 2
    Number of columns compared with all values completely equal: 0
    
    ****** Columns with Unequal Values ******
    Base Column Name  Compare Column Name     Base Dtype     Compare Dtype  # Matches  # Known Diffs  # Mismatches
    ----------------  -------------------     -------------  -------------  ---------  -------------  ------------
    clsd_reas_cd      AM00_STATC_CLOSED       string         string                 2              2             1
    open_dt           AM00_DATE_ACCOUNT_OPEN  date           bigint                 0              5             0
    tbal_cd           AM0B_FC_TBAL            string         double                 0              5             0