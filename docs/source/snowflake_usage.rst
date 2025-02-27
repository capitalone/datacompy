Snowpark/Snowflake Usage
========================

For ``SnowflakeCompare``

- ``on_index`` is not supported.
- Joining is done using ``EQUAL_NULL`` which is the equality test that is safe for null values.
- Compares ``snowflake.snowpark.DataFrame``, which can be provided as either raw Snowflake dataframes
  or as the names of full names of valid snowflake tables, which we will process into Snowpark dataframes.
- Note that if Snowflake tables are provided, that dataframe names will default to the full name of their
respective Snowflake tables. This can be overriden by setting the ``df1_name`` and ``df2_name`` arguments
when creating the Compare object.


SnowflakeCompare setup
----------------------

There are two ways to specify input dataframes for ``SnowflakeCompare``

Provide Snowpark dataframes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from snowflake.snowpark import Session
    from snowflake.snowpark import Row
    import datetime
    import datacompy.snowflake as sp

    connection_parameters = {
        ...
    }
    session = Session.builder.configs(connection_parameters).create()

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
        date_fld=datetime.date(2017, 1, 1)),
    ]

    data2 = [
        Row(acct_id=10000001234, dollar_amt=123.4, name='George Michael Bluth', float_fld=14530.155),
        Row(acct_id=10000001235, dollar_amt=0.45, name='Michael Bluth', float_fld=None),
        Row(acct_id=None, dollar_amt=1345.0, name='George Bluth', float_fld=1.0),
        Row(acct_id=10000001237, dollar_amt=123456.0, name='Robert Loblaw', float_fld=345.12),
        Row(acct_id=10000001238, dollar_amt=1.05, name='Loose Seal Bluth', float_fld=111.0),
    ]

    df_1 = session.createDataFrame(data1)
    df_2 = session.createDataFrame(data2)

    compare = sp.SnowflakeCompare(
        session,
        df_1,
        df_2,
        #df1_name='original', # optional param for naming df1
        #df2_name='new' # optional param for naming df2
        join_columns=['acct_id'],
        rel_tol=1e-03,
        abs_tol=1e-04,
    )
    compare.matches(ignore_extra_columns=False)

    # This method prints out a human-readable report summarizing and sampling differences
    print(compare.report())


Provide the full name (``db.schema.table_name``) of valid Snowflake tables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Given the dataframes from the prior examples...

.. code-block:: python

    df_1.write.mode("overwrite").save_as_table("toy_table_1")
    df_2.write.mode("overwrite").save_as_table("toy_table_2")

    compare = sp.SnowflakeCompare(
        session,
        f"{db}.{schema}.toy_table_1",
        f"{db}.{schema}.toy_table_2",
        #df1_name='original', # optional param for naming df1
        #df2_name='new' # optional param for naming df2
        join_columns=['acct_id'],
        rel_tol=1e-03,
        abs_tol=1e-04,
    )
    compare.matches(ignore_extra_columns=False)

    # This method prints out a human-readable report summarizing and sampling differences
    print(compare.report())

Reports
-------

A report is generated by calling ``report()``, which returns a string.
Here is a sample report generated by ``datacompy`` for the two tables above,
joined on ``acct_id`` (Note: the names for your dataframes are extracted from
the name of the provided Snowflake table. If you chose to directly use Snowpark
dataframes, then the names will default to ``DF1`` and ``DF2``.)::

    DataComPy Comparison
    --------------------

    DataFrame Summary
    -----------------

    DataFrame  Columns  Rows
    0       DF1        5     5
    1       DF2        4     5

    Column Summary
    --------------

    Number of columns in common: 4
    Number of columns in DF1 but not in DF2: 1
    Number of columns in DF2 but not in DF1: 0

    Row Summary
    -----------

    Matched on: ACCT_ID
    Any duplicates on match values: No
    Absolute Tolerance: 0
    Relative Tolerance: 0
    Number of rows in common: 4
    Number of rows in DF1 but not in DF2: 1
    Number of rows in DF2 but not in DF1: 1

    Number of rows with some compared columns unequal: 4
    Number of rows with all compared columns equal: 0

    Column Comparison
    -----------------

    Number of columns compared with some values unequal: 3
    Number of columns compared with all values equal: 1
    Total number of values which compare unequal: 6

    Columns with Unequal Values or Types
    ------------------------------------

        Column         DF1 dtype         DF2 dtype  # Unequal  Max Diff  # Null Diff
    0  DOLLAR_AMT            double            double          1    0.0500            0
    2   FLOAT_FLD            double            double          3    0.0005            2
    1        NAME  string(16777216)  string(16777216)          2       NaN            0

    Sample Rows with Unequal Values
    -------------------------------

        ACCT_ID  DOLLAR_AMT (DF1)  DOLLAR_AMT (DF2)
    0  10000001234            123.45             123.4

        ACCT_ID      NAME (DF1)            NAME (DF2)
    0  10000001234  George Maharis  George Michael Bluth
    1  10000001237      Bob Loblaw         Robert Loblaw

        ACCT_ID  FLOAT_FLD (DF1)  FLOAT_FLD (DF2)
    0  10000001234       14530.1555        14530.155
    1  10000001235           1.0000              NaN
    2  10000001236              NaN            1.000

    Sample Rows Only in DF1 (First 10 Columns)
    ------------------------------------------

    ACCT_ID_DF1  DOLLAR_AMT_DF1       NAME_DF1  FLOAT_FLD_DF1 DATE_FLD_DF1
    0  10000001239            1.05  Lucille Bluth            NaN   2017-01-01

    Sample Rows Only in DF2 (First 10 Columns)
    ------------------------------------------

    ACCT_ID_DF2  DOLLAR_AMT_DF2          NAME_DF2  FLOAT_FLD_DF2
    0  10000001238            1.05  Loose Seal Bluth          111.0


Convenience Methods
-------------------

There are a few convenience methods and attributes available after the comparison has been run:

.. code-block:: python

    compare.intersect_rows[['name_df1', 'name_df2', 'name_match']].show()
    # --------------------------------------------------------
    # |"NAME_DF1"      |"NAME_DF2"            |"NAME_MATCH"  |
    # --------------------------------------------------------
    # |George Maharis  |George Michael Bluth  |False         |
    # |Michael Bluth   |Michael Bluth         |True          |
    # |George Bluth    |George Bluth          |True          |
    # |Bob Loblaw      |Robert Loblaw         |False         |
    # --------------------------------------------------------

    compare.df1_unq_rows.show()
    # ---------------------------------------------------------------------------------------
    # |"ACCT_ID_DF1"  |"DOLLAR_AMT_DF1"  |"NAME_DF1"     |"FLOAT_FLD_DF1"  |"DATE_FLD_DF1"  |
    # ---------------------------------------------------------------------------------------
    # |10000001239    |1.05              |Lucille Bluth  |NULL             |2017-01-01      |
    # ---------------------------------------------------------------------------------------

    compare.df2_unq_rows.show()
    # -------------------------------------------------------------------------
    # |"ACCT_ID_DF2"  |"DOLLAR_AMT_DF2"  |"NAME_DF2"        |"FLOAT_FLD_DF2"  |
    # -------------------------------------------------------------------------
    # |10000001238    |1.05              |Loose Seal Bluth  |111.0            |
    # -------------------------------------------------------------------------

    print(compare.intersect_columns())
    # OrderedSet(['acct_id', 'dollar_amt', 'name', 'float_fld'])

    print(compare.df1_unq_columns())
    # OrderedSet(['date_fld'])

    print(compare.df2_unq_columns())
    # OrderedSet()


Duplicate rows
--------------

Datacompy will try to handle rows that are duplicate in the join columns.  It does this behind the
scenes by generating a unique ID within each unique group of the join columns.  For example, if you
have two dataframes you're trying to join on acct_id:

=========== ================
acct_id     name
=========== ================
1           George Maharis
1           Michael Bluth
2           George Bluth
=========== ================

=========== ================
acct_id     name
=========== ================
1           George Maharis
1           Michael Bluth
1           Tony Wonder
2           George Bluth
=========== ================

Datacompy will generate a unique temporary ID for joining:

=========== ================ ========
acct_id     name             temp_id
=========== ================ ========
1           George Maharis   0
1           Michael Bluth    1
2           George Bluth     0
=========== ================ ========

=========== ================ ========
acct_id     name             temp_id
=========== ================ ========
1           George Maharis   0
1           Michael Bluth    1
1           Tony Wonder      2
2           George Bluth     0
=========== ================ ========

And then merge the two dataframes on a combination of the join_columns you specified and the temporary
ID, before dropping the temp_id again.  So the first two rows in the first dataframe will match the
first two rows in the second dataframe, and the third row in the second dataframe will be recognized
as uniquely in the second.

Additional considerations
-------------------------

- It is strongly recommended against joining on float columns or any column with floating point precision.
  Columns joining tables are compared on the basis of an exact comparison, therefore if the values
  comparing your float columns are not exact, you will likely get unexpected results.
- Case-sensitive columns are only partially supported. We essentially treat case-sensitive columns as
  if they are case-insensitive. Therefore you may use case-sensitive columns as long as you don't have several
  columns with the same name differentiated only be case sensitivity.
