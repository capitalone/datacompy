Spark Usage
===========

*Under Construction*

Meanwhile, see the Readme "Spark Detail" section for a usage example and comments on ``SparkCompare``. You may also
want to checkout the :class:`datacompy.SparkCompare` API documentation, which is pretty well-documented, if I do say
so myself.

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