#
# Copyright 2024 Capital One Services, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import io
import logging
import re
from decimal import Decimal

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    DateType,
    DecimalType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

import datacompy
from datacompy import SparkCompare
from datacompy.spark import _is_comparable

# Turn off py4j debug messages for all tests in this module
logging.getLogger("py4j").setLevel(logging.INFO)

CACHE_INTERMEDIATES = True


# Declare fixtures
# (if we need to use these in other modules, move to conftest.py)
@pytest.fixture(scope="module", name="spark")
def spark_fixture():
    spark = (
        SparkSession.builder.master("local[2]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .appName("pytest")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="module", name="base_df1")
def base_df1_fixture(spark):
    mock_data = [
        Row(
            acct=10000001234,
            dollar_amt=123,
            name="George Maharis",
            float_fld=14530.1555,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001235,
            dollar_amt=0,
            name="Michael Bluth",
            float_fld=1.0,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001236,
            dollar_amt=1345,
            name="George Bluth",
            float_fld=None,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001237,
            dollar_amt=123456,
            name="Bob Loblaw",
            float_fld=345.12,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001239,
            dollar_amt=1,
            name="Lucille Bluth",
            float_fld=None,
            date_fld=datetime.date(2017, 1, 1),
        ),
    ]

    return spark.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="base_df2")
def base_df2_fixture(spark):
    mock_data = [
        Row(
            acct=10000001234,
            dollar_amt=123,
            super_duper_big_long_name="George Maharis",
            float_fld=14530.1555,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001235,
            dollar_amt=0,
            super_duper_big_long_name="Michael Bluth",
            float_fld=1.0,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001236,
            dollar_amt=1345,
            super_duper_big_long_name="George Bluth",
            float_fld=None,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001237,
            dollar_amt=123456,
            super_duper_big_long_name="Bob Loblaw",
            float_fld=345.12,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001239,
            dollar_amt=1,
            super_duper_big_long_name="Lucille Bluth",
            float_fld=None,
            date_fld=datetime.date(2017, 1, 1),
        ),
    ]

    return spark.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="compare_df1")
def compare_df1_fixture(spark):
    mock_data2 = [
        Row(
            acct=10000001234,
            dollar_amt=123.4,
            name="George Michael Bluth",
            float_fld=14530.155,
            accnt_purge=False,
        ),
        Row(
            acct=10000001235,
            dollar_amt=0.45,
            name="Michael Bluth",
            float_fld=None,
            accnt_purge=False,
        ),
        Row(
            acct=10000001236,
            dollar_amt=1345.0,
            name="George Bluth",
            float_fld=1.0,
            accnt_purge=False,
        ),
        Row(
            acct=10000001237,
            dollar_amt=123456.0,
            name="Bob Loblaw",
            float_fld=345.12,
            accnt_purge=False,
        ),
        Row(
            acct=10000001238,
            dollar_amt=1.05,
            name="Loose Seal Bluth",
            float_fld=111.0,
            accnt_purge=True,
        ),
        Row(
            acct=10000001238,
            dollar_amt=1.05,
            name="Loose Seal Bluth",
            float_fld=111.0,
            accnt_purge=True,
        ),
    ]

    return spark.createDataFrame(mock_data2)


@pytest.fixture(scope="module", name="compare_df2")
def compare_df2_fixture(spark):
    mock_data = [
        Row(
            acct=10000001234,
            dollar_amt=123,
            name="George Maharis",
            float_fld=14530.1555,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001235,
            dollar_amt=0,
            name="Michael Bluth",
            float_fld=1.0,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001236,
            dollar_amt=1345,
            name="George Bluth",
            float_fld=None,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001237,
            dollar_amt=123456,
            name="Bob Loblaw",
            float_fld=345.12,
            date_fld=datetime.date(2017, 1, 1),
        ),
        Row(
            acct=10000001239,
            dollar_amt=1,
            name="Lucille Bluth",
            float_fld=None,
            date_fld=datetime.date(2017, 1, 1),
        ),
    ]

    return spark.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="compare_df3")
def compare_df3_fixture(spark):
    mock_data2 = [
        Row(
            account_identifier=10000001234,
            dollar_amount=123.4,
            name="George Michael Bluth",
            float_field=14530.155,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),
        Row(
            account_identifier=10000001235,
            dollar_amount=0.45,
            name="Michael Bluth",
            float_field=1.0,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),
        Row(
            account_identifier=10000001236,
            dollar_amount=1345.0,
            name="George Bluth",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),
        Row(
            account_identifier=10000001237,
            dollar_amount=123456.0,
            name="Bob Loblaw",
            float_field=345.12,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),
        Row(
            account_identifier=10000001239,
            dollar_amount=1.05,
            name="Lucille Bluth",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),
    ]

    return spark.createDataFrame(mock_data2)


@pytest.fixture(scope="module", name="base_tol")
def base_tol_fixture(spark):
    tol_data1 = [
        Row(
            account_identifier=10000001234,
            dollar_amount=123.4,
            name="Franklin Delano Bluth",
            float_field=14530.155,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),
        Row(
            account_identifier=10000001235,
            dollar_amount=500.0,
            name="Surely Funke",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),
        Row(
            account_identifier=10000001236,
            dollar_amount=-1100.0,
            name="Nichael Bluth",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),
        Row(
            account_identifier=10000001237,
            dollar_amount=0.45,
            name="Mr. F",
            float_field=1.0,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),
        Row(
            account_identifier=10000001238,
            dollar_amount=1345.0,
            name="Steve Holt!",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),
        Row(
            account_identifier=10000001239,
            dollar_amount=123456.0,
            name="Blue Man Group",
            float_field=345.12,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),
        Row(
            account_identifier=10000001240,
            dollar_amount=1.1,
            name="Her?",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),
        Row(
            account_identifier=10000001241,
            dollar_amount=0.0,
            name="Mrs. Featherbottom",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),
        Row(
            account_identifier=10000001242,
            dollar_amount=0.0,
            name="Ice",
            float_field=345.12,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),
        Row(
            account_identifier=10000001243,
            dollar_amount=-10.0,
            name="Frank Wrench",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),
        Row(
            account_identifier=10000001244,
            dollar_amount=None,
            name="Lucille 2",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),
        Row(
            account_identifier=10000001245,
            dollar_amount=0.009999,
            name="Gene Parmesan",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),
        Row(
            account_identifier=10000001246,
            dollar_amount=None,
            name="Motherboy",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),
    ]

    return spark.createDataFrame(tol_data1)


@pytest.fixture(scope="module", name="compare_abs_tol")
def compare_tol2_fixture(spark):
    tol_data2 = [
        Row(
            account_identifier=10000001234,
            dollar_amount=123.4,
            name="Franklin Delano Bluth",
            float_field=14530.155,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # full match
        Row(
            account_identifier=10000001235,
            dollar_amount=500.01,
            name="Surely Funke",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # off by 0.01
        Row(
            account_identifier=10000001236,
            dollar_amount=-1100.01,
            name="Nichael Bluth",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # off by -0.01
        Row(
            account_identifier=10000001237,
            dollar_amount=0.46000000001,
            name="Mr. F",
            float_field=1.0,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # off by 0.01000000001
        Row(
            account_identifier=10000001238,
            dollar_amount=1344.8999999999,
            name="Steve Holt!",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # off by -0.01000000001
        Row(
            account_identifier=10000001239,
            dollar_amount=123456.0099999999,
            name="Blue Man Group",
            float_field=345.12,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # off by 0.00999999999
        Row(
            account_identifier=10000001240,
            dollar_amount=1.090000001,
            name="Her?",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # off by -0.00999999999
        Row(
            account_identifier=10000001241,
            dollar_amount=0.0,
            name="Mrs. Featherbottom",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # both zero
        Row(
            account_identifier=10000001242,
            dollar_amount=1.0,
            name="Ice",
            float_field=345.12,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # base 0, compare 1
        Row(
            account_identifier=10000001243,
            dollar_amount=0.0,
            name="Frank Wrench",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # base -10, compare 0
        Row(
            account_identifier=10000001244,
            dollar_amount=-1.0,
            name="Lucille 2",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # base NULL, compare -1
        Row(
            account_identifier=10000001245,
            dollar_amount=None,
            name="Gene Parmesan",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # base 0.009999, compare NULL
        Row(
            account_identifier=10000001246,
            dollar_amount=None,
            name="Motherboy",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # both NULL
    ]

    return spark.createDataFrame(tol_data2)


@pytest.fixture(scope="module", name="compare_rel_tol")
def compare_tol3_fixture(spark):
    tol_data3 = [
        Row(
            account_identifier=10000001234,
            dollar_amount=123.4,
            name="Franklin Delano Bluth",
            float_field=14530.155,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # full match   #MATCH
        Row(
            account_identifier=10000001235,
            dollar_amount=550.0,
            name="Surely Funke",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # off by 10%   #MATCH
        Row(
            account_identifier=10000001236,
            dollar_amount=-1000.0,
            name="Nichael Bluth",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # off by -10%    #MATCH
        Row(
            account_identifier=10000001237,
            dollar_amount=0.49501,
            name="Mr. F",
            float_field=1.0,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # off by greater than 10%
        Row(
            account_identifier=10000001238,
            dollar_amount=1210.001,
            name="Steve Holt!",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # off by greater than -10%
        Row(
            account_identifier=10000001239,
            dollar_amount=135801.59999,
            name="Blue Man Group",
            float_field=345.12,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # off by just under 10%   #MATCH
        Row(
            account_identifier=10000001240,
            dollar_amount=1.000001,
            name="Her?",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # off by just under -10%   #MATCH
        Row(
            account_identifier=10000001241,
            dollar_amount=0.0,
            name="Mrs. Featherbottom",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # both zero   #MATCH
        Row(
            account_identifier=10000001242,
            dollar_amount=1.0,
            name="Ice",
            float_field=345.12,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # base 0, compare 1
        Row(
            account_identifier=10000001243,
            dollar_amount=0.0,
            name="Frank Wrench",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # base -10, compare 0
        Row(
            account_identifier=10000001244,
            dollar_amount=-1.0,
            name="Lucille 2",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # base NULL, compare -1
        Row(
            account_identifier=10000001245,
            dollar_amount=None,
            name="Gene Parmesan",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # base 0.009999, compare NULL
        Row(
            account_identifier=10000001246,
            dollar_amount=None,
            name="Motherboy",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # both NULL  #MATCH
    ]

    return spark.createDataFrame(tol_data3)


@pytest.fixture(scope="module", name="compare_both_tol")
def compare_tol4_fixture(spark):
    tol_data4 = [
        Row(
            account_identifier=10000001234,
            dollar_amount=123.4,
            name="Franklin Delano Bluth",
            float_field=14530.155,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # full match
        Row(
            account_identifier=10000001235,
            dollar_amount=550.01,
            name="Surely Funke",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # off by 10% and +0.01
        Row(
            account_identifier=10000001236,
            dollar_amount=-1000.01,
            name="Nichael Bluth",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # off by -10% and -0.01
        Row(
            account_identifier=10000001237,
            dollar_amount=0.505000000001,
            name="Mr. F",
            float_field=1.0,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # off by greater than 10% and +0.01
        Row(
            account_identifier=10000001238,
            dollar_amount=1209.98999,
            name="Steve Holt!",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # off by greater than -10% and -0.01
        Row(
            account_identifier=10000001239,
            dollar_amount=135801.609999,
            name="Blue Man Group",
            float_field=345.12,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # off by just under 10% and just under +0.01
        Row(
            account_identifier=10000001240,
            dollar_amount=0.99000001,
            name="Her?",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # off by just under -10% and just under -0.01
        Row(
            account_identifier=10000001241,
            dollar_amount=0.0,
            name="Mrs. Featherbottom",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # both zero
        Row(
            account_identifier=10000001242,
            dollar_amount=1.0,
            name="Ice",
            float_field=345.12,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=False,
        ),  # base 0, compare 1
        Row(
            account_identifier=10000001243,
            dollar_amount=0.0,
            name="Frank Wrench",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # base -10, compare 0
        Row(
            account_identifier=10000001244,
            dollar_amount=-1.0,
            name="Lucille 2",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # base NULL, compare -1
        Row(
            account_identifier=10000001245,
            dollar_amount=None,
            name="Gene Parmesan",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # base 0.009999, compare NULL
        Row(
            account_identifier=10000001246,
            dollar_amount=None,
            name="Motherboy",
            float_field=None,
            date_field=datetime.date(2017, 1, 1),
            accnt_purge=True,
        ),  # both NULL
    ]

    return spark.createDataFrame(tol_data4)


@pytest.fixture(scope="module", name="base_td")
def base_td_fixture(spark):
    mock_data = [
        Row(
            acct=10000001234,
            acct_seq=0,
            stat_cd="*2",
            open_dt=datetime.date(2017, 5, 1),
            cd="0001",
        ),
        Row(
            acct=10000001235,
            acct_seq=0,
            stat_cd="V1",
            open_dt=datetime.date(2017, 5, 2),
            cd="0002",
        ),
        Row(
            acct=10000001236,
            acct_seq=0,
            stat_cd="V2",
            open_dt=datetime.date(2017, 5, 3),
            cd="0003",
        ),
        Row(
            acct=10000001237,
            acct_seq=0,
            stat_cd="*2",
            open_dt=datetime.date(2017, 5, 4),
            cd="0004",
        ),
        Row(
            acct=10000001238,
            acct_seq=0,
            stat_cd="*2",
            open_dt=datetime.date(2017, 5, 5),
            cd="0005",
        ),
    ]

    return spark.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="compare_source")
def compare_source_fixture(spark):
    mock_data = [
        Row(
            ACCOUNT_IDENTIFIER=10000001234,
            SEQ_NUMBER=0,
            STATC=None,
            ACCOUNT_OPEN=2017121,
            CODE=1.0,
        ),
        Row(
            ACCOUNT_IDENTIFIER=10000001235,
            SEQ_NUMBER=0,
            STATC="V1",
            ACCOUNT_OPEN=2017122,
            CODE=2.0,
        ),
        Row(
            ACCOUNT_IDENTIFIER=10000001236,
            SEQ_NUMBER=0,
            STATC="V2",
            ACCOUNT_OPEN=2017123,
            CODE=3.0,
        ),
        Row(
            ACCOUNT_IDENTIFIER=10000001237,
            SEQ_NUMBER=0,
            STATC="V3",
            ACCOUNT_OPEN=2017124,
            CODE=4.0,
        ),
        Row(
            ACCOUNT_IDENTIFIER=10000001238,
            SEQ_NUMBER=0,
            STATC=None,
            ACCOUNT_OPEN=2017125,
            CODE=5.0,
        ),
    ]

    return spark.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="base_decimal")
def base_decimal_fixture(spark):
    mock_data = [
        Row(acct=10000001234, dollar_amt=Decimal(123.4)),
        Row(acct=10000001235, dollar_amt=Decimal(0.45)),
    ]

    return spark.createDataFrame(
        mock_data,
        schema=StructType(
            [
                StructField("acct", LongType(), True),
                StructField("dollar_amt", DecimalType(8, 2), True),
            ]
        ),
    )


@pytest.fixture(scope="module", name="compare_decimal")
def compare_decimal_fixture(spark):
    mock_data = [
        Row(acct=10000001234, dollar_amt=123.4),
        Row(acct=10000001235, dollar_amt=0.456),
    ]

    return spark.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="comparison_abs_tol")
def comparison_abs_tol_fixture(base_tol, compare_abs_tol, spark):
    return SparkCompare(
        spark,
        base_tol,
        compare_abs_tol,
        join_columns=["account_identifier"],
        abs_tol=0.01,
    )


@pytest.fixture(scope="module", name="comparison_rel_tol")
def comparison_rel_tol_fixture(base_tol, compare_rel_tol, spark):
    return SparkCompare(
        spark,
        base_tol,
        compare_rel_tol,
        join_columns=["account_identifier"],
        rel_tol=0.1,
    )


@pytest.fixture(scope="module", name="comparison_both_tol")
def comparison_both_tol_fixture(base_tol, compare_both_tol, spark):
    return SparkCompare(
        spark,
        base_tol,
        compare_both_tol,
        join_columns=["account_identifier"],
        rel_tol=0.1,
        abs_tol=0.01,
    )


@pytest.fixture(scope="module", name="comparison_neg_tol")
def comparison_neg_tol_fixture(base_tol, compare_both_tol, spark):
    return SparkCompare(
        spark,
        base_tol,
        compare_both_tol,
        join_columns=["account_identifier"],
        rel_tol=-0.2,
        abs_tol=0.01,
    )


@pytest.fixture(scope="module", name="show_all_columns_and_match_rate")
def show_all_columns_and_match_rate_fixture(base_tol, compare_both_tol, spark):
    return SparkCompare(
        spark,
        base_tol,
        compare_both_tol,
        join_columns=["account_identifier"],
        show_all_columns=True,
        match_rates=True,
    )


@pytest.fixture(scope="module", name="comparison_kd1")
def comparison_known_diffs1(base_td, compare_source, spark):
    return SparkCompare(
        spark,
        base_td,
        compare_source,
        join_columns=[("acct", "ACCOUNT_IDENTIFIER"), ("acct_seq", "SEQ_NUMBER")],
        column_mapping=[
            ("stat_cd", "STATC"),
            ("open_dt", "ACCOUNT_OPEN"),
            ("cd", "CODE"),
        ],
        known_differences=[
            {
                "name": "Left-padded, four-digit numeric code",
                "types": datacompy.NUMERIC_SPARK_TYPES,
                "transformation": "lpad(cast({input} AS bigint), 4, '0')",
            },
            {
                "name": "Null to *2",
                "types": ["string"],
                "transformation": "case when {input} is null then '*2' else {input} end",
            },
            {
                "name": "Julian date -> date",
                "types": ["bigint"],
                "transformation": "to_date(cast(unix_timestamp(cast({input} AS string), 'yyyyDDD') AS timestamp))",
            },
        ],
    )


@pytest.fixture(scope="module", name="comparison_kd2")
def comparison_known_diffs2(base_td, compare_source, spark):
    return SparkCompare(
        spark,
        base_td,
        compare_source,
        join_columns=[("acct", "ACCOUNT_IDENTIFIER"), ("acct_seq", "SEQ_NUMBER")],
        column_mapping=[
            ("stat_cd", "STATC"),
            ("open_dt", "ACCOUNT_OPEN"),
            ("cd", "CODE"),
        ],
        known_differences=[
            {
                "name": "Left-padded, four-digit numeric code",
                "types": datacompy.NUMERIC_SPARK_TYPES,
                "transformation": "lpad(cast({input} AS bigint), 4, '0')",
            },
            {
                "name": "Null to *2",
                "types": ["string"],
                "transformation": "case when {input} is null then '*2' else {input} end",
            },
        ],
    )


@pytest.fixture(scope="module", name="comparison1")
def comparison1_fixture(base_df1, compare_df1, spark):
    return SparkCompare(
        spark,
        base_df1,
        compare_df1,
        join_columns=["acct"],
        cache_intermediates=CACHE_INTERMEDIATES,
    )


@pytest.fixture(scope="module", name="comparison2")
def comparison2_fixture(base_df1, compare_df2, spark):
    return SparkCompare(spark, base_df1, compare_df2, join_columns=["acct"])


@pytest.fixture(scope="module", name="comparison3")
def comparison3_fixture(base_df1, compare_df3, spark):
    return SparkCompare(
        spark,
        base_df1,
        compare_df3,
        join_columns=[("acct", "account_identifier")],
        column_mapping=[
            ("dollar_amt", "dollar_amount"),
            ("float_fld", "float_field"),
            ("date_fld", "date_field"),
        ],
        cache_intermediates=CACHE_INTERMEDIATES,
    )


@pytest.fixture(scope="module", name="comparison4")
def comparison4_fixture(base_df2, compare_df1, spark):
    return SparkCompare(
        spark,
        base_df2,
        compare_df1,
        join_columns=["acct"],
        column_mapping=[("super_duper_big_long_name", "name")],
    )


@pytest.fixture(scope="module", name="comparison_decimal")
def comparison_decimal_fixture(base_decimal, compare_decimal, spark):
    return SparkCompare(spark, base_decimal, compare_decimal, join_columns=["acct"])


def test_absolute_tolerances(comparison_abs_tol):
    stdout = io.StringIO()

    comparison_abs_tol.report(file=stdout)
    stdout.seek(0)
    assert "****** Row Comparison ******" in stdout.getvalue()
    assert "Number of rows with some columns unequal: 6" in stdout.getvalue()
    assert "Number of rows with all columns equal: 7" in stdout.getvalue()
    assert "Number of columns compared with some values unequal: 1" in stdout.getvalue()
    assert "Number of columns compared with all values equal: 4" in stdout.getvalue()


def test_relative_tolerances(comparison_rel_tol):
    stdout = io.StringIO()

    comparison_rel_tol.report(file=stdout)
    stdout.seek(0)
    assert "****** Row Comparison ******" in stdout.getvalue()
    assert "Number of rows with some columns unequal: 6" in stdout.getvalue()
    assert "Number of rows with all columns equal: 7" in stdout.getvalue()
    assert "Number of columns compared with some values unequal: 1" in stdout.getvalue()
    assert "Number of columns compared with all values equal: 4" in stdout.getvalue()


def test_both_tolerances(comparison_both_tol):
    stdout = io.StringIO()

    comparison_both_tol.report(file=stdout)
    stdout.seek(0)
    assert "****** Row Comparison ******" in stdout.getvalue()
    assert "Number of rows with some columns unequal: 6" in stdout.getvalue()
    assert "Number of rows with all columns equal: 7" in stdout.getvalue()
    assert "Number of columns compared with some values unequal: 1" in stdout.getvalue()
    assert "Number of columns compared with all values equal: 4" in stdout.getvalue()


def test_negative_tolerances(spark, base_tol, compare_both_tol):
    with pytest.raises(ValueError, match="Please enter positive valued tolerances"):
        comp = SparkCompare(
            spark,
            base_tol,
            compare_both_tol,
            join_columns=["account_identifier"],
            rel_tol=-0.2,
            abs_tol=0.01,
        )
        comp.report()
        pass


def test_show_all_columns_and_match_rate(show_all_columns_and_match_rate):
    stdout = io.StringIO()

    show_all_columns_and_match_rate.report(file=stdout)

    assert "****** Columns with Equal/Unequal Values ******" in stdout.getvalue()
    assert (
        "accnt_purge       accnt_purge          boolean        boolean               13             0     100.00000"
        in stdout.getvalue()
    )
    assert (
        "date_field        date_field           date           date                  13             0     100.00000"
        in stdout.getvalue()
    )
    assert (
        "dollar_amount     dollar_amount        double         double                 3            10      23.07692"
        in stdout.getvalue()
    )
    assert (
        "float_field       float_field          double         double                13             0     100.00000"
        in stdout.getvalue()
    )
    assert (
        "name              name                 string         string                13             0     100.00000"
        in stdout.getvalue()
    )


def test_decimal_comparisons():
    true_decimals = ["decimal", "decimal()", "decimal(20, 10)"]
    assert all(v in datacompy.NUMERIC_SPARK_TYPES for v in true_decimals)


def test_decimal_comparator_acts_like_string():
    acc = False
    for t in datacompy.NUMERIC_SPARK_TYPES:
        acc = acc or (len(t) > 2 and t[0:3] == "dec")
    assert acc


def test_decimals_and_doubles_are_comparable():
    assert _is_comparable("double", "decimal(10, 2)")


def test_report_outputs_the_column_summary(comparison1):
    stdout = io.StringIO()

    comparison1.report(file=stdout)

    assert "****** Column Summary ******" in stdout.getvalue()
    assert "Number of columns in common with matching schemas: 3" in stdout.getvalue()
    assert "Number of columns in common with schema differences: 1" in stdout.getvalue()
    assert "Number of columns in base but not compare: 1" in stdout.getvalue()
    assert "Number of columns in compare but not base: 1" in stdout.getvalue()


def test_report_outputs_the_column_summary_for_identical_schemas(comparison2):
    stdout = io.StringIO()

    comparison2.report(file=stdout)

    assert "****** Column Summary ******" in stdout.getvalue()
    assert "Number of columns in common with matching schemas: 5" in stdout.getvalue()
    assert "Number of columns in common with schema differences: 0" in stdout.getvalue()
    assert "Number of columns in base but not compare: 0" in stdout.getvalue()
    assert "Number of columns in compare but not base: 0" in stdout.getvalue()


def test_report_outputs_the_column_summary_for_differently_named_columns(comparison3):
    stdout = io.StringIO()

    comparison3.report(file=stdout)

    assert "****** Column Summary ******" in stdout.getvalue()
    assert "Number of columns in common with matching schemas: 4" in stdout.getvalue()
    assert "Number of columns in common with schema differences: 1" in stdout.getvalue()
    assert "Number of columns in base but not compare: 0" in stdout.getvalue()
    assert "Number of columns in compare but not base: 1" in stdout.getvalue()


def test_report_outputs_the_row_summary(comparison1):
    stdout = io.StringIO()

    comparison1.report(file=stdout)

    assert "****** Row Summary ******" in stdout.getvalue()
    assert "Number of rows in common: 4" in stdout.getvalue()
    assert "Number of rows in base but not compare: 1" in stdout.getvalue()
    assert "Number of rows in compare but not base: 1" in stdout.getvalue()
    assert "Number of duplicate rows found in base: 0" in stdout.getvalue()
    assert "Number of duplicate rows found in compare: 1" in stdout.getvalue()


def test_report_outputs_the_row_equality_comparison(comparison1):
    stdout = io.StringIO()

    comparison1.report(file=stdout)

    assert "****** Row Comparison ******" in stdout.getvalue()
    assert "Number of rows with some columns unequal: 3" in stdout.getvalue()
    assert "Number of rows with all columns equal: 1" in stdout.getvalue()


def test_report_outputs_the_row_summary_for_differently_named_columns(comparison3):
    stdout = io.StringIO()

    comparison3.report(file=stdout)

    assert "****** Row Summary ******" in stdout.getvalue()
    assert "Number of rows in common: 5" in stdout.getvalue()
    assert "Number of rows in base but not compare: 0" in stdout.getvalue()
    assert "Number of rows in compare but not base: 0" in stdout.getvalue()
    assert "Number of duplicate rows found in base: 0" in stdout.getvalue()
    assert "Number of duplicate rows found in compare: 0" in stdout.getvalue()


def test_report_outputs_the_row_equality_comparison_for_differently_named_columns(
    comparison3,
):
    stdout = io.StringIO()

    comparison3.report(file=stdout)

    assert "****** Row Comparison ******" in stdout.getvalue()
    assert "Number of rows with some columns unequal: 3" in stdout.getvalue()
    assert "Number of rows with all columns equal: 2" in stdout.getvalue()


def test_report_outputs_column_detail_for_columns_in_only_one_dataframe(comparison1):
    stdout = io.StringIO()

    comparison1.report(file=stdout)
    comparison1.report()
    assert "****** Columns In Base Only ******" in stdout.getvalue()
    r2 = r"""Column\s*Name \s* Dtype \n -+ \s+ -+ \ndate_fld \s+ date"""
    assert re.search(r2, str(stdout.getvalue()), re.X) is not None


def test_report_outputs_column_detail_for_columns_in_only_compare_dataframe(
    comparison1,
):
    stdout = io.StringIO()

    comparison1.report(file=stdout)
    comparison1.report()
    assert "****** Columns In Compare Only ******" in stdout.getvalue()
    r2 = r"""Column\s*Name \s* Dtype \n -+ \s+ -+ \n accnt_purge \s+  boolean"""
    assert re.search(r2, str(stdout.getvalue()), re.X) is not None


def test_report_outputs_schema_difference_details(comparison1):
    stdout = io.StringIO()

    comparison1.report(file=stdout)

    assert "****** Schema Differences ******" in stdout.getvalue()
    assert re.search(
        r"""Base\sColumn\sName \s+ Compare\sColumn\sName \s+ Base\sDtype \s+ Compare\sDtype \n
            -+ \s+ -+ \s+ -+ \s+ -+ \n
            dollar_amt \s+ dollar_amt \s+ bigint \s+ double""",
        stdout.getvalue(),
        re.X,
    )


def test_report_outputs_schema_difference_details_for_differently_named_columns(
    comparison3,
):
    stdout = io.StringIO()

    comparison3.report(file=stdout)

    assert "****** Schema Differences ******" in stdout.getvalue()
    assert re.search(
        r"""Base\sColumn\sName \s+ Compare\sColumn\sName \s+ Base\sDtype \s+ Compare\sDtype \n
            -+ \s+ -+ \s+ -+ \s+ -+ \n
            dollar_amt \s+ dollar_amount \s+ bigint \s+ double""",
        stdout.getvalue(),
        re.X,
    )


def test_column_comparison_outputs_number_of_columns_with_differences(comparison1):
    stdout = io.StringIO()

    comparison1.report(file=stdout)

    assert "****** Column Comparison ******" in stdout.getvalue()
    assert "Number of columns compared with some values unequal: 3" in stdout.getvalue()
    assert "Number of columns compared with all values equal: 0" in stdout.getvalue()


def test_column_comparison_outputs_all_columns_equal_for_identical_dataframes(
    comparison2,
):
    stdout = io.StringIO()

    comparison2.report(file=stdout)

    assert "****** Column Comparison ******" in stdout.getvalue()
    assert "Number of columns compared with some values unequal: 0" in stdout.getvalue()
    assert "Number of columns compared with all values equal: 4" in stdout.getvalue()


def test_column_comparison_outputs_number_of_columns_with_differences_for_differently_named_columns(
    comparison3,
):
    stdout = io.StringIO()

    comparison3.report(file=stdout)

    assert "****** Column Comparison ******" in stdout.getvalue()
    assert "Number of columns compared with some values unequal: 3" in stdout.getvalue()
    assert "Number of columns compared with all values equal: 1" in stdout.getvalue()


def test_column_comparison_outputs_number_of_columns_with_differences_for_known_diffs(
    comparison_kd1,
):
    stdout = io.StringIO()

    comparison_kd1.report(file=stdout)

    assert "****** Column Comparison ******" in stdout.getvalue()
    assert (
        "Number of columns compared with unexpected differences in some values: 1"
        in stdout.getvalue()
    )
    assert (
        "Number of columns compared with all values equal but known differences found: 2"
        in stdout.getvalue()
    )
    assert (
        "Number of columns compared with all values completely equal: 0"
        in stdout.getvalue()
    )


def test_column_comparison_outputs_number_of_columns_with_differences_for_custom_known_diffs(
    comparison_kd2,
):
    stdout = io.StringIO()

    comparison_kd2.report(file=stdout)

    assert "****** Column Comparison ******" in stdout.getvalue()
    assert (
        "Number of columns compared with unexpected differences in some values: 2"
        in stdout.getvalue()
    )
    assert (
        "Number of columns compared with all values equal but known differences found: 1"
        in stdout.getvalue()
    )
    assert (
        "Number of columns compared with all values completely equal: 0"
        in stdout.getvalue()
    )


def test_columns_with_unequal_values_show_mismatch_counts(comparison1):
    stdout = io.StringIO()

    comparison1.report(file=stdout)

    assert "****** Columns with Unequal Values ******" in stdout.getvalue()
    assert re.search(
        r"""Base\s*Column\s*Name \s+ Compare\s*Column\s*Name \s+ Base\s*Dtype \s+ Compare\sDtype \s*
            \#\sMatches \s* \#\sMismatches \n
            -+ \s+ -+ \s+ -+ \s+ -+ \s+ -+ \s+ -+""",
        stdout.getvalue(),
        re.X,
    )
    assert re.search(
        r"""dollar_amt \s+ dollar_amt \s+ bigint \s+ double \s+ 2 \s+ 2""",
        stdout.getvalue(),
        re.X,
    )
    assert re.search(
        r"""float_fld \s+ float_fld \s+ double \s+ double \s+ 1 \s+ 3""",
        stdout.getvalue(),
        re.X,
    )
    assert re.search(
        r"""name \s+ name \s+ string \s+ string \s+ 3 \s+ 1""", stdout.getvalue(), re.X
    )


def test_columns_with_different_names_with_unequal_values_show_mismatch_counts(
    comparison3,
):
    stdout = io.StringIO()

    comparison3.report(file=stdout)

    assert "****** Columns with Unequal Values ******" in stdout.getvalue()
    assert re.search(
        r"""Base\s*Column\s*Name \s+ Compare\s*Column\s*Name \s+ Base\s*Dtype \s+ Compare\sDtype \s*
            \#\sMatches \s* \#\sMismatches \n
            -+ \s+ -+ \s+ -+ \s+ -+ \s+ -+ \s+ -+""",
        stdout.getvalue(),
        re.X,
    )
    assert re.search(
        r"""dollar_amt \s+ dollar_amount \s+ bigint \s+ double \s+ 2 \s+ 3""",
        stdout.getvalue(),
        re.X,
    )
    assert re.search(
        r"""float_fld \s+ float_field \s+ double \s+ double \s+ 4 \s+ 1""",
        stdout.getvalue(),
        re.X,
    )
    assert re.search(
        r"""name \s+ name \s+ string \s+ string \s+ 4 \s+ 1""", stdout.getvalue(), re.X
    )


def test_rows_only_base_returns_a_dataframe_with_rows_only_in_base(spark, comparison1):
    # require schema if contains only 1 row and contain field value as None
    schema = StructType(
        [
            StructField("acct", LongType(), True),
            StructField("date_fld", DateType(), True),
            StructField("dollar_amt", LongType(), True),
            StructField("float_fld", DoubleType(), True),
            StructField("name", StringType(), True),
        ]
    )
    expected_df = spark.createDataFrame(
        [
            Row(
                acct=10000001239,
                date_fld=datetime.date(2017, 1, 1),
                dollar_amt=1,
                float_fld=None,
                name="Lucille Bluth",
            )
        ],
        schema,
    )
    assert comparison1.rows_only_base.count() == 1
    assert (
        expected_df.union(
            comparison1.rows_only_base.select(
                "acct", "date_fld", "dollar_amt", "float_fld", "name"
            )
        )
        .distinct()
        .count()
        == 1
    )


def test_rows_only_compare_returns_a_dataframe_with_rows_only_in_compare(
    spark, comparison1
):
    expected_df = spark.createDataFrame(
        [
            Row(
                acct=10000001238,
                dollar_amt=1.05,
                name="Loose Seal Bluth",
                float_fld=111.0,
                accnt_purge=True,
            )
        ]
    )

    assert comparison1.rows_only_compare.count() == 1
    assert expected_df.union(comparison1.rows_only_compare).distinct().count() == 1


def test_rows_both_mismatch_returns_a_dataframe_with_rows_where_variables_mismatched(
    spark, comparison1
):
    expected_df = spark.createDataFrame(
        [
            Row(
                accnt_purge=False,
                acct=10000001234,
                date_fld=datetime.date(2017, 1, 1),
                dollar_amt_base=123,
                dollar_amt_compare=123.4,
                dollar_amt_match=False,
                float_fld_base=14530.1555,
                float_fld_compare=14530.155,
                float_fld_match=False,
                name_base="George Maharis",
                name_compare="George Michael Bluth",
                name_match=False,
            ),
            Row(
                accnt_purge=False,
                acct=10000001235,
                date_fld=datetime.date(2017, 1, 1),
                dollar_amt_base=0,
                dollar_amt_compare=0.45,
                dollar_amt_match=False,
                float_fld_base=1.0,
                float_fld_compare=None,
                float_fld_match=False,
                name_base="Michael Bluth",
                name_compare="Michael Bluth",
                name_match=True,
            ),
            Row(
                accnt_purge=False,
                acct=10000001236,
                date_fld=datetime.date(2017, 1, 1),
                dollar_amt_base=1345,
                dollar_amt_compare=1345.0,
                dollar_amt_match=True,
                float_fld_base=None,
                float_fld_compare=1.0,
                float_fld_match=False,
                name_base="George Bluth",
                name_compare="George Bluth",
                name_match=True,
            ),
        ]
    )

    assert comparison1.rows_both_mismatch.count() == 3
    assert expected_df.union(comparison1.rows_both_mismatch).distinct().count() == 3


def test_rows_both_mismatch_only_includes_rows_with_true_mismatches_when_known_diffs_are_present(
    spark, comparison_kd1
):
    expected_df = spark.createDataFrame(
        [
            Row(
                acct=10000001237,
                acct_seq=0,
                cd_base="0004",
                cd_compare=4.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
                open_dt_base=datetime.date(2017, 5, 4),
                open_dt_compare=2017124,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                stat_cd_base="*2",
                stat_cd_compare="V3",
                stat_cd_match=False,
                stat_cd_match_type="MISMATCH",
            )
        ]
    )
    assert comparison_kd1.rows_both_mismatch.count() == 1
    assert expected_df.union(comparison_kd1.rows_both_mismatch).distinct().count() == 1


def test_rows_both_all_returns_a_dataframe_with_all_rows_in_both_dataframes(
    spark, comparison1
):
    expected_df = spark.createDataFrame(
        [
            Row(
                accnt_purge=False,
                acct=10000001234,
                date_fld=datetime.date(2017, 1, 1),
                dollar_amt_base=123,
                dollar_amt_compare=123.4,
                dollar_amt_match=False,
                float_fld_base=14530.1555,
                float_fld_compare=14530.155,
                float_fld_match=False,
                name_base="George Maharis",
                name_compare="George Michael Bluth",
                name_match=False,
            ),
            Row(
                accnt_purge=False,
                acct=10000001235,
                date_fld=datetime.date(2017, 1, 1),
                dollar_amt_base=0,
                dollar_amt_compare=0.45,
                dollar_amt_match=False,
                float_fld_base=1.0,
                float_fld_compare=None,
                float_fld_match=False,
                name_base="Michael Bluth",
                name_compare="Michael Bluth",
                name_match=True,
            ),
            Row(
                accnt_purge=False,
                acct=10000001236,
                date_fld=datetime.date(2017, 1, 1),
                dollar_amt_base=1345,
                dollar_amt_compare=1345.0,
                dollar_amt_match=True,
                float_fld_base=None,
                float_fld_compare=1.0,
                float_fld_match=False,
                name_base="George Bluth",
                name_compare="George Bluth",
                name_match=True,
            ),
            Row(
                accnt_purge=False,
                acct=10000001237,
                date_fld=datetime.date(2017, 1, 1),
                dollar_amt_base=123456,
                dollar_amt_compare=123456.0,
                dollar_amt_match=True,
                float_fld_base=345.12,
                float_fld_compare=345.12,
                float_fld_match=True,
                name_base="Bob Loblaw",
                name_compare="Bob Loblaw",
                name_match=True,
            ),
        ]
    )

    assert comparison1.rows_both_all.count() == 4
    assert expected_df.union(comparison1.rows_both_all).distinct().count() == 4


def test_rows_both_all_shows_known_diffs_flag_and_known_diffs_count_as_matches(
    spark, comparison_kd1
):
    expected_df = spark.createDataFrame(
        [
            Row(
                acct=10000001234,
                acct_seq=0,
                cd_base="0001",
                cd_compare=1.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
                open_dt_base=datetime.date(2017, 5, 1),
                open_dt_compare=2017121,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                stat_cd_base="*2",
                stat_cd_compare=None,
                stat_cd_match=True,
                stat_cd_match_type="KNOWN_DIFFERENCE",
            ),
            Row(
                acct=10000001235,
                acct_seq=0,
                cd_base="0002",
                cd_compare=2.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
                open_dt_base=datetime.date(2017, 5, 2),
                open_dt_compare=2017122,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                stat_cd_base="V1",
                stat_cd_compare="V1",
                stat_cd_match=True,
                stat_cd_match_type="MATCH",
            ),
            Row(
                acct=10000001236,
                acct_seq=0,
                cd_base="0003",
                cd_compare=3.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
                open_dt_base=datetime.date(2017, 5, 3),
                open_dt_compare=2017123,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                stat_cd_base="V2",
                stat_cd_compare="V2",
                stat_cd_match=True,
                stat_cd_match_type="MATCH",
            ),
            Row(
                acct=10000001237,
                acct_seq=0,
                cd_base="0004",
                cd_compare=4.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
                open_dt_base=datetime.date(2017, 5, 4),
                open_dt_compare=2017124,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                stat_cd_base="*2",
                stat_cd_compare="V3",
                stat_cd_match=False,
                stat_cd_match_type="MISMATCH",
            ),
            Row(
                acct=10000001238,
                acct_seq=0,
                cd_base="0005",
                cd_compare=5.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
                open_dt_base=datetime.date(2017, 5, 5),
                open_dt_compare=2017125,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                stat_cd_base="*2",
                stat_cd_compare=None,
                stat_cd_match=True,
                stat_cd_match_type="KNOWN_DIFFERENCE",
            ),
        ]
    )

    assert comparison_kd1.rows_both_all.count() == 5
    assert expected_df.union(comparison_kd1.rows_both_all).distinct().count() == 5


def test_rows_both_all_returns_a_dataframe_with_all_rows_in_identical_dataframes(
    spark, comparison2
):
    expected_df = spark.createDataFrame(
        [
            Row(
                acct=10000001234,
                date_fld_base=datetime.date(2017, 1, 1),
                date_fld_compare=datetime.date(2017, 1, 1),
                date_fld_match=True,
                dollar_amt_base=123,
                dollar_amt_compare=123,
                dollar_amt_match=True,
                float_fld_base=14530.1555,
                float_fld_compare=14530.1555,
                float_fld_match=True,
                name_base="George Maharis",
                name_compare="George Maharis",
                name_match=True,
            ),
            Row(
                acct=10000001235,
                date_fld_base=datetime.date(2017, 1, 1),
                date_fld_compare=datetime.date(2017, 1, 1),
                date_fld_match=True,
                dollar_amt_base=0,
                dollar_amt_compare=0,
                dollar_amt_match=True,
                float_fld_base=1.0,
                float_fld_compare=1.0,
                float_fld_match=True,
                name_base="Michael Bluth",
                name_compare="Michael Bluth",
                name_match=True,
            ),
            Row(
                acct=10000001236,
                date_fld_base=datetime.date(2017, 1, 1),
                date_fld_compare=datetime.date(2017, 1, 1),
                date_fld_match=True,
                dollar_amt_base=1345,
                dollar_amt_compare=1345,
                dollar_amt_match=True,
                float_fld_base=None,
                float_fld_compare=None,
                float_fld_match=True,
                name_base="George Bluth",
                name_compare="George Bluth",
                name_match=True,
            ),
            Row(
                acct=10000001237,
                date_fld_base=datetime.date(2017, 1, 1),
                date_fld_compare=datetime.date(2017, 1, 1),
                date_fld_match=True,
                dollar_amt_base=123456,
                dollar_amt_compare=123456,
                dollar_amt_match=True,
                float_fld_base=345.12,
                float_fld_compare=345.12,
                float_fld_match=True,
                name_base="Bob Loblaw",
                name_compare="Bob Loblaw",
                name_match=True,
            ),
            Row(
                acct=10000001239,
                date_fld_base=datetime.date(2017, 1, 1),
                date_fld_compare=datetime.date(2017, 1, 1),
                date_fld_match=True,
                dollar_amt_base=1,
                dollar_amt_compare=1,
                dollar_amt_match=True,
                float_fld_base=None,
                float_fld_compare=None,
                float_fld_match=True,
                name_base="Lucille Bluth",
                name_compare="Lucille Bluth",
                name_match=True,
            ),
        ]
    )

    assert comparison2.rows_both_all.count() == 5
    assert expected_df.union(comparison2.rows_both_all).distinct().count() == 5


def test_rows_both_all_returns_all_rows_in_both_dataframes_for_differently_named_columns(
    spark, comparison3
):
    expected_df = spark.createDataFrame(
        [
            Row(
                accnt_purge=False,
                acct=10000001234,
                date_fld_base=datetime.date(2017, 1, 1),
                date_fld_compare=datetime.date(2017, 1, 1),
                date_fld_match=True,
                dollar_amt_base=123,
                dollar_amt_compare=123.4,
                dollar_amt_match=False,
                float_fld_base=14530.1555,
                float_fld_compare=14530.155,
                float_fld_match=False,
                name_base="George Maharis",
                name_compare="George Michael Bluth",
                name_match=False,
            ),
            Row(
                accnt_purge=False,
                acct=10000001235,
                date_fld_base=datetime.date(2017, 1, 1),
                date_fld_compare=datetime.date(2017, 1, 1),
                date_fld_match=True,
                dollar_amt_base=0,
                dollar_amt_compare=0.45,
                dollar_amt_match=False,
                float_fld_base=1.0,
                float_fld_compare=1.0,
                float_fld_match=True,
                name_base="Michael Bluth",
                name_compare="Michael Bluth",
                name_match=True,
            ),
            Row(
                accnt_purge=False,
                acct=10000001236,
                date_fld_base=datetime.date(2017, 1, 1),
                date_fld_compare=datetime.date(2017, 1, 1),
                date_fld_match=True,
                dollar_amt_base=1345,
                dollar_amt_compare=1345.0,
                dollar_amt_match=True,
                float_fld_base=None,
                float_fld_compare=None,
                float_fld_match=True,
                name_base="George Bluth",
                name_compare="George Bluth",
                name_match=True,
            ),
            Row(
                accnt_purge=False,
                acct=10000001237,
                date_fld_base=datetime.date(2017, 1, 1),
                date_fld_compare=datetime.date(2017, 1, 1),
                date_fld_match=True,
                dollar_amt_base=123456,
                dollar_amt_compare=123456.0,
                dollar_amt_match=True,
                float_fld_base=345.12,
                float_fld_compare=345.12,
                float_fld_match=True,
                name_base="Bob Loblaw",
                name_compare="Bob Loblaw",
                name_match=True,
            ),
            Row(
                accnt_purge=True,
                acct=10000001239,
                date_fld_base=datetime.date(2017, 1, 1),
                date_fld_compare=datetime.date(2017, 1, 1),
                date_fld_match=True,
                dollar_amt_base=1,
                dollar_amt_compare=1.05,
                dollar_amt_match=False,
                float_fld_base=None,
                float_fld_compare=None,
                float_fld_match=True,
                name_base="Lucille Bluth",
                name_compare="Lucille Bluth",
                name_match=True,
            ),
        ]
    )

    assert comparison3.rows_both_all.count() == 5
    assert expected_df.union(comparison3.rows_both_all).distinct().count() == 5


def test_columns_with_unequal_values_text_is_aligned(comparison4):
    stdout = io.StringIO()

    comparison4.report(file=stdout)
    stdout.seek(0)  # Back up to the beginning of the stream

    text_alignment_validator(
        report=stdout,
        section_start="****** Columns with Unequal Values ******",
        section_end="\n",
        left_indices=(1, 2, 3, 4),
        right_indices=(5, 6),
        column_regexes=[
            r"""(Base\sColumn\sName) \s+ (Compare\sColumn\sName) \s+ (Base\sDtype) \s+ (Compare\sDtype) \s+
                (\#\sMatches) \s+ (\#\sMismatches)""",
            r"""(-+) \s+ (-+) \s+ (-+) \s+ (-+) \s+ (-+) \s+ (-+)""",
            r"""(dollar_amt) \s+ (dollar_amt) \s+ (bigint) \s+ (double) \s+ (2) \s+ (2)""",
            r"""(float_fld) \s+ (float_fld) \s+ (double) \s+ (double) \s+ (1) \s+ (3)""",
            r"""(super_duper_big_long_name) \s+ (name) \s+ (string) \s+ (string) \s+ (3) \s+ (1)\s*""",
        ],
    )


def test_columns_with_unequal_values_text_is_aligned_with_known_differences(
    comparison_kd1,
):
    stdout = io.StringIO()

    comparison_kd1.report(file=stdout)
    stdout.seek(0)  # Back up to the beginning of the stream

    text_alignment_validator(
        report=stdout,
        section_start="****** Columns with Unequal Values ******",
        section_end="\n",
        left_indices=(1, 2, 3, 4),
        right_indices=(5, 6, 7),
        column_regexes=[
            r"""(Base\sColumn\sName) \s+ (Compare\sColumn\sName) \s+ (Base\sDtype) \s+ (Compare\sDtype) \s+
                (\#\sMatches) \s+ (\#\sKnown\sDiffs) \s+ (\#\sMismatches)""",
            r"""(-+) \s+ (-+) \s+ (-+) \s+ (-+) \s+ (-+) \s+ (-+) \s+ (-+)""",
            r"""(stat_cd) \s+ (STATC) \s+ (string) \s+ (string) \s+ (2) \s+ (2) \s+ (1)""",
            r"""(open_dt) \s+ (ACCOUNT_OPEN) \s+ (date) \s+ (bigint) \s+ (0) \s+ (5) \s+ (0)""",
            r"""(cd) \s+ (CODE) \s+ (string) \s+ (double) \s+ (0) \s+ (5) \s+ (0)\s*""",
        ],
    )


def test_columns_with_unequal_values_text_is_aligned_with_custom_known_differences(
    comparison_kd2,
):
    stdout = io.StringIO()

    comparison_kd2.report(file=stdout)
    stdout.seek(0)  # Back up to the beginning of the stream

    text_alignment_validator(
        report=stdout,
        section_start="****** Columns with Unequal Values ******",
        section_end="\n",
        left_indices=(1, 2, 3, 4),
        right_indices=(5, 6, 7),
        column_regexes=[
            r"""(Base\sColumn\sName) \s+ (Compare\sColumn\sName) \s+ (Base\sDtype) \s+ (Compare\sDtype) \s+
                (\#\sMatches) \s+ (\#\sKnown\sDiffs) \s+ (\#\sMismatches)""",
            r"""(-+) \s+ (-+) \s+ (-+) \s+ (-+) \s+ (-+) \s+ (-+) \s+ (-+)""",
            r"""(stat_cd) \s+ (STATC) \s+ (string) \s+ (string) \s+ (2) \s+ (2) \s+ (1)""",
            r"""(open_dt) \s+ (ACCOUNT_OPEN) \s+ (date) \s+ (bigint) \s+ (0) \s+ (0) \s+ (5)""",
            r"""(cd) \s+ (CODE) \s+ (string) \s+ (double) \s+ (0) \s+ (5) \s+ (0)\s*""",
        ],
    )


def test_columns_with_unequal_values_text_is_aligned_for_decimals(comparison_decimal):
    stdout = io.StringIO()

    comparison_decimal.report(file=stdout)
    stdout.seek(0)  # Back up to the beginning of the stream

    text_alignment_validator(
        report=stdout,
        section_start="****** Columns with Unequal Values ******",
        section_end="\n",
        left_indices=(1, 2, 3, 4),
        right_indices=(5, 6),
        column_regexes=[
            r"""(Base\sColumn\sName) \s+ (Compare\sColumn\sName) \s+ (Base\sDtype) \s+ (Compare\sDtype) \s+
                (\#\sMatches) \s+ (\#\sMismatches)""",
            r"""(-+) \s+ (-+) \s+ (-+) \s+ (-+) \s+ (-+) \s+ (-+)""",
            r"""(dollar_amt) \s+ (dollar_amt) \s+ (decimal\(8,2\)) \s+ (double) \s+ (1) \s+ (1)""",
        ],
    )


def test_schema_differences_text_is_aligned(comparison4):
    stdout = io.StringIO()

    comparison4.report(file=stdout)
    comparison4.report()
    stdout.seek(0)  # Back up to the beginning of the stream

    text_alignment_validator(
        report=stdout,
        section_start="****** Schema Differences ******",
        section_end="\n",
        left_indices=(1, 2, 3, 4),
        right_indices=(),
        column_regexes=[
            r"""(Base\sColumn\sName) \s+ (Compare\sColumn\sName) \s+ (Base\sDtype) \s+ (Compare\sDtype)""",
            r"""(-+) \s+ (-+) \s+ (-+) \s+ (-+)""",
            r"""(dollar_amt) \s+ (dollar_amt) \s+ (bigint) \s+ (double)""",
        ],
    )


def test_schema_differences_text_is_aligned_for_decimals(comparison_decimal):
    stdout = io.StringIO()

    comparison_decimal.report(file=stdout)
    stdout.seek(0)  # Back up to the beginning of the stream

    text_alignment_validator(
        report=stdout,
        section_start="****** Schema Differences ******",
        section_end="\n",
        left_indices=(1, 2, 3, 4),
        right_indices=(),
        column_regexes=[
            r"""(Base\sColumn\sName) \s+ (Compare\sColumn\sName) \s+ (Base\sDtype) \s+ (Compare\sDtype)""",
            r"""(-+) \s+ (-+) \s+ (-+) \s+ (-+)""",
            r"""(dollar_amt) \s+ (dollar_amt) \s+ (decimal\(8,2\)) \s+ (double)""",
        ],
    )


def test_base_only_columns_text_is_aligned(comparison4):
    stdout = io.StringIO()

    comparison4.report(file=stdout)
    stdout.seek(0)  # Back up to the beginning of the stream

    text_alignment_validator(
        report=stdout,
        section_start="****** Columns In Base Only ******",
        section_end="\n",
        left_indices=(1, 2),
        right_indices=(),
        column_regexes=[
            r"""(Column\sName) \s+ (Dtype)""",
            r"""(-+) \s+ (-+)""",
            r"""(date_fld) \s+ (date)""",
        ],
    )


def test_compare_only_columns_text_is_aligned(comparison4):
    stdout = io.StringIO()

    comparison4.report(file=stdout)
    stdout.seek(0)  # Back up to the beginning of the stream

    text_alignment_validator(
        report=stdout,
        section_start="****** Columns In Compare Only ******",
        section_end="\n",
        left_indices=(1, 2),
        right_indices=(),
        column_regexes=[
            r"""(Column\sName) \s+ (Dtype)""",
            r"""(-+) \s+ (-+)""",
            r"""(accnt_purge) \s+ (boolean)""",
        ],
    )


def text_alignment_validator(
    report, section_start, section_end, left_indices, right_indices, column_regexes
):
    r"""Check to make sure that report output columns are vertically aligned.

    Parameters
    ----------
    report: An iterable returning lines of report output to be validated.
    section_start: A string that represents the beginning of the section to be validated.
    section_end: A string that represents the end of the section to be validated.
    left_indices: The match group indexes (starting with 1) that should be left-aligned
        in the output column.
    right_indices: The match group indexes (starting with 1) that should be right-aligned
        in the output column.
    column_regexes: A list of regular expressions representing the expected output, with
        each column enclosed with parentheses to return a match. The regular expression will
        use the "X" flag, so it may contain whitespace, and any whitespace to be matched
        should be explicitly given with \s. The first line will represent the alignments
        that are expected in the following lines. The number of match groups should cover
        all of the indices given in left/right_indices.

    Runs assertions for every match group specified by left/right_indices to ensure that
    all lines past the first are either left- or right-aligned with the same match group
    on the first line.
    """

    at_column_section = False
    processed_first_line = False
    match_positions = [None] * (len(left_indices + right_indices) + 1)

    for line in report:
        if at_column_section:
            if line == section_end:  # Detect end of section and stop
                break

            if (
                not processed_first_line
            ):  # First line in section - capture text start/end positions
                matches = re.search(column_regexes[0], line, re.X)
                assert matches is not None  # Make sure we found at least this...

                for n in left_indices:
                    match_positions[n] = matches.start(n)
                for n in right_indices:
                    match_positions[n] = matches.end(n)
                processed_first_line = True
            else:  # Match the stuff after the header text
                match = None
                for regex in column_regexes[1:]:
                    match = re.search(regex, line, re.X)
                    if match:
                        break

                if not match:
                    raise AssertionError(f'Did not find a match for line: "{line}"')

                for n in left_indices:
                    assert match_positions[n] == match.start(n)
                for n in right_indices:
                    assert match_positions[n] == match.end(n)

        if not at_column_section and section_start in line:
            at_column_section = True
