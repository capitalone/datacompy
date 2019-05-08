# -*- coding: utf-8 -*-
#
# Copyright 2017 Capital One Services, LLC
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
import logging
import re
import textwrap
from decimal import Decimal

import pytest
import six
from pyspark.sql import Row, SparkSession, types

import datacompy
from datacompy import SparkCompare
from datacompy.sparkcompare import _is_comparable

# Turn off py4j debug messages for all tests in this module
logging.getLogger("py4j").setLevel(logging.INFO)

CACHE_INTERMEDIATES = True


# Declare fixtures
# (if we need to use these in other modules, move to conftest.py)
# @pytest.fixture(scope="module", name="spark")
# def spark_fixture():
#     spark = SparkSession.builder.master("local[2]").appName("pytest").getOrCreate()
#     yield spark
#     spark.stop()


@pytest.fixture(scope="module", name="base_df1")
def base_df1_fixture(spark_session):
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

    return spark_session.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="base_df2")
def base_df2_fixture(spark_session):
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

    return spark_session.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="compare_df1")
def compare_df1_fixture(spark_session):
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

    return spark_session.createDataFrame(mock_data2)


@pytest.fixture(scope="module", name="compare_df2")
def compare_df2_fixture(spark_session):
    """This equals the base_df1 fixture"""
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

    return spark_session.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="compare_df3")
def compare_df3_fixture(spark_session):
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

    return spark_session.createDataFrame(mock_data2)


@pytest.fixture(scope="module", name="base_tol")
def base_tol_fixture(spark_session):
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

    return spark_session.createDataFrame(tol_data1)


@pytest.fixture(scope="module", name="compare_abs_tol")
def compare_tol2_fixture(spark_session):
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

    return spark_session.createDataFrame(tol_data2)


@pytest.fixture(scope="module", name="compare_rel_tol")
def compare_tol3_fixture(spark_session):
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

    return spark_session.createDataFrame(tol_data3)


@pytest.fixture(scope="module", name="compare_both_tol")
def compare_tol4_fixture(spark_session):
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

    return spark_session.createDataFrame(tol_data4)


@pytest.fixture(scope="module", name="base_td")
def base_td_fixture(spark_session):
    mock_data = [
        Row(
            acct=10000001234, acct_seq=0, stat_cd="*2", open_dt=datetime.date(2017, 5, 1), cd="0001"
        ),
        Row(
            acct=10000001235, acct_seq=0, stat_cd="V1", open_dt=datetime.date(2017, 5, 2), cd="0002"
        ),
        Row(
            acct=10000001236, acct_seq=0, stat_cd="V2", open_dt=datetime.date(2017, 5, 3), cd="0003"
        ),
        Row(
            acct=10000001237, acct_seq=0, stat_cd="*2", open_dt=datetime.date(2017, 5, 4), cd="0004"
        ),
        Row(
            acct=10000001238, acct_seq=0, stat_cd="*2", open_dt=datetime.date(2017, 5, 5), cd="0005"
        ),
    ]

    return spark_session.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="compare_source")
def compare_source_fixture(spark_session):
    mock_data = [
        Row(
            ACCOUNT_IDENTIFIER=10000001234, SEQ_NUMBER=0, STATC=None, ACCOUNT_OPEN=2017121, CODE=1.0
        ),
        Row(
            ACCOUNT_IDENTIFIER=10000001235, SEQ_NUMBER=0, STATC="V1", ACCOUNT_OPEN=2017122, CODE=2.0
        ),
        Row(
            ACCOUNT_IDENTIFIER=10000001236, SEQ_NUMBER=0, STATC="V2", ACCOUNT_OPEN=2017123, CODE=3.0
        ),
        Row(
            ACCOUNT_IDENTIFIER=10000001237, SEQ_NUMBER=0, STATC="V3", ACCOUNT_OPEN=2017124, CODE=4.0
        ),
        Row(
            ACCOUNT_IDENTIFIER=10000001238, SEQ_NUMBER=0, STATC=None, ACCOUNT_OPEN=2017125, CODE=5.0
        ),
    ]

    return spark_session.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="base_decimal")
def base_decimal_fixture(spark_session):
    mock_data = [
        Row(acct=10000001234, dollar_amt=Decimal(123.4)),
        Row(acct=10000001235, dollar_amt=Decimal(0.45)),
    ]

    return spark_session.createDataFrame(
        mock_data,
        schema=types.StructType(
            [
                types.StructField("acct", types.LongType(), True),
                types.StructField("dollar_amt", types.DecimalType(8, 2), True),
            ]
        ),
    )


@pytest.fixture(scope="module", name="compare_decimal")
def compare_decimal_fixture(spark_session):
    mock_data = [Row(acct=10000001234, dollar_amt=123.4), Row(acct=10000001235, dollar_amt=0.456)]

    return spark_session.createDataFrame(mock_data)


@pytest.fixture(scope="module", name="comparison_abs_tol")
def comparison_abs_tol_fixture(spark_session, base_tol, compare_abs_tol):
    return SparkCompare(
        spark_session, base_tol, compare_abs_tol, join_columns=["account_identifier"], abs_tol=0.01
    )


@pytest.fixture(scope="module", name="comparison_rel_tol")
def comparison_rel_tol_fixture(spark_session, base_tol, compare_rel_tol):
    return SparkCompare(
        spark_session, base_tol, compare_rel_tol, join_columns=["account_identifier"], rel_tol=0.1
    )


@pytest.fixture(scope="module", name="comparison_both_tol")
def comparison_both_tol_fixture(spark_session, base_tol, compare_both_tol):
    return SparkCompare(
        spark_session,
        base_tol,
        compare_both_tol,
        join_columns=["account_identifier"],
        rel_tol=0.1,
        abs_tol=0.01,
    )


@pytest.fixture(scope="module", name="comparison_neg_tol")
def comparison_neg_tol_fixture(spark_session, base_tol, compare_both_tol):
    return SparkCompare(
        spark_session,
        base_tol,
        compare_both_tol,
        join_columns=["account_identifier"],
        rel_tol=-0.2,
        abs_tol=0.01,
    )


@pytest.fixture(scope="module", name="comparison_kd1")
def comparison_known_diffs1(spark_session, base_td, compare_source):
    return SparkCompare(
        spark_session,
        base_td,
        compare_source,
        join_columns=[("acct", "ACCOUNT_IDENTIFIER"), ("acct_seq", "SEQ_NUMBER")],
        column_mapping=[("stat_cd", "STATC"), ("open_dt", "ACCOUNT_OPEN"), ("cd", "CODE")],
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
def comparison_known_diffs2(spark_session, base_td, compare_source):
    return SparkCompare(
        spark_session,
        base_td,
        compare_source,
        join_columns=[("acct", "ACCOUNT_IDENTIFIER"), ("acct_seq", "SEQ_NUMBER")],
        column_mapping=[("stat_cd", "STATC"), ("open_dt", "ACCOUNT_OPEN"), ("cd", "CODE")],
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
def comparison1_fixture(spark_session, base_df1, compare_df1):
    return SparkCompare(
        spark_session,
        base_df1,
        compare_df1,
        join_columns=["acct"],
        cache_intermediates=CACHE_INTERMEDIATES,
    )


@pytest.fixture(scope="module", name="compare_identical")
def compare_identical_fixture(spark_session, base_df1, compare_df2):
    return SparkCompare(spark_session, base_df1, compare_df2, join_columns=["acct"])


@pytest.fixture(scope="module", name="compare_name_change_extra_col_type_diff")
def compare_name_change_extra_col_type_diff_fixture(spark_session, base_df1, compare_df3):
    return SparkCompare(
        spark_session,
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
def comparison4_fixture(spark_session, base_df2, compare_df1):
    return SparkCompare(
        spark_session,
        base_df2,
        compare_df1,
        join_columns=["acct"],
        column_mapping=[("super_duper_big_long_name", "name")],
    )


@pytest.fixture(scope="module", name="comparison_decimal")
def comparison_decimal_fixture(spark_session, base_decimal, compare_decimal):
    return SparkCompare(spark_session, base_decimal, compare_decimal, join_columns=["acct"])


def test_absolute_tolerances(comparison_abs_tol):
    stdout = six.StringIO()
    comparison_abs_tol.report(file=stdout)
    report = stdout.getvalue()
    print(report)
    assert "Number of rows with some compared columns unequal: 6" in report
    assert "Number of rows with all compared columns equal: 7" in report
    assert "Number of columns compared with some values unequal: 1" in report
    assert "Number of columns compared with all values equal: 4" in report


def test_relative_tolerances(comparison_rel_tol):
    stdout = six.StringIO()
    comparison_rel_tol.report(file=stdout)
    report = stdout.getvalue()
    print(report)
    assert "Number of rows with some compared columns unequal: 6" in report
    assert "Number of rows with all compared columns equal: 7" in report
    assert "Number of columns compared with some values unequal: 1" in report
    assert "Number of columns compared with all values equal: 4" in report


def test_both_tolerances(comparison_both_tol):
    stdout = six.StringIO()
    comparison_both_tol.report(file=stdout)
    report = stdout.getvalue()
    print(report)
    assert "Number of rows with some compared columns unequal: 6" in report
    assert "Number of rows with all compared columns equal: 7" in report
    assert "Number of columns compared with some values unequal: 1" in report
    assert "Number of columns compared with all values equal: 4" in report


def test_negative_tolerances(spark_session, base_tol, compare_both_tol):
    with pytest.raises(ValueError, message="Please enter positive valued tolerances"):
        comp = SparkCompare(
            spark_session,
            base_tol,
            compare_both_tol,
            join_columns=["account_identifier"],
            rel_tol=-0.2,
            abs_tol=0.01,
        )
        comp.report()


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


def test_report_comparison1(comparison1):
    stdout = six.StringIO()

    comparison1.report(file=stdout)
    report = stdout.getvalue()
    print(report)

    # Report outputs the column summary
    assert "Column Summary\n--------------\n" in report
    assert "Number of columns in common: 4" in report
    assert "Number of columns in df1 but not in df2: 1" in report
    assert "Number of columns in df2 but not in df1: 1" in report

    # Check on unequal column renderings
    assert "Columns with Unequal Values or Types\n------------------------------------" in report
    # Header
    assert re.search(r"Column\s+df1 dtype\s+df2 dtype\s+# Unequal\s+Max Diff\s+# Null Diff", report)
    # Lines
    assert re.search(r"dollar_amt\s+bigint\s+double\s+2\s+-1\s+0", report)
    assert re.search(r"float_fld\s+double\s+double\s+3\s+-1\s+2", report)
    assert re.search(r"name\s+string\s+string\s+1\s+-1\s+0", report)

    # test_report_outputs_the_row_summary
    assert "Row Summary\n-----------\n" in report
    assert "Number of rows in common: 4" in report
    assert "Number of rows in df1 but not in df2: 1" in report
    assert "Number of rows in df2 but not in df1: 1" in report

    # test_report_outputs_the_row_equality_comparison
    assert "Number of rows with some compared columns unequal: 3" in report
    assert "Number of rows with all compared columns equal: 1" in report

    # test column comparison outputs number of columns with differences
    assert "Column Comparison\n-----------------" in report
    assert "Number of columns compared with some values unequal: 3" in report
    assert "Number of columns compared with all values equal: 0" in report
    assert "Total number of values which compare unequal: 6" in report

    assert "Sample Rows Only in df1 (First 10 Columns)" in report
    expected = """\
    +-----------+----------+----------+---------+-------------+
    |       acct|  date_fld|dollar_amt|float_fld|         name|
    +-----------+----------+----------+---------+-------------+
    |10000001239|2017-01-01|         1|     null|Lucille Bluth|
    +-----------+----------+----------+---------+-------------+"""
    assert textwrap.dedent(expected) in report

    assert "Sample Rows Only in df2 (First 10 Columns)" in report
    expected = """\
    +-----------+-----------+----------+---------+----------------+
    |accnt_purge|       acct|dollar_amt|float_fld|            name|
    +-----------+-----------+----------+---------+----------------+
    |       true|10000001238|      1.05|    111.0|Loose Seal Bluth|
    +-----------+-----------+----------+---------+----------------+"""
    assert textwrap.dedent(expected) in report


def test_report_identical(compare_identical):
    stdout = six.StringIO()
    compare_identical.report(file=stdout)
    report = stdout.getvalue()
    print(report)  # For debugging tests

    assert "Column Summary\n--------------\n" in report
    assert "Number of columns in common: 5" in report
    assert "Number of columns in df1 but not in df2: 0" in report
    assert "Number of columns in df2 but not in df1: 0" in report

    assert "Column Comparison\n-----------------" in report
    assert "Number of columns compared with some values unequal: 0" in report
    assert "Number of columns compared with all values equal: 4" in report
    assert "Total number of values which compare unequal: 0" in report

    assert "Number of rows with some compared columns unequal: 0" in report
    assert "Number of rows with all compared columns equal: 5" in report


def test_report_compare_name_change_extra_col_type_diff(compare_name_change_extra_col_type_diff):
    stdout = six.StringIO()
    compare_name_change_extra_col_type_diff.report(file=stdout)
    report = stdout.getvalue()
    print(report)

    assert "Column Summary\n--------------\n" in report
    assert "Number of columns in common: 5" in report
    assert "Number of columns in df1 but not in df2: 0" in report
    assert "Number of columns in df2 but not in df1: 1" in report

    assert "Row Summary\n-----------\n" in report
    assert "Number of rows in common: 5" in report
    assert "Number of rows in df1 but not in df2: 0" in report
    assert "Number of rows in df2 but not in df1: 0" in report

    assert "Columns with Unequal Values or Types\n------------------------------------" in report
    # Header
    assert re.search(r"Column\s+df1 dtype\s+df2 dtype\s+# Unequal\s+Max Diff\s+# Null Diff", report)
    # Lines
    assert re.search(r"dollar_amt\s+bigint\s+double\s+3\s+-1\s+0", report)

    assert "Column Comparison\n-----------------" in report
    assert "Number of columns compared with some values unequal: 3" in report
    assert "Number of columns compared with all values equal: 1" in report

    # Check out sample mismatches
    expected_header = """\
    +-----------+----------------+----------------+
    |       acct|dollar_amt (df1)|dollar_amt (df2)|
    +-----------+----------------+----------------+"""
    assert textwrap.dedent(expected_header) in report
    assert "|10000001235|               0|            0.45|" in report
    assert "|10000001234|             123|           123.4|" in report
    assert "|10000001239|               1|            1.05|" in report

    expected = """\
    +-----------+--------------+--------------------+
    |       acct|    name (df1)|          name (df2)|
    +-----------+--------------+--------------------+
    |10000001234|George Maharis|George Michael Bluth|
    +-----------+--------------+--------------------+"""
    assert textwrap.dedent(expected) in report


def test_column_comparison_outputs_number_of_columns_with_differences_for_known_diffs(
    comparison_kd1
):
    stdout = six.StringIO()
    comparison_kd1.report(file=stdout)
    report = stdout.getvalue()

    assert "Column Comparison\n-----------------" in report
    assert "Number of columns compared with some values unequal: 1" in report
    assert "Number of columns compared with all values equal: 2" in report
    assert "Number of columns compared with expected differences: 3" in report

    assert "Columns with Unequal Values or Types\n------------------------------------" in report
    # Header
    assert re.search(r"Column\s+df1 dtype\s+df2 dtype\s+# Unequal\s+Max Diff\s+# Null Diff", report)
    # Lines
    assert re.search(r"stat_cd\s+string\s+string\s+1\s+-1\s+0", report)


def test_column_comparison_outputs_number_of_columns_with_differences_for_custom_known_diffs(
    comparison_kd2
):
    stdout = six.StringIO()
    comparison_kd2.report(file=stdout)
    report = stdout.getvalue()

    assert "Column Comparison\n-----------------" in report
    assert "Number of columns compared with some values unequal: 2" in report
    assert "Number of columns compared with all values equal: 1" in report
    assert "Number of columns compared with expected differences: 2" in report

    assert "Columns with Unequal Values or Types\n------------------------------------" in report
    # Header
    assert re.search(r"Column\s+df1 dtype\s+df2 dtype\s+# Unequal\s+Max Diff\s+# Null Diff", report)
    # Lines
    assert re.search(r"open_dt\s+date\s+bigint\s+5\s+-1\s+0", report)
    assert re.search(r"stat_cd\s+string\s+string\s+1\s+-1\s+0", report)


def test_rows_only_base_returns_a_dataframe_with_rows_only_in_base(spark_session, comparison1):
    # require schema if contains only 1 row and contain field value as None
    schema = types.StructType(
        [
            types.StructField("acct", types.LongType(), True),
            types.StructField("date_fld", types.DateType(), True),
            types.StructField("dollar_amt", types.LongType(), True),
            types.StructField("float_fld", types.DoubleType(), True),
            types.StructField("name", types.StringType(), True),
        ]
    )
    expected_df = spark_session.createDataFrame(
        [
            Row(
                acct=10000001239,
                dollar_amt=1,
                name="Lucille Bluth",
                float_fld=None,
                date_fld=datetime.date(2017, 1, 1),
            )
        ],
        schema,
    )

    assert comparison1.df1_unq_rows.count() == 1
    assert expected_df.union(comparison1.df1_unq_rows).distinct().count() == 1


def test_rows_only_compare_returns_a_dataframe_with_rows_only_in_compare(
    spark_session, comparison1
):
    expected_df = spark_session.createDataFrame(
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

    assert comparison1.df2_unq_rows.count() == 1
    assert expected_df.union(comparison1.df2_unq_rows).distinct().count() == 1


def test_all_rows_mismatched_returns_a_dataframe_with_rows_where_variables_mismatched(
    spark_session, comparison1
):
    expected_df = spark_session.createDataFrame(
        [
            Row(
                acct=10000001234,
                dollar_amt_df1=123,
                dollar_amt_df2=123.4,
                dollar_amt_match=False,
                name_df1="George Maharis",
                name_df2="George Michael Bluth",
                name_match=False,
                float_fld_df1=14530.1555,
                float_fld_df2=14530.155,
                float_fld_match=False,
                date_fld=datetime.date(2017, 1, 1),
                accnt_purge=False,
            ),
            Row(
                acct=10000001235,
                dollar_amt_df1=0,
                dollar_amt_df2=0.45,
                dollar_amt_match=False,
                name_df1="Michael Bluth",
                name_df2="Michael Bluth",
                name_match=True,
                float_fld_df1=1.0,
                float_fld_df2=None,
                float_fld_match=False,
                date_fld=datetime.date(2017, 1, 1),
                accnt_purge=False,
            ),
            Row(
                acct=10000001236,
                dollar_amt_df1=1345,
                dollar_amt_df2=1345.0,
                dollar_amt_match=True,
                name_df1="George Bluth",
                name_df2="George Bluth",
                name_match=True,
                float_fld_df1=None,
                float_fld_df2=1.0,
                float_fld_match=False,
                date_fld=datetime.date(2017, 1, 1),
                accnt_purge=False,
            ),
        ]
    )

    assert comparison1._all_rows_mismatched.count() == 3
    assert expected_df.unionAll(comparison1._all_rows_mismatched).distinct().count() == 3


def test_all_rows_mismatched_only_includes_rows_with_true_mismatches_when_known_diffs_are_present(
    spark_session, comparison_kd1
):
    expected_df = spark_session.createDataFrame(
        [
            Row(
                acct=10000001237,
                acct_seq=0,
                stat_cd_df1="*2",
                stat_cd_df2="V3",
                stat_cd_match=False,
                stat_cd_match_type="MISMATCH",
                open_dt_df1=datetime.date(2017, 5, 4),
                open_dt_df2=2017124,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                cd_df1="0004",
                cd_df2=4.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
            )
        ]
    )

    assert comparison_kd1._all_rows_mismatched.count() == 1
    assert expected_df.unionAll(comparison_kd1._all_rows_mismatched).distinct().count() == 1


def test_all_matched_rows_returns_a_dataframe_with_all_rows_in_both_dataframes(
    spark_session, comparison1
):
    expected_df = spark_session.createDataFrame(
        [
            Row(
                acct=10000001234,
                dollar_amt_df1=123,
                dollar_amt_df2=123.4,
                dollar_amt_match=False,
                name_df1="George Maharis",
                name_df2="George Michael Bluth",
                name_match=False,
                float_fld_df1=14530.1555,
                float_fld_df2=14530.155,
                float_fld_match=False,
                date_fld=datetime.date(2017, 1, 1),
                accnt_purge=False,
            ),
            Row(
                acct=10000001235,
                dollar_amt_df1=0,
                dollar_amt_df2=0.45,
                dollar_amt_match=False,
                name_df1="Michael Bluth",
                name_df2="Michael Bluth",
                name_match=True,
                float_fld_df1=1.0,
                float_fld_df2=None,
                float_fld_match=False,
                date_fld=datetime.date(2017, 1, 1),
                accnt_purge=False,
            ),
            Row(
                acct=10000001236,
                dollar_amt_df1=1345,
                dollar_amt_df2=1345.0,
                dollar_amt_match=True,
                name_df1="George Bluth",
                name_df2="George Bluth",
                name_match=True,
                float_fld_df1=None,
                float_fld_df2=1.0,
                float_fld_match=False,
                date_fld=datetime.date(2017, 1, 1),
                accnt_purge=False,
            ),
            Row(
                acct=10000001237,
                dollar_amt_df1=123456,
                dollar_amt_df2=123456.0,
                dollar_amt_match=True,
                name_df1="Bob Loblaw",
                name_df2="Bob Loblaw",
                name_match=True,
                float_fld_df1=345.12,
                float_fld_df2=345.12,
                float_fld_match=True,
                date_fld=datetime.date(2017, 1, 1),
                accnt_purge=False,
            ),
        ]
    )

    assert comparison1._all_matched_rows.count() == 4
    assert expected_df.unionAll(comparison1._all_matched_rows).distinct().count() == 4


def test_all_matched_rows_shows_known_diffs_flag_and_known_diffs_count_as_matches(
    spark_session, comparison_kd1
):
    expected_df = spark_session.createDataFrame(
        [
            Row(
                acct=10000001234,
                acct_seq=0,
                stat_cd_df1="*2",
                stat_cd_df2=None,
                stat_cd_match=True,
                stat_cd_match_type="KNOWN_DIFFERENCE",
                open_dt_df1=datetime.date(2017, 5, 1),
                open_dt_df2=2017121,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                cd_df1="0001",
                cd_df2=1.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
            ),
            Row(
                acct=10000001235,
                acct_seq=0,
                stat_cd_df1="V1",
                stat_cd_df2="V1",
                stat_cd_match=True,
                stat_cd_match_type="MATCH",
                open_dt_df1=datetime.date(2017, 5, 2),
                open_dt_df2=2017122,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                cd_df1="0002",
                cd_df2=2.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
            ),
            Row(
                acct=10000001236,
                acct_seq=0,
                stat_cd_df1="V2",
                stat_cd_df2="V2",
                stat_cd_match=True,
                stat_cd_match_type="MATCH",
                open_dt_df1=datetime.date(2017, 5, 3),
                open_dt_df2=2017123,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                cd_df1="0003",
                cd_df2=3.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
            ),
            Row(
                acct=10000001237,
                acct_seq=0,
                stat_cd_df1="*2",
                stat_cd_df2="V3",
                stat_cd_match=False,
                stat_cd_match_type="MISMATCH",
                open_dt_df1=datetime.date(2017, 5, 4),
                open_dt_df2=2017124,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                cd_df1="0004",
                cd_df2=4.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
            ),
            Row(
                acct=10000001238,
                acct_seq=0,
                stat_cd_df1="*2",
                stat_cd_df2=None,
                stat_cd_match=True,
                stat_cd_match_type="KNOWN_DIFFERENCE",
                open_dt_df1=datetime.date(2017, 5, 5),
                open_dt_df2=2017125,
                open_dt_match=True,
                open_dt_match_type="KNOWN_DIFFERENCE",
                cd_df1="0005",
                cd_df2=5.0,
                cd_match=True,
                cd_match_type="KNOWN_DIFFERENCE",
            ),
        ]
    )

    assert comparison_kd1._all_matched_rows.count() == 5
    assert expected_df.unionAll(comparison_kd1._all_matched_rows).distinct().count() == 5


def test_all_matched_rows_returns_a_dataframe_with_all_rows_in_identical_dataframes(
    spark_session, compare_identical
):
    expected_df = spark_session.createDataFrame(
        [
            Row(
                acct=10000001234,
                dollar_amt_df1=123,
                dollar_amt_df2=123,
                dollar_amt_match=True,
                name_df1="George Maharis",
                name_df2="George Maharis",
                name_match=True,
                float_fld_df1=14530.1555,
                float_fld_df2=14530.1555,
                float_fld_match=True,
                date_fld_df1=datetime.date(2017, 1, 1),
                date_fld_df2=datetime.date(2017, 1, 1),
                date_fld_match=True,
            ),
            Row(
                acct=10000001235,
                dollar_amt_df1=0,
                dollar_amt_df2=0,
                dollar_amt_match=True,
                name_df1="Michael Bluth",
                name_df2="Michael Bluth",
                name_match=True,
                float_fld_df1=1.0,
                float_fld_df2=1.0,
                float_fld_match=True,
                date_fld_df1=datetime.date(2017, 1, 1),
                date_fld_df2=datetime.date(2017, 1, 1),
                date_fld_match=True,
            ),
            Row(
                acct=10000001236,
                dollar_amt_df1=1345,
                dollar_amt_df2=1345,
                dollar_amt_match=True,
                name_df1="George Bluth",
                name_df2="George Bluth",
                name_match=True,
                float_fld_df1=None,
                float_fld_df2=None,
                float_fld_match=True,
                date_fld_df1=datetime.date(2017, 1, 1),
                date_fld_df2=datetime.date(2017, 1, 1),
                date_fld_match=True,
            ),
            Row(
                acct=10000001237,
                dollar_amt_df1=123456,
                dollar_amt_df2=123456,
                dollar_amt_match=True,
                name_df1="Bob Loblaw",
                name_df2="Bob Loblaw",
                name_match=True,
                float_fld_df1=345.12,
                float_fld_df2=345.12,
                float_fld_match=True,
                date_fld_df1=datetime.date(2017, 1, 1),
                date_fld_df2=datetime.date(2017, 1, 1),
                date_fld_match=True,
            ),
            Row(
                acct=10000001239,
                dollar_amt_df1=1,
                dollar_amt_df2=1,
                dollar_amt_match=True,
                name_df1="Lucille Bluth",
                name_df2="Lucille Bluth",
                name_match=True,
                float_fld_df1=None,
                float_fld_df2=None,
                float_fld_match=True,
                date_fld_df1=datetime.date(2017, 1, 1),
                date_fld_df2=datetime.date(2017, 1, 1),
                date_fld_match=True,
            ),
        ]
    )

    assert compare_identical._all_matched_rows.count() == 5
    assert expected_df.unionAll(compare_identical._all_matched_rows).distinct().count() == 5


def test_all_matched_rows_returns_all_rows_in_both_dataframes_for_differently_named_columns(
    spark_session, compare_name_change_extra_col_type_diff
):
    expected_df = spark_session.createDataFrame(
        [
            Row(
                acct=10000001234,
                dollar_amt_df1=123,
                dollar_amt_df2=123.4,
                dollar_amt_match=False,
                name_df1="George Maharis",
                name_df2="George Michael Bluth",
                name_match=False,
                float_fld_df1=14530.1555,
                float_fld_df2=14530.155,
                float_fld_match=False,
                date_fld_df1=datetime.date(2017, 1, 1),
                date_fld_df2=datetime.date(2017, 1, 1),
                date_fld_match=True,
                accnt_purge=False,
            ),
            Row(
                acct=10000001235,
                dollar_amt_df1=0,
                dollar_amt_df2=0.45,
                dollar_amt_match=False,
                name_df1="Michael Bluth",
                name_df2="Michael Bluth",
                name_match=True,
                float_fld_df1=1.0,
                float_fld_df2=1.0,
                float_fld_match=True,
                date_fld_df1=datetime.date(2017, 1, 1),
                date_fld_df2=datetime.date(2017, 1, 1),
                date_fld_match=True,
                accnt_purge=False,
            ),
            Row(
                acct=10000001236,
                dollar_amt_df1=1345,
                dollar_amt_df2=1345.0,
                dollar_amt_match=True,
                name_df1="George Bluth",
                name_df2="George Bluth",
                name_match=True,
                float_fld_df1=None,
                float_fld_df2=None,
                float_fld_match=True,
                date_fld_df1=datetime.date(2017, 1, 1),
                date_fld_df2=datetime.date(2017, 1, 1),
                date_fld_match=True,
                accnt_purge=False,
            ),
            Row(
                acct=10000001237,
                dollar_amt_df1=123456,
                dollar_amt_df2=123456.0,
                dollar_amt_match=True,
                name_df1="Bob Loblaw",
                name_df2="Bob Loblaw",
                name_match=True,
                float_fld_df1=345.12,
                float_fld_df2=345.12,
                float_fld_match=True,
                date_fld_df1=datetime.date(2017, 1, 1),
                date_fld_df2=datetime.date(2017, 1, 1),
                date_fld_match=True,
                accnt_purge=False,
            ),
            Row(
                acct=10000001239,
                dollar_amt_df1=1,
                dollar_amt_df2=1.05,
                dollar_amt_match=False,
                name_df1="Lucille Bluth",
                name_df2="Lucille Bluth",
                name_match=True,
                float_fld_df1=None,
                float_fld_df2=None,
                float_fld_match=True,
                date_fld_df1=datetime.date(2017, 1, 1),
                date_fld_df2=datetime.date(2017, 1, 1),
                date_fld_match=True,
                accnt_purge=True,
            ),
        ]
    )

    assert compare_name_change_extra_col_type_diff._all_matched_rows.count() == 5
    assert (
        expected_df.unionAll(compare_name_change_extra_col_type_diff._all_matched_rows)
        .distinct()
        .count()
        == 5
    )
