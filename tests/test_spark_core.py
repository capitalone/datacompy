#
# Copyright 2020 Capital One Services, LLC
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

"""
Testing out the Spark datacompy functionality
"""
import io
import logging
import sys
from datetime import datetime
from decimal import Decimal
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_series_equal
from pyspark.sql.types import *
from pytest import raises

from datacompy.spark_core import (
    SparkCompare,
    calculate_max_diff,
    columns_equal,
    temp_column_name,
)
from tests.utility import assert_columns_equal, spark_to_pandas

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def test_numeric_columns_equal_abs(spark_session):
    data = """a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "a", "b", "expected", abs_tol=0.2)
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )


def test_numeric_columns_equal_rel(spark_session):
    data = """a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "a", "b", "expected", rel_tol=0.2)
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )


def test_string_columns_equal(spark_session):

    schema = StructType(
        [
            StructField("a", StringType(), True),
            StructField("b", StringType(), True),
            StructField("expected", BooleanType(), True),
        ]
    )

    data = """a|b|expected
Hi|Hi|True
Yo|Yo|True
Hey|Hey |False
résumé|resume|False
résumé|résumé|True
💩|💩|True
💩|🤔|False
 | |True
  | |False
datacompy|DataComPy|False
something||False
|something|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf, schema)
    actual_out = columns_equal(df, "a", "b", "expected", rel_tol=0.2)
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )


def test_string_columns_equal_with_ignore_spaces(spark_session):

    schema = StructType(
        [
            StructField("a", StringType(), True),
            StructField("b", StringType(), True),
            StructField("expected", BooleanType(), True),
        ]
    )

    data = """a|b|expected
Hi|Hi|True
Yo|Yo|True
Hey|Hey |True
résumé|resume|False
résumé|résumé|True
💩|💩|True
💩|🤔|False
 | |True
  |       |True
datacompy|DataComPy|False
something||False
|something|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf, schema)
    actual_out = columns_equal(df, "a", "b", "expected", ignore_spaces=True)
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )


def test_string_columns_equal_with_ignore_spaces_and_case(spark_session):

    schema = StructType(
        [
            StructField("a", StringType(), True),
            StructField("b", StringType(), True),
            StructField("expected", BooleanType(), True),
        ]
    )

    data = """a|b|expected
Hi|Hi|True
Yo|Yo|True
Hey|Hey |True
résumé|resume|False
résumé|résumé|True
💩|💩|True
💩|🤔|False
 | |True
  |       |True
datacompy|DataComPy|True
something||False
|something|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf, schema)
    actual_out = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True, ignore_case=True
    )
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )


def test_date_columns_equal(spark_session):

    schema = StructType(
        [
            StructField("a", StringType(), True),
            StructField("b", StringType(), True),
            StructField("expected", BooleanType(), True),
        ]
    )

    data = """a|b|expected
2017-01-01|2017-01-01|True
2017-01-02|2017-01-02|True
2017-10-01|2017-10-10|False
2017-01-01||False
|2017-01-01|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf, schema)
    # First compare just the strings
    actual_out = columns_equal(df, "a", "b", "expected", rel_tol=0.2)
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )

    # Then compare converted to datetime objects
    pdf["a"] = pd.to_datetime(pdf["a"])
    pdf["b"] = pd.to_datetime(pdf["b"])
    actual_out = columns_equal(df, "a", "b", "expected", rel_tol=0.2)
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )
    # and reverse
    actual_out_rev = columns_equal(df, "b", "a", "expected", rel_tol=0.2)
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out_rev)["expected"], check_names=False
    )

    # Then compare converted to date objects
    pdf["a"] = pd.to_datetime(pdf["a"]).dt.date
    pdf["b"] = pd.to_datetime(pdf["b"]).dt.date
    actual_out = columns_equal(df, "a", "b", "expected", rel_tol=0.2)
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )
    # and reverse
    actual_out_rev = columns_equal(df, "b", "a", "expected", rel_tol=0.2)
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out_rev)["expected"], check_names=False
    )


def test_date_columns_equal_with_ignore_spaces(spark_session):

    schema = StructType(
        [
            StructField("a", StringType(), True),
            StructField("b", StringType(), True),
            StructField("expected", BooleanType(), True),
        ]
    )

    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf, schema)
    # First compare just the strings
    actual_out = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True
    )
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )

    # Then compare converted to datetime objects
    pdf["a"] = pd.to_datetime(pdf["a"])
    pdf["b"] = pd.to_datetime(pdf["b"])
    actual_out = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True
    )
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )
    # and reverse
    actual_out_rev = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True
    )
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out_rev)["expected"], check_names=False
    )

    # Then compare converted to date objects
    pdf["a"] = pd.to_datetime(pdf["a"]).dt.date
    pdf["b"] = pd.to_datetime(pdf["b"]).dt.date
    actual_out = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True
    )
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )
    # and reverse
    actual_out_rev = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True
    )
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out_rev)["expected"], check_names=False
    )


def test_date_columns_equal_with_ignore_spaces_and_case(spark_session):

    schema = StructType(
        [
            StructField("a", StringType(), True),
            StructField("b", StringType(), True),
            StructField("expected", BooleanType(), True),
        ]
    )

    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf, schema)
    # First compare just the strings
    actual_out = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True, ignore_case=True
    )
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )

    # Then compare converted to datetime objects
    pdf["a"] = pd.to_datetime(pdf["a"])
    pdf["b"] = pd.to_datetime(pdf["b"])
    actual_out = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True, ignore_case=True
    )
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )
    # and reverse
    actual_out_rev = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True
    )
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out_rev)["expected"], check_names=False
    )

    # Then compare converted to date objects
    pdf["a"] = pd.to_datetime(pdf["a"]).dt.date
    pdf["b"] = pd.to_datetime(pdf["b"]).dt.date
    actual_out = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True, ignore_case=True
    )
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )
    # and reverse
    actual_out_rev = columns_equal(
        df, "a", "b", "expected", rel_tol=0.2, ignore_spaces=True
    )
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out_rev)["expected"], check_names=False
    )


def test_date_columns_unequal(spark_session):
    """different data types are skipped, should always return 0"""
    schema = StructType(
        [
            StructField("a", StringType(), True),
            StructField("b", StringType(), True),
            StructField("a_dt", DateType(), True),
            StructField("b_dt", DateType(), True),
            StructField("expected", BooleanType(), True),
        ]
    )
    pdf = pd.DataFrame([{"a": "2017-01-01", "b": "2017-01-02"}, {"a": "2017-01-01"}])
    pdf["a_dt"] = pd.to_datetime(pdf["a"])
    pdf["b_dt"] = pd.to_datetime(pdf["b"])
    pdf["expected"] = False
    df = spark_session.createDataFrame(pdf, schema)

    # different data types are skipped, should always return 0
    assert (
        columns_equal(df, "a", "a_dt", "expected").where("expected == True").count()
        == 0
    )
    assert (
        columns_equal(df, "b", "b_dt", "expected").where("expected == True").count()
        == 0
    )
    assert (
        columns_equal(df, "a_dt", "a", "expected").where("expected == True").count()
        == 0
    )
    assert (
        columns_equal(df, "b_dt", "b", "expected").where("expected == True").count()
        == 0
    )


def test_decimal_double_columns_equal(spark_session):

    pdf = pd.DataFrame(
        [
            {"a": Decimal("1"), "b": 1, "expected": True},
            {"a": Decimal("1.3"), "b": 1.3, "expected": True},
            {"a": Decimal("1.000003"), "b": 1.000003, "expected": True},
            {"a": Decimal("1.000000004"), "b": 1.000000003, "expected": False},
            {"a": Decimal("1.3"), "b": 1.2, "expected": False},
            {"a": Decimal("1.3"), "b": np.nan, "expected": False},
            {"a": Decimal("1.3"), "b": 1, "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    df = spark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "a", "b", "expected")
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )


def test_decimal_double_columns_equal_rel(spark_session):
    pdf = pd.DataFrame(
        [
            {"a": Decimal("1"), "b": 1, "expected": True},
            {"a": Decimal("1.3"), "b": 1.3, "expected": True},
            {"a": Decimal("1.000003"), "b": 1.000003, "expected": True},
            {"a": Decimal("1.000000004"), "b": 1.000000003, "expected": True},
            {"a": Decimal("1.3"), "b": 1.2, "expected": False},
            {"a": Decimal("1.3"), "b": np.nan, "expected": False},
            {"a": Decimal("1.3"), "b": 1, "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    df = spark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "a", "b", "expected", abs_tol=0.001)
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )


def test_decimal_columns_equal(spark_session):
    pdf = pd.DataFrame(
        [
            {"a": Decimal("1"), "b": Decimal("1"), "expected": True},
            {"a": Decimal("1.3"), "b": Decimal("1.3"), "expected": True},
            {"a": Decimal("1.000003"), "b": Decimal("1.000003"), "expected": True},
            {
                "a": Decimal("1.000000004"),
                "b": Decimal("1.000000003"),
                "expected": False,
            },
            {"a": Decimal("1.3"), "b": Decimal("1.2"), "expected": False},
        ]
    )
    df = spark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "a", "b", "expected")
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )


def test_decimal_columns_equal_rel(spark_session):
    pdf = pd.DataFrame(
        [
            {"a": Decimal("1"), "b": Decimal("1"), "expected": True},
            {"a": Decimal("1.3"), "b": Decimal("1.3"), "expected": True},
            {"a": Decimal("1.000003"), "b": Decimal("1.000003"), "expected": True},
            {
                "a": Decimal("1.000000004"),
                "b": Decimal("1.000000003"),
                "expected": True,
            },
            {"a": Decimal("1.3"), "b": Decimal("1.2"), "expected": False},
        ]
    )
    df = spark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "a", "b", "expected", abs_tol=0.001)
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )


def test_infinity_and_beyond(spark_session):
    # https://spark.apache.org/docs/latest/sql-ref-datatypes.html#positivenegative-infinity-semantics
    # Positive/negative infinity multiplied by 0 returns NaN.
    # Positive infinity sorts lower than NaN and higher than any other values.
    # Negative infinity sorts lower than any other values.
    pdf = pd.DataFrame(
        [
            {"a": np.inf, "b": np.inf, "expected": True},
            {"a": -np.inf, "b": -np.inf, "expected": True},
            {"a": -np.inf, "b": np.inf, "expected": True},
            {"a": np.inf, "b": -np.inf, "expected": True},
            {"a": 1, "b": 1, "expected": True},
            {"a": 1, "b": 0, "expected": False},
        ]
    )
    df = spark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "a", "b", "expected")
    expect_out = pdf["expected"]
    assert_series_equal(
        expect_out, spark_to_pandas(actual_out)["expected"], check_names=False
    )


def test_compare_df_setter_bad(spark_session):
    pdf = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df = spark_session.createDataFrame(pdf)
    df = df.withColumnRenamed("b", "a")
    with raises(TypeError, match="df1 must be a Spark DataFrame"):
        compare = SparkCompare(spark_session, "a", "a", ["a"])
    with raises(ValueError, match="df1 must have all columns from join_columns"):
        compare = SparkCompare(spark_session, df, df, ["b"])
    with raises(ValueError, match="df1 must have unique column names"):
        compare = SparkCompare(spark_session, df, df, ["a"])


def test_compare_df_setter_good(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, ["a"])
    assert compare.df1.columns == ["a", "b"]
    assert compare.df2.columns == ["a", "b"]
    assert compare.join_columns == ["a"]
    compare = SparkCompare(spark_session, df1, df2, ["a", "b"])
    assert compare.df1.columns == ["a", "b"]
    assert compare.df2.columns == ["a", "b"]
    assert compare.join_columns == ["a", "b"]


def test_compare_df_setter_different_cases(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    pdf2 = pd.DataFrame([{"A": 1, "b": 2}, {"A": 2, "b": 3}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, ["a"])
    assert compare.df1.columns == ["a", "b"]
    assert compare.df2.columns == ["a", "b"]


def test_compare_df_setter_bad_index(spark_session):
    pdf = pd.DataFrame([{"a": 1, "A": 2}, {"a": 2, "A": 2}])
    df = df2 = spark_session.createDataFrame(pdf)
    with raises(TypeError, match="df1 must be a Spark DataFrame"):
        compare = SparkCompare(spark_session, "a", "a", ["a"])
    with raises(ValueError, match="df1 must have unique column names"):
        compare = SparkCompare(spark_session, df, df2, ["a"])


def test_columns_overlap(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, ["a"])
    assert compare.df1_unq_columns() == set()
    assert compare.df2_unq_columns() == set()
    assert compare.intersect_columns() == {"a", "b"}


def test_columns_no_overlap(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2, "d": "oh"}, {"a": 2, "b": 3, "d": "ya"}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, ["a"])
    assert compare.df1_unq_columns() == {"c"}
    assert compare.df2_unq_columns() == {"d"}
    assert compare.intersect_columns() == {"a", "b"}


def test_columns_maintain_order_through_set_operations(spark_session):
    pdf1 = pd.DataFrame(
        [
            (("A"), (0), (1), (2), (3), (4), (-2)),
            (("B"), (0), (2), (2), (3), (4), (-3)),
        ],
        columns=["join", "f", "g", "b", "h", "a", "c"],
    )
    pdf2 = pd.DataFrame(
        [
            (("A"), (0), (1), (2), (-1), (4), (-3)),
            (("B"), (1), (2), (3), (-1), (4), (-2)),
        ],
        columns=["join", "e", "h", "b", "a", "g", "d"],
    )
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, ["join"])
    assert list(compare.df1_unq_columns()) == ["f", "c"]
    assert list(compare.df2_unq_columns()) == ["e", "d"]
    assert list(compare.intersect_columns()) == ["join", "g", "b", "h", "a"]


def test_10k_rows(spark_session):
    pdf1 = pd.DataFrame(np.random.randint(0, 100, size=(10000, 2)), columns=["b", "c"])
    pdf1.reset_index(inplace=True)
    pdf1.columns = ["a", "b", "c"]
    pdf2 = pdf1.copy()
    pdf2["b"] = pdf2["b"] + 0.1
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare_tol = SparkCompare(spark_session, df1, df2, ["a"], abs_tol=0.2)
    assert compare_tol.matches()
    assert compare_tol.df1_unq_rows.count() == 0
    assert compare_tol.df2_unq_rows.count() == 0
    assert compare_tol.intersect_columns() == {"a", "b", "c"}
    assert compare_tol.all_columns_match()
    assert compare_tol.all_rows_overlap()
    assert compare_tol.intersect_rows_match()

    compare_no_tol = SparkCompare(spark_session, df1, df2, ["a"])
    assert not compare_no_tol.matches()
    assert compare_no_tol.df1_unq_rows.count() == 0
    assert compare_no_tol.df2_unq_rows.count() == 0
    assert compare_no_tol.intersect_columns() == {"a", "b", "c"}
    assert compare_no_tol.all_columns_match()
    assert compare_no_tol.all_rows_overlap()
    assert not compare_no_tol.intersect_rows_match()


def test_subset(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    pdf2 = pd.DataFrame([{"a": 1, "c": "hi"}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    comp = SparkCompare(spark_session, df1, df2, ["a"])
    assert comp.subset()


def test_not_subset(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "great"}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    comp = SparkCompare(spark_session, df1, df2, ["a"])
    assert not comp.subset()


def test_large_subset(spark_session):
    pdf1 = pd.DataFrame(np.random.randint(0, 100, size=(10000, 2)), columns=["b", "c"])
    pdf1.reset_index(inplace=True)
    pdf1.columns = ["a", "b", "c"]
    pdf2 = pdf1[["a", "b"]].sample(50).copy()
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    comp = SparkCompare(spark_session, df1, df2, ["a"])
    assert not comp.matches()
    assert comp.subset()


def test_string_joiner(spark_session):
    pdf1 = pd.DataFrame([{"ab": 1, "bc": 2}, {"ab": 2, "bc": 2}])
    pdf2 = pd.DataFrame([{"ab": 1, "bc": 2}, {"ab": 2, "bc": 2}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, "ab")
    assert compare.matches()


def test_decimal_with_joins(spark_session):
    pdf1 = pd.DataFrame([{"a": Decimal("1"), "b": 2}, {"a": Decimal("2"), "b": 2}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, "a")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_nulls(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": Decimal("2")}, {"a": 2, "b": Decimal("2")}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}, {"a": 3, "b": 2}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, "a")
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_joins(spark_session):
    pdf1 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    pdf2 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, "a")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_temp_column_name(spark_session):
    pdf1 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    pdf2 = pd.DataFrame(
        [{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_0"


def test_temp_column_name_one_has(spark_session):
    pdf1 = pd.DataFrame([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    pdf2 = pd.DataFrame(
        [{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_1"


def test_temp_column_name_both_have(spark_session):
    pdf1 = pd.DataFrame([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    pdf2 = pd.DataFrame(
        [
            {"_temp_0": "hi", "b": 2},
            {"_temp_0": "bye", "b": 2},
        ]
    )
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_1"


def test_temp_column_name_both_have_2(spark_session):
    pdf1 = pd.DataFrame(
        [
            {"_temp_0": "hi", "_temp_1": "hi", "b": 2},
            {"_temp_0": "hi", "_temp_1": "hi", "b": 2},
        ]
    )
    pdf2 = pd.DataFrame(
        [
            {"_temp_0": "hi", "_temp_1": "hi", "b": 2},
            {"_temp_0": "hi", "_temp_1": "hi", "b": 2},
        ]
    )
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_2"


def test_temp_column_name_both_have_3(spark_session):
    pdf1 = pd.DataFrame(
        [
            {"_temp_0": "hi", "_temp_1": "hi", "_temp_2": "hi", "b": 2},
            {"_temp_0": "hi", "_temp_1": "hi", "_temp_2": "hi", "b": 2},
        ]
    )
    pdf2 = pd.DataFrame(
        [
            {"_temp_0": "hi", "_temp_1": "hi", "_temp_2": "hi", "b": 2},
            {"_temp_0": "hi", "_temp_1": "hi", "_temp_2": "hi", "b": 2},
        ]
    )
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_3"


def test_temp_column_name_one_already(spark_session):
    pdf1 = pd.DataFrame(
        [
            {"_temp_1": "hi", "_temp_2": "hi", "b": 2},
            {"_temp_1": "hi", "_temp_2": "hi", "b": 2},
        ]
    )
    pdf2 = pd.DataFrame(
        [
            {"_temp_1": "hi", "_temp_2": "hi", "b": 2},
            {"_temp_1": "hi", "_temp_2": "hi", "b": 2},
        ]
    )
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_0"


# ### Duplicate testing!
def test_simple_dupes_one_field(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_two_fields(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2, "c": 2}, {"a": 1, "b": 2, "c": 2}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2, "c": 2}, {"a": 1, "b": 2, "c": 2}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, join_columns=["a", "b"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_one_field_two_vals(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_one_field_two_vals_2(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert not compare.matches()
    assert compare.df1_unq_rows.count() == 0
    assert compare.df2_unq_rows.count() == 1
    assert compare.intersect_rows.count() == 1
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_one_field_three_to_two_vals(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 2}])
    pdf2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert compare.matches()
    assert compare.df1_unq_rows.count() == 0
    assert compare.df2_unq_rows.count() == 0
    assert compare.intersect_rows.count() == 1
    # Just render the report to make sure it renders.
    t = compare.report()


def test_dupes_from_real_data(spark_session):

    schema = StructType(
        [
            StructField("acct_id", IntegerType(), True),
            StructField("acct_sfx_num", IntegerType(), True),
            StructField("trxn_post_dt", StringType(), True),
            StructField("trxn_post_seq_num", IntegerType(), True),
            StructField("trxn_amt", DoubleType(), True),
            StructField("trxn_dt", StringType(), True),
            StructField("debit_cr_cd", StringType(), True),
            StructField("cash_adv_trxn_comn_cntry_cd", StringType(), True),
            StructField("mrch_catg_cd", IntegerType(), True),
            StructField("mrch_pstl_cd", StringType(), True),
            StructField("visa_mail_phn_cd", DoubleType(), True),
            StructField("visa_rqstd_pmt_svc_cd", StringType(), True),
            StructField("mc_pmt_facilitator_idn_num", DoubleType(), True),
        ]
    )

    data = """acct_id,acct_sfx_num,trxn_post_dt,trxn_post_seq_num,trxn_amt,trxn_dt,debit_cr_cd,cash_adv_trxn_comn_cntry_cd,mrch_catg_cd,mrch_pstl_cd,visa_mail_phn_cd,visa_rqstd_pmt_svc_cd,mc_pmt_facilitator_idn_num
100,0,2017-06-17,1537019,30.64,2017-06-15,D,CAN,5812,M2N5P5,,,0.0
200,0,2017-06-24,1022477,485.32,2017-06-22,D,USA,4511,7114,7.0,1,
100,0,2017-06-17,1537039,2.73,2017-06-16,D,CAN,5812,M4J 1M9,,,0.0
200,0,2017-06-29,1049223,22.41,2017-06-28,D,USA,4789,21211,,A,
100,0,2017-06-17,1537029,34.05,2017-06-16,D,CAN,5812,M4E 2C7,,,0.0
200,0,2017-06-29,1049213,9.12,2017-06-28,D,CAN,5814,0,,,
100,0,2017-06-19,1646426,165.21,2017-06-17,D,CAN,5411,M4M 3H9,,,0.0
200,0,2017-06-30,1233082,28.54,2017-06-29,D,USA,4121,94105,7.0,G,
100,0,2017-06-19,1646436,17.87,2017-06-18,D,CAN,5812,M4J 1M9,,,0.0
200,0,2017-06-30,1233092,24.39,2017-06-29,D,USA,4121,94105,7.0,G,
100,0,2017-06-19,1646446,5.27,2017-06-17,D,CAN,5200,M4M 3G6,,,0.0
200,0,2017-06-30,1233102,61.8,2017-06-30,D,CAN,4121,0,,,
100,0,2017-06-20,1607573,41.99,2017-06-19,D,CAN,5661,M4C1M9,,,0.0
200,0,2017-07-01,1009403,2.31,2017-06-29,D,USA,5814,22102,,F,
100,0,2017-06-20,1607553,86.88,2017-06-19,D,CAN,4812,H2R3A8,,,0.0
200,0,2017-07-01,1009423,5.5,2017-06-29,D,USA,5812,2903,,F,
100,0,2017-06-20,1607563,25.17,2017-06-19,D,CAN,5641,M4C 1M9,,,0.0
200,0,2017-07-01,1009433,214.12,2017-06-29,D,USA,3640,20170,,A,
100,0,2017-06-20,1607593,1.67,2017-06-19,D,CAN,5814,M2N 6L7,,,0.0
200,0,2017-07-01,1009393,2.01,2017-06-29,D,USA,5814,22102,,F,"""
    pdf1 = pd.read_csv(io.StringIO(data), sep=",")
    pdf2 = pdf1.copy()
    df1 = spark_session.createDataFrame(pdf1, schema)
    df2 = spark_session.createDataFrame(pdf2, schema)
    compare_acct = SparkCompare(spark_session, df1, df2, join_columns=["acct_id"])
    assert compare_acct.matches()
    compare_unq = SparkCompare(
        spark_session,
        df1,
        df2,
        join_columns=["acct_id", "acct_sfx_num", "trxn_post_dt", "trxn_post_seq_num"],
    )
    assert compare_unq.matches()
    # Just render the report to make sure it renders.
    t = compare_acct.report()
    r = compare_unq.report()


def test_strings_with_joins_with_ignore_spaces(spark_session):
    pdf1 = pd.DataFrame([{"a": "hi", "b": " A"}, {"a": "bye", "b": "A"}])
    pdf2 = pd.DataFrame([{"a": "hi", "b": "A"}, {"a": "bye", "b": "A "}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = SparkCompare(spark_session, df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_joins_with_ignore_case(spark_session):
    pdf1 = pd.DataFrame([{"a": "hi", "b": "a"}, {"a": "bye", "b": "A"}])
    pdf2 = pd.DataFrame([{"a": "hi", "b": "A"}, {"a": "bye", "b": "a"}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, "a", ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = SparkCompare(spark_session, df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_joins_with_ignore_spaces(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": " A"}, {"a": 2, "b": "A"}])
    pdf2 = pd.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A "}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = SparkCompare(spark_session, df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_joins_with_ignore_case(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": "a"}, {"a": 2, "b": "A"}])
    pdf2 = pd.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "a"}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, "a", ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = SparkCompare(spark_session, df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_ignore_spaces_and_join_columns(spark_session):
    pdf1 = pd.DataFrame([{"a": "hi", "b": "A"}, {"a": "bye", "b": "A"}])
    pdf2 = pd.DataFrame([{"a": " hi ", "b": "A"}, {"a": " bye ", "b": "A"}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.count_matching_rows() == 0

    compare = SparkCompare(spark_session, df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2


#
def test_integers_with_ignore_spaces_and_join_columns(spark_session):
    pdf1 = pd.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A"}])
    pdf2 = pd.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A"}])
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, "a", ignore_spaces=False)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2

    compare = SparkCompare(spark_session, df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2


def test_sample_mismatch(spark_session):

    schema = StructType(
        [
            StructField("acct_id", LongType(), True),
            StructField("dollar_amt", DoubleType(), True),
            StructField("name", StringType(), True),
            StructField("float_fld", DoubleType(), True),
            StructField("date_fld", StringType(), True),
        ]
    )

    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.4,George Michael Bluth,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Robert Loblaw,345.12,
    10000001238,1.05,Loose Seal Bluth,111,
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """
    pdf1 = pd.read_csv(io.StringIO(data1), sep=",")
    pdf2 = pd.read_csv(io.StringIO(data2), sep=",")
    df1 = spark_session.createDataFrame(pdf1, schema)
    df2 = spark_session.createDataFrame(pdf2, schema)
    compare = SparkCompare(spark_session, df1, df2, "acct_id")

    output = compare.sample_mismatch(column="name", sample_count=1)
    output_df = spark_to_pandas(output)
    assert output_df.shape[0] == 1
    assert (output_df.name_df1 != output_df.name_df2).all()

    output = compare.sample_mismatch(column="name", sample_count=2)
    output_df = spark_to_pandas(output)
    assert output_df.shape[0] == 2
    assert (output_df.name_df1 != output_df.name_df2).all()

    output = compare.sample_mismatch(column="name", sample_count=3)
    output_df = spark_to_pandas(output)
    assert output_df.shape[0] == 2
    assert (output_df.name_df1 != output_df.name_df2).all()


def test_all_mismatch(spark_session):
    schema = StructType(
        [
            StructField("acct_id", LongType(), True),
            StructField("dollar_amt", DoubleType(), True),
            StructField("name", StringType(), True),
            StructField("float_fld", DoubleType(), True),
            StructField("date_fld", StringType(), True),
        ]
    )
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.4,George Michael Bluth,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Robert Loblaw,345.12,
    10000001238,1.05,Loose Seal Bluth,111,
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """
    pdf1 = pd.read_csv(io.StringIO(data1), sep=",")
    pdf2 = pd.read_csv(io.StringIO(data2), sep=",")
    df1 = spark_session.createDataFrame(pdf1, schema)
    df2 = spark_session.createDataFrame(pdf2, schema)
    compare = SparkCompare(spark_session, df1, df2, "acct_id")

    output = compare.all_mismatch()
    output = spark_to_pandas(output)
    assert output.shape[0] == 4

    assert (output.name_df1 != output.name_df2).values.sum() == 2
    assert (~(output.name_df1 != output.name_df2)).values.sum() == 2

    assert (output.dollar_amt_df1 != output.dollar_amt_df2).values.sum() == 1
    assert (~(output.dollar_amt_df1 != output.dollar_amt_df2)).values.sum() == 3

    assert (output.float_fld_df1 != output.float_fld_df2).values.sum() == 3
    assert (~(output.float_fld_df1 != output.float_fld_df2)).values.sum() == 1


MAX_DIFF_DF = pd.DataFrame(
    {
        "base": [1, 1, 1, 1, 1],
        "floats": [1.1, 1.1, 1.1, 1.2, 0.9],
        "decimals": [
            Decimal("1.1"),
            Decimal("1.1"),
            Decimal("1.1"),
            Decimal("1.1"),
            Decimal("1.1"),
        ],
        "null_floats": [np.nan, 1.1, 1, 1, 1],
        "strings": ["1", "1", "1", "1.1", "1"],
        "infinity": [1, 1, 1, 1, np.inf],
    }
)


@pytest.mark.parametrize(
    "column,expected",
    [
        ("base", 0),
        ("floats", 0.2),
        ("decimals", 0.1),
        ("null_floats", 0.1),
        ("strings", 0.1),
        ("infinity", np.inf),
    ],
)
def test_calculate_max_diff(spark_session, column, expected):
    df = spark_session.createDataFrame(MAX_DIFF_DF)
    assert np.isclose(calculate_max_diff(df, "base", column), expected)


def test_cols_with_nulls(spark_session):
    pdf1 = pd.DataFrame(
        {
            "fld_1": [1, 2, 3, 4, 5],
            "fld_2": [
                "A",
                "B",
                "C",
                None,
                None,
            ],
        }
    )
    pdf2 = pd.DataFrame(
        {
            "fld_1": [1, 2, 3, 4, 5],
            "fld_2": [
                "A",
                "B",
                "C",
                None,
                None,
            ],
        }
    )
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    comp = SparkCompare(spark_session, df1, df2, join_columns=["fld_1"])
    assert comp.subset()
    assert comp.count_matching_rows() == 5


def test_join_with_nulls(spark_session):
    pdf1 = pd.DataFrame(
        {
            "fld_1": [1, 2, 3, 4, 5, 6],
            "fld_2": ["A", "B", "C", None, "Y", None],
        }
    )
    pdf2 = pd.DataFrame(
        {
            "fld_1": [1, 2, 3, 4, 5, 6],
            "fld_2": ["A", "B", "C", "X", None, None],
        }
    )
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    comp = SparkCompare(spark_session, df1, df2, join_columns=["fld_1", "fld_2"])
    assert comp.df1_unq_rows.count() == 2
    assert comp.df2_unq_rows.count() == 2
    assert comp.intersect_rows.count() == 4
    assert len(comp.intersect_rows.columns) == 5
    assert "fld_1" in comp.intersect_rows.columns
    assert "fld_2" in comp.intersect_rows.columns


def test_lower(spark_session):
    """This function tests the toggle to use lower case for column names or not"""
    # should match
    pdf1 = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    pdf2 = pd.DataFrame({"a": [1, 2, 3], "B": [0, 1, 2]})
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert compare.matches()

    # test join column
    # should match
    pdf1 = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    pdf2 = pd.DataFrame({"A": [1, 2, 3], "B": [0, 1, 2]})
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert compare.matches()


def test_integer_column_names(spark_session):
    """This function tests that integer column names would also work"""
    pdf1 = pd.DataFrame({1: [1, 2, 3], 2: [0, 1, 2]})
    pdf2 = pd.DataFrame({1: [1, 2, 3], 2: [0, 1, 2]})
    df1 = spark_session.createDataFrame(pdf1)
    df2 = spark_session.createDataFrame(pdf2)
    compare = SparkCompare(spark_session, df1, df2, join_columns=[1])
    assert compare.matches()
