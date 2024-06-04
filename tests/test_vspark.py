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

"""
Testing out the datacompy functionality
"""

import io
import logging
import re
import sys
from datetime import datetime
from decimal import Decimal
from io import StringIO
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pytest import raises

pytest.importorskip("pyspark")

import pyspark.pandas as ps  # noqa: E402
from pandas.testing import assert_series_equal  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    BooleanType,
    StringType,
    StructField,
    StructType,
)

from datacompy.vspark import (  # noqa: E402
    VSparkCompare,
    _generate_id_within_group,
    calculate_max_diff,
    columns_equal,
    temp_column_name,
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


pandas_version = pytest.mark.skipif(
    pd.__version__ >= "2.0.0", reason="Pandas 2 is currently not supported"
)

pd.DataFrame.iteritems = pd.DataFrame.items  # Pandas 2+ compatability
np.bool = np.bool_  # Numpy 1.24.3+ comptability


@pandas_version
def test_numeric_columns_equal_abs(spark_session):
    data = """a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""

    df = spark_session.createDataFrame(pd.read_csv(StringIO(data), sep="|"))
    actual_out = columns_equal(df, "a", "b", "actual", abs_tol=0.2).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


@pandas_version
def test_numeric_columns_equal_rel(spark_session):
    data = """a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""
    df = spark_session.createDataFrame(pd.read_csv(StringIO(data), sep="|"))
    actual_out = columns_equal(df, "a", "b", "actual", rel_tol=0.2).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


@pandas_version
def test_string_columns_equal(spark_session):
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
    df = spark_session.createDataFrame(pd.read_csv(StringIO(data), sep="|"))
    actual_out = columns_equal(df, "a", "b", "actual", rel_tol=0.2).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


@pandas_version
def test_string_columns_equal_with_ignore_spaces(spark_session):
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
    df = spark_session.createDataFrame(pd.read_csv(StringIO(data), sep="|"))
    actual_out = columns_equal(
        df, "a", "b", "actual", rel_tol=0.2, ignore_spaces=True
    ).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


@pandas_version
def test_string_columns_equal_with_ignore_spaces_and_case(spark_session):
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
    df = spark_session.createDataFrame(pd.read_csv(StringIO(data), sep="|"))
    actual_out = columns_equal(
        df, "a", "b", "actual", rel_tol=0.2, ignore_spaces=True, ignore_case=True
    ).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


@pandas_version
def test_date_columns_equal(spark_session):
    data = """a|b|expected
2017-01-01|2017-01-01|True
2017-01-02|2017-01-02|True
2017-10-01|2017-10-10|False
2017-01-01||False
|2017-01-01|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf)
    # First compare just the strings
    actual_out = columns_equal(df, "a", "b", "actual", rel_tol=0.2).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    pdf["a"] = pd.to_datetime(pdf["a"])
    pdf["b"] = pd.to_datetime(pdf["b"])
    df = spark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "a", "b", "actual", rel_tol=0.2).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)
    # and reverse
    actual_out_rev = columns_equal(df, "b", "a", "actual", rel_tol=0.2).toPandas()[
        "actual"
    ]
    assert_series_equal(expect_out, actual_out_rev, check_names=False)


@pandas_version
def test_date_columns_equal_with_ignore_spaces(spark_session):
    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf)
    # First compare just the strings
    actual_out = columns_equal(
        df, "a", "b", "actual", rel_tol=0.2, ignore_spaces=True
    ).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    pdf["a"] = pd.to_datetime(pdf["a"])
    pdf["b"] = pd.to_datetime(pdf["b"])
    df = spark_session.createDataFrame(pdf)
    actual_out = columns_equal(
        df, "a", "b", "actual", rel_tol=0.2, ignore_spaces=True
    ).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)
    # and reverse
    actual_out_rev = columns_equal(
        df, "b", "a", "actual", rel_tol=0.2, ignore_spaces=True
    ).toPandas()["actual"]
    assert_series_equal(expect_out, actual_out_rev, check_names=False)


@pandas_version
def test_date_columns_equal_with_ignore_spaces_and_case(spark_session):
    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = spark_session.createDataFrame(pdf)
    # First compare just the strings
    actual_out = columns_equal(
        df, "a", "b", "actual", rel_tol=0.2, ignore_spaces=True, ignore_case=True
    ).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    pdf["a"] = pd.to_datetime(pdf["a"])
    pdf["b"] = pd.to_datetime(pdf["b"])
    df = spark_session.createDataFrame(pdf)
    actual_out = columns_equal(
        df, "a", "b", "actual", rel_tol=0.2, ignore_spaces=True, ignore_case=True
    ).toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)
    # and reverse
    actual_out_rev = columns_equal(
        df, "b", "a", "actual", rel_tol=0.2, ignore_spaces=True, ignore_case=True
    ).toPandas()["actual"]
    assert_series_equal(expect_out, actual_out_rev, check_names=False)


@pandas_version
def test_date_columns_unequal(spark_session):
    """I want datetime fields to match with dates stored as strings"""
    data = [{"a": "2017-01-01", "b": "2017-01-02"}, {"a": "2017-01-01"}]
    pdf = pd.DataFrame(data)
    pdf["a_dt"] = pd.to_datetime(pdf["a"])
    pdf["b_dt"] = pd.to_datetime(pdf["b"])
    df = spark_session.createDataFrame(pdf)
    assert columns_equal(df, "a", "a_dt", "actual").toPandas()["actual"].all()
    assert columns_equal(df, "b", "b_dt", "actual").toPandas()["actual"].all()
    assert columns_equal(df, "a_dt", "a", "actual").toPandas()["actual"].all()
    assert columns_equal(df, "b_dt", "b", "actual").toPandas()["actual"].all()
    assert not columns_equal(df, "b_dt", "a", "actual").toPandas()["actual"].any()
    assert not columns_equal(df, "a_dt", "b", "actual").toPandas()["actual"].any()
    assert not columns_equal(df, "a", "b_dt", "actual").toPandas()["actual"].any()
    assert not columns_equal(df, "b", "a_dt", "actual").toPandas()["actual"].any()


@pandas_version
def test_bad_date_columns(spark_session):
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    data = [
        {"a": "2017-01-01", "b": "2017-01-01"},
        {"a": "2017-01-01", "b": "217-01-01"},
    ]
    pdf = pd.DataFrame(data)
    pdf["a_dt"] = pd.to_datetime(pdf["a"])
    df = spark_session.createDataFrame(pdf)
    assert not columns_equal(df, "a_dt", "b", "actual").toPandas()["actual"].all()
    assert columns_equal(df, "a_dt", "b", "actual").toPandas()["actual"].any()


@pandas_version
def test_rounded_date_columns(spark_session):
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    data = [
        {"a": "2017-01-01", "b": "2017-01-01 00:00:00.000000", "exp": True},
        {"a": "2017-01-01", "b": "2017-01-01 00:00:00.123456", "exp": False},
        {"a": "2017-01-01", "b": "2017-01-01 00:00:01.000000", "exp": False},
        {"a": "2017-01-01", "b": "2017-01-01 00:00:00", "exp": True},
    ]
    pdf = pd.DataFrame(data)
    pdf["a_dt"] = pd.to_datetime(pdf["a"])
    df = spark_session.createDataFrame(pdf)
    actual = columns_equal(df, "a_dt", "b", "actual").toPandas()["actual"]
    expected = df.select("exp").toPandas()["exp"]
    assert_series_equal(actual, expected, check_names=False)


@pandas_version
def test_decimal_float_columns_equal(spark_session):
    df = spark_session.createDataFrame(
        [
            {"a": Decimal("1"), "b": 1, "expected": True},
            {"a": Decimal("1.3"), "b": 1.3, "expected": True},
            {"a": Decimal("1.000003"), "b": 1.000003, "expected": True},
            {"a": Decimal("1.000000004"), "b": 1.000000003, "expected": False},
            {"a": Decimal("1.3"), "b": 1.2, "expected": False},
            {"a": np.nan, "b": np.nan, "expected": True},
            {"a": np.nan, "b": 1, "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    actual_out = columns_equal(df, "a", "b", "actual").toPandas()["actual"]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


@pandas_version
def test_decimal_float_columns_equal_rel(spark_session):
    df = spark_session.createDataFrame(
        [
            {"a": Decimal("1"), "b": 1, "expected": True},
            {"a": Decimal("1.3"), "b": 1.3, "expected": True},
            {"a": Decimal("1.000003"), "b": 1.000003, "expected": True},
            {"a": Decimal("1.000000004"), "b": 1.000000003, "expected": True},
            {"a": Decimal("1.3"), "b": 1.2, "expected": False},
            {"a": np.nan, "b": np.nan, "expected": True},
            {"a": np.nan, "b": 1, "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    actual_out = columns_equal(df, "a", "b", "actual", abs_tol=0.001).toPandas()[
        "actual"
    ]
    expect_out = df.select("expected").toPandas()["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


@pandas_version
def test_decimal_columns_equal(spark_session):
    df = spark_session.createDataFrame(
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
            {"a": np.nan, "b": np.nan, "expected": True},
            {"a": np.nan, "b": Decimal("1"), "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    actual_out = columns_equal(df.a, df.b)
    expect_out = df["expected"]
    assert_series_equal(
        expect_out.to_pandas(), actual_out.to_pandas(), check_names=False
    )


@pandas_version
def test_decimal_columns_equal_rel(spark_session):
    df = spark_session.createDataFrame(
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
            {"a": np.nan, "b": np.nan, "expected": True},
            {"a": np.nan, "b": Decimal("1"), "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    actual_out = columns_equal(df.a, df.b, abs_tol=0.001)
    expect_out = df["expected"]
    assert_series_equal(
        expect_out.to_pandas(), actual_out.to_pandas(), check_names=False
    )


@pandas_version
def test_infinity_and_beyond(spark_session):
    # https://spark.apache.org/docs/latest/sql-ref-datatypes.html#positivenegative-infinity-semantics
    # Positive/negative infinity multiplied by 0 returns NaN.
    # Positive infinity sorts lower than NaN and higher than any other values.
    # Negative infinity sorts lower than any other values.
    df = spark_session.createDataFrame(
        [
            {"a": np.inf, "b": np.inf, "expected": True},
            {"a": -np.inf, "b": -np.inf, "expected": True},
            {"a": -np.inf, "b": np.inf, "expected": True},
            {"a": np.inf, "b": -np.inf, "expected": True},
            {"a": 1, "b": 1, "expected": True},
            {"a": 1, "b": 0, "expected": False},
        ]
    )
    actual_out = columns_equal(df.a, df.b)
    expect_out = df["expected"]
    assert_series_equal(
        expect_out.to_pandas(), actual_out.to_pandas(), check_names=False
    )


@pandas_version
def test_compare_df_setter_bad(spark_session):
    df = spark_session.createDataFrame([{"a": 1, "c": 2}, {"a": 2, "c": 2}])
    with raises(TypeError, match="df1 must be a pyspark.pandas.frame.DataFrame"):
        compare = VSparkCompare(spark_session, "a", "a", ["a"])
    with raises(ValueError, match="df1 must have all columns from join_columns"):
        compare = VSparkCompare(spark_session, df, df.copy(), ["b"])
    df_dupe = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 3}])
    assert (
        VSparkCompare(spark_session, df_dupe, df_dupe.copy(), ["a", "b"])
        .df1.equals(df_dupe)
        .all()
        .all()
    )


@pandas_version
def test_compare_df_setter_good(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = spark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 3}])
    compare = VSparkCompare(spark_session, df1, df2, ["a"])
    assert compare.df1.equals(df1).all().all()
    assert compare.df2.equals(df2).all().all()
    assert compare.join_columns == ["a"]
    compare = VSparkCompare(spark_session, df1, df2, ["A", "b"])
    assert compare.df1.equals(df1).all().all()
    assert compare.df2.equals(df2).all().all()
    assert compare.join_columns == ["a", "b"]


@pandas_version
def test_compare_df_setter_different_cases(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = spark_session.createDataFrame([{"A": 1, "b": 2}, {"A": 2, "b": 3}])
    compare = VSparkCompare(spark_session, df1, df2, ["a"])
    assert compare.df1.equals(df1).all().all()
    assert compare.df2.equals(df2).all().all()


@pandas_version
def test_columns_overlap(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    compare = VSparkCompare(spark_session, df1, df2, ["a"])
    assert compare.df1_unq_columns() == set()
    assert compare.df2_unq_columns() == set()
    assert compare.intersect_columns() == {"a", "b"}


@pandas_version
def test_columns_no_overlap(spark_session):
    df1 = spark_session.createDataFrame(
        [{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}]
    )
    df2 = spark_session.createDataFrame(
        [{"a": 1, "b": 2, "d": "oh"}, {"a": 2, "b": 3, "d": "ya"}]
    )
    compare = VSparkCompare(spark_session, df1, df2, ["a"])
    assert compare.df1_unq_columns() == {"c"}
    assert compare.df2_unq_columns() == {"d"}
    assert compare.intersect_columns() == {"a", "b"}


@pandas_version
def test_columns_maintain_order_through_set_operations(spark_session):
    df1 = spark_session.createDataFrame(
        [
            (("A"), (0), (1), (2), (3), (4), (-2)),
            (("B"), (0), (2), (2), (3), (4), (-3)),
        ],
        columns=["join", "f", "g", "b", "h", "a", "c"],
    )
    df2 = spark_session.createDataFrame(
        [
            (("A"), (0), (1), (2), (-1), (4), (-3)),
            (("B"), (1), (2), (3), (-1), (4), (-2)),
        ],
        columns=["join", "e", "h", "b", "a", "g", "d"],
    )
    compare = VSparkCompare(spark_session, df1, df2, ["join"])
    assert list(compare.df1_unq_columns()) == ["f", "c"]
    assert list(compare.df2_unq_columns()) == ["e", "d"]
    assert list(compare.intersect_columns()) == ["join", "g", "b", "h", "a"]


@pandas_version
def test_10k_rows(spark_session):
    pdf = pd.DataFrame(np.random.randint(0, 100, size=(10000, 2)), columns=["b", "c"])
    pdf.reset_index(inplace=True)
    pdf.columns = ["a", "b", "c"]
    pdf2 = pdf.copy()
    pdf2["b"] = pdf2["b"] + 0.1
    df1 = spark_session.createDataFrame(pdf)
    df2 = spark_session.createDataFrame(pdf2)
    compare_tol = VSparkCompare(spark_session, df1, df2, ["a"], abs_tol=0.2)
    assert compare_tol.matches()
    assert compare_tol.df1_unq_rows.count() == 0
    assert compare_tol.df2_unq_rows.count() == 0
    assert compare_tol.intersect_columns() == {"a", "b", "c"}
    assert compare_tol.all_columns_match()
    assert compare_tol.all_rows_overlap()
    assert compare_tol.intersect_rows_match()

    compare_no_tol = VSparkCompare(spark_session, df1, df2, ["a"])
    assert not compare_no_tol.matches()
    assert compare_no_tol.df1_unq_rows.count() == 0
    assert compare_no_tol.df2_unq_rows.count() == 0
    assert compare_no_tol.intersect_columns() == {"a", "b", "c"}
    assert compare_no_tol.all_columns_match()
    assert compare_no_tol.all_rows_overlap()
    assert not compare_no_tol.intersect_rows_match()


@pandas_version
def test_subset(spark_session, caplog):
    caplog.set_level(logging.DEBUG)
    df1 = spark_session.createDataFrame(
        [{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}]
    )
    df2 = spark_session.createDataFrame([{"a": 1, "c": "hi"}])
    comp = VSparkCompare(spark_session, df1, df2, ["a"])
    assert comp.subset()
    assert "Checking equality" in caplog.text


@pandas_version
def test_not_subset(spark_session, caplog):
    caplog.set_level(logging.INFO)
    df1 = spark_session.createDataFrame(
        [{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}]
    )
    df2 = spark_session.createDataFrame(
        [{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "great"}]
    )
    comp = VSparkCompare(spark_session, df1, df2, ["a"])
    assert not comp.subset()
    assert "c: 1 / 2 (50.00%) match" in caplog.text


@pandas_version
def test_large_subset(spark_session):
    df1 = spark_session.createDataFrame(
        np.random.randint(0, 100, size=(10000, 2)), columns=["b", "c"]
    )
    df1.reset_index(inplace=True)
    df1.columns = ["a", "b", "c"]
    df2 = df1[["a", "b"]].head(50).copy()
    comp = VSparkCompare(spark_session, df1, df2, ["a"])
    assert not comp.matches()
    assert comp.subset()


@pandas_version
def test_string_joiner(spark_session):
    df1 = spark_session.createDataFrame([{"ab": 1, "bc": 2}, {"ab": 2, "bc": 2}])
    df2 = spark_session.createDataFrame([{"ab": 1, "bc": 2}, {"ab": 2, "bc": 2}])
    compare = VSparkCompare(spark_session, df1, df2, "ab")
    assert compare.matches()


@pandas_version
def test_decimal_with_joins(spark_session):
    df1 = spark_session.createDataFrame(
        [{"a": Decimal("1"), "b": 2}, {"a": Decimal("2"), "b": 2}]
    )
    df2 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    compare = VSparkCompare(spark_session, df1, df2, "a")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


@pandas_version
def test_decimal_with_nulls(spark_session):
    df1 = spark_session.createDataFrame(
        [{"a": 1, "b": Decimal("2")}, {"a": 2, "b": Decimal("2")}]
    )
    df2 = spark_session.createDataFrame(
        [{"a": 1, "b": 2}, {"a": 2, "b": 2}, {"a": 3, "b": 2}]
    )
    compare = VSparkCompare(spark_session, df1, df2, "a")
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.intersect_rows_match()


@pandas_version
def test_strings_with_joins(spark_session):
    df1 = spark_session.createDataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df2 = spark_session.createDataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    compare = VSparkCompare(spark_session, df1, df2, "a")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


@pandas_version
def test_temp_column_name(spark_session):
    df1 = spark_session.createDataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df2 = spark_session.createDataFrame(
        [{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_0"


@pandas_version
def test_temp_column_name_one_has(spark_session):
    df1 = spark_session.createDataFrame(
        [{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}]
    )
    df2 = spark_session.createDataFrame(
        [{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_1"


@pandas_version
def test_temp_column_name_both_have_temp_1(spark_session):
    df1 = spark_session.createDataFrame(
        [{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}]
    )
    df2 = spark_session.createDataFrame(
        [
            {"_temp_0": "hi", "b": 2},
            {"_temp_0": "bye", "b": 2},
            {"a": "back fo mo", "b": 3},
        ]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_1"


@pandas_version
def test_temp_column_name_both_have_temp_2(spark_session):
    df1 = spark_session.createDataFrame(
        [{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}]
    )
    df2 = spark_session.createDataFrame(
        [
            {"_temp_0": "hi", "b": 2},
            {"_temp_1": "bye", "b": 2},
            {"a": "back fo mo", "b": 3},
        ]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_2"


@pandas_version
def test_temp_column_name_one_already(spark_session):
    df1 = spark_session.createDataFrame(
        [{"_temp_1": "hi", "b": 2}, {"_temp_1": "bye", "b": 2}]
    )
    df2 = spark_session.createDataFrame(
        [
            {"_temp_1": "hi", "b": 2},
            {"_temp_1": "bye", "b": 2},
            {"a": "back fo mo", "b": 3},
        ]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_0"


### Duplicate testing!
@pandas_version
def test_simple_dupes_one_field(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    compare = VSparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


@pandas_version
def test_simple_dupes_two_fields(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 2}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 2}])
    compare = VSparkCompare(spark_session, df1, df2, join_columns=["a", "b"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


@pandas_version
def test_simple_dupes_one_field_two_vals_1(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    compare = VSparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


@pandas_version
def test_simple_dupes_one_field_two_vals_2(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 0}])
    compare = VSparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert not compare.matches()
    assert len(compare.df1_unq_rows) == 1
    assert len(compare.df2_unq_rows) == 1
    assert len(compare.intersect_rows) == 1
    # Just render the report to make sure it renders.
    t = compare.report()


@pandas_version
def test_simple_dupes_one_field_three_to_two_vals(spark_session):
    df1 = spark_session.createDataFrame(
        [{"a": 1, "b": 2}, {"a": 1, "b": 0}, {"a": 1, "b": 0}]
    )
    df2 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    compare = VSparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert not compare.matches()
    assert len(compare.df1_unq_rows) == 1
    assert len(compare.df2_unq_rows) == 0
    assert len(compare.intersect_rows) == 2
    # Just render the report to make sure it renders.
    t = compare.report()

    assert "(First 1 Columns)" in compare.report(column_count=1)
    assert "(First 2 Columns)" in compare.report(column_count=2)


@pandas_version
def test_dupes_from_real_data(spark_session):
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
    df1 = spark_session.createDataFrame(pd.read_csv(StringIO(data), sep=","))
    df2 = df1.copy()
    compare_acct = VSparkCompare(spark_session, df1, df2, join_columns=["acct_id"])
    assert compare_acct.matches()
    compare_unq = VSparkCompare(
        spark_session,
        df1,
        df2,
        join_columns=["acct_id", "acct_sfx_num", "trxn_post_dt", "trxn_post_seq_num"],
    )
    assert compare_unq.matches()
    # Just render the report to make sure it renders.
    t = compare_acct.report()
    r = compare_unq.report()


@pandas_version
def test_strings_with_joins_with_ignore_spaces(spark_session):
    df1 = spark_session.createDataFrame(
        [{"a": "hi", "b": " A"}, {"a": "bye", "b": "A"}]
    )
    df2 = spark_session.createDataFrame(
        [{"a": "hi", "b": "A"}, {"a": "bye", "b": "A "}]
    )
    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


@pandas_version
def test_strings_with_joins_with_ignore_case(spark_session):
    df1 = spark_session.createDataFrame([{"a": "hi", "b": "a"}, {"a": "bye", "b": "A"}])
    df2 = spark_session.createDataFrame([{"a": "hi", "b": "A"}, {"a": "bye", "b": "a"}])
    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


@pandas_version
def test_decimal_with_joins_with_ignore_spaces(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": " A"}, {"a": 2, "b": "A"}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A "}])
    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


@pandas_version
def test_decimal_with_joins_with_ignore_case(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": "a"}, {"a": 2, "b": "A"}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "a"}])
    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


@pandas_version
def test_joins_with_ignore_spaces(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": " A"}, {"a": 2, "b": "A"}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A "}])

    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


@pandas_version
def test_joins_with_ignore_case(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": "a"}, {"a": 2, "b": "A"}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "a"}])

    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


@pandas_version
def test_strings_with_ignore_spaces_and_join_columns(spark_session):
    df1 = spark_session.createDataFrame([{"a": "hi", "b": "A"}, {"a": "bye", "b": "A"}])
    df2 = spark_session.createDataFrame(
        [{"a": " hi ", "b": "A"}, {"a": " bye ", "b": "A"}]
    )
    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.count_matching_rows() == 0

    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2


@pandas_version
def test_integers_with_ignore_spaces_and_join_columns(spark_session):
    df1 = spark_session.createDataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A"}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A"}])
    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_spaces=False)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2

    compare = VSparkCompare(spark_session, df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2


@pandas_version
def test_sample_mismatch(spark_session):
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

    df1 = spark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = spark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))

    compare = VSparkCompare(spark_session, df1, df2, "acct_id")

    output = compare.sample_mismatch(column="name", sample_count=1).toPandas()
    assert output.shape[0] == 1
    assert (output.name_df1 != output.name_df2).all()

    output = compare.sample_mismatch(column="name", sample_count=2).toPandas()
    assert output.shape[0] == 2
    assert (output.name_df1 != output.name_df2).all()

    output = compare.sample_mismatch(column="name", sample_count=3).toPandas()
    assert output.shape[0] == 2
    assert (output.name_df1 != output.name_df2).all()


@pandas_version
def test_all_mismatch_not_ignore_matching_cols_no_cols_matching(spark_session):
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
    df1 = spark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = spark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = VSparkCompare(spark_session, df1, df2, "acct_id")

    output = compare.all_mismatch()
    assert output.shape[0] == 4
    assert output.shape[1] == 10

    assert (output.name_df1 != output.name_df2).values.sum() == 2
    assert (~(output.name_df1 != output.name_df2)).values.sum() == 2

    assert (output.dollar_amt_df1 != output.dollar_amt_df2).values.sum() == 1
    assert (~(output.dollar_amt_df1 != output.dollar_amt_df2)).values.sum() == 3

    assert (output.float_fld_df1 != output.float_fld_df2).values.sum() == 3
    assert (~(output.float_fld_df1 != output.float_fld_df2)).values.sum() == 1

    assert (output.date_fld_df1 != output.date_fld_df2).values.sum() == 4
    assert (~(output.date_fld_df1 != output.date_fld_df2)).values.sum() == 0


@pandas_version
def test_all_mismatch_not_ignore_matching_cols_some_cols_matching(spark_session):
    # Columns dollar_amt and name are matching
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
        10000001234,123.45,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1,2017-01-01
        10000001236,1345,George Bluth,,2017-01-01
        10000001237,123456,Bob Loblaw,345.12,2017-01-01
        10000001239,1.05,Lucille Bluth,,2017-01-01
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
        10000001234,123.45,George Maharis,14530.155,
        10000001235,0.45,Michael Bluth,,
        10000001236,1345,George Bluth,1,
        10000001237,123456,Bob Loblaw,345.12,
        10000001238,1.05,Lucille Bluth,111,
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """
    df1 = spark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = spark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = VSparkCompare(spark_session, df1, df2, "acct_id")

    output = compare.all_mismatch()
    assert output.shape[0] == 4
    assert output.shape[1] == 10

    assert (output.name_df1 != output.name_df2).values.sum() == 0
    assert (~(output.name_df1 != output.name_df2)).values.sum() == 4

    assert (output.dollar_amt_df1 != output.dollar_amt_df2).values.sum() == 0
    assert (~(output.dollar_amt_df1 != output.dollar_amt_df2)).values.sum() == 4

    assert (output.float_fld_df1 != output.float_fld_df2).values.sum() == 3
    assert (~(output.float_fld_df1 != output.float_fld_df2)).values.sum() == 1

    assert (output.date_fld_df1 != output.date_fld_df2).values.sum() == 4
    assert (~(output.date_fld_df1 != output.date_fld_df2)).values.sum() == 0


@pandas_version
def test_all_mismatch_ignore_matching_cols_some_cols_matching_diff_rows(spark_session):
    # Case where there are rows on either dataset which don't match up.
    # Columns dollar_amt and name are matching
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    10000001241,1111.05,Lucille Bluth,
    """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Bob Loblaw,345.12,
    10000001238,1.05,Lucille Bluth,111,
    """
    df1 = spark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = spark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = VSparkCompare(spark_session, df1, df2, "acct_id")

    output = compare.all_mismatch(ignore_matching_cols=True)

    assert output.shape[0] == 4
    assert output.shape[1] == 6

    assert (output.float_fld_df1 != output.float_fld_df2).values.sum() == 3
    assert (~(output.float_fld_df1 != output.float_fld_df2)).values.sum() == 1

    assert (output.date_fld_df1 != output.date_fld_df2).values.sum() == 4
    assert (~(output.date_fld_df1 != output.date_fld_df2)).values.sum() == 0

    assert not ("name_df1" in output and "name_df2" in output)
    assert not ("dollar_amt_df1" in output and "dollar_amt_df1" in output)


@pandas_version
def test_all_mismatch_ignore_matching_cols_some_calls_matching(spark_session):
    # Columns dollar_amt and name are matching
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Bob Loblaw,345.12,
    10000001238,1.05,Lucille Bluth,111,
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """
    df1 = spark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = spark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = VSparkCompare(spark_session, df1, df2, "acct_id")

    output = compare.all_mismatch(ignore_matching_cols=True)

    assert output.shape[0] == 4
    assert output.shape[1] == 6

    assert (output.float_fld_df1 != output.float_fld_df2).values.sum() == 3
    assert (~(output.float_fld_df1 != output.float_fld_df2)).values.sum() == 1

    assert (output.date_fld_df1 != output.date_fld_df2).values.sum() == 4
    assert (~(output.date_fld_df1 != output.date_fld_df2)).values.sum() == 0

    assert not ("name_df1" in output and "name_df2" in output)
    assert not ("dollar_amt_df1" in output and "dollar_amt_df1" in output)


@pandas_version
def test_all_mismatch_ignore_matching_cols_no_cols_matching(spark_session):
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
    df1 = spark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = spark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = VSparkCompare(spark_session, df1, df2, "acct_id")

    output = compare.all_mismatch()
    assert output.shape[0] == 4
    assert output.shape[1] == 10

    assert (output.name_df1 != output.name_df2).values.sum() == 2
    assert (~(output.name_df1 != output.name_df2)).values.sum() == 2

    assert (output.dollar_amt_df1 != output.dollar_amt_df2).values.sum() == 1
    assert (~(output.dollar_amt_df1 != output.dollar_amt_df2)).values.sum() == 3

    assert (output.float_fld_df1 != output.float_fld_df2).values.sum() == 3
    assert (~(output.float_fld_df1 != output.float_fld_df2)).values.sum() == 1

    assert (output.date_fld_df1 != output.date_fld_df2).values.sum() == 4
    assert (~(output.date_fld_df1 != output.date_fld_df2)).values.sum() == 0


@pandas_version
@pytest.mark.parametrize(
    "column,expected",
    [
        ("base", 0),
        ("floats", 0.2),
        ("decimals", 0.1),
        ("null_floats", 0.1),
        ("strings", 0.1),
        ("mixed_strings", 1),
        ("infinity", np.inf),
    ],
)
def test_calculate_max_diff(spark_session, column, expected):
    MAX_DIFF_DF = spark_session.createDataFrame(
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
            "mixed_strings": ["1", "1", "1", "2", "some string"],
            "infinity": [1, 1, 1, 1, np.inf],
        }
    )
    assert np.isclose(
        calculate_max_diff(MAX_DIFF_DF["base"], MAX_DIFF_DF[column]), expected
    )


@pandas_version
def test_dupes_with_nulls_strings(spark_session):
    df1 = spark_session.createDataFrame(
        {
            "fld_1": [1, 2, 2, 3, 3, 4, 5, 5],
            "fld_2": ["A", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "fld_3": [1, 2, 2, 3, 3, 4, 5, 5],
        }
    )
    df2 = spark_session.createDataFrame(
        {
            "fld_1": [1, 2, 3, 4, 5],
            "fld_2": ["A", np.nan, np.nan, np.nan, np.nan],
            "fld_3": [1, 2, 3, 4, 5],
        }
    )
    comp = VSparkCompare(spark_session, df1, df2, join_columns=["fld_1", "fld_2"])
    assert comp.subset()


@pandas_version
def test_dupes_with_nulls_ints(spark_session):
    df1 = spark_session.createDataFrame(
        {
            "fld_1": [1, 2, 2, 3, 3, 4, 5, 5],
            "fld_2": [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "fld_3": [1, 2, 2, 3, 3, 4, 5, 5],
        }
    )
    df2 = spark_session.createDataFrame(
        {
            "fld_1": [1, 2, 3, 4, 5],
            "fld_2": [1, np.nan, np.nan, np.nan, np.nan],
            "fld_3": [1, 2, 3, 4, 5],
        }
    )
    comp = VSparkCompare(spark_session, df1, df2, join_columns=["fld_1", "fld_2"])
    assert comp.subset()


@pandas_version
def test_generate_id_within_group(spark_session):
    matrix = [
        (
            ps.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "__index": [1, 2, 3]}),
            ps.Series([0, 0, 0]),
        ),
        (
            ps.DataFrame(
                {
                    "a": ["a", "a", "DATACOMPY_NULL"],
                    "b": [1, 1, 2],
                    "__index": [1, 2, 3],
                }
            ),
            ps.Series([0, 1, 0]),
        ),
        (
            ps.DataFrame({"a": [-999, 2, 3], "b": [1, 2, 3], "__index": [1, 2, 3]}),
            ps.Series([0, 0, 0]),
        ),
        (
            ps.DataFrame(
                {"a": [1, np.nan, np.nan], "b": [1, 2, 2], "__index": [1, 2, 3]}
            ),
            ps.Series([0, 0, 1]),
        ),
        (
            ps.DataFrame(
                {"a": ["1", np.nan, np.nan], "b": ["1", "2", "2"], "__index": [1, 2, 3]}
            ),
            ps.Series([0, 0, 1]),
        ),
        (
            ps.DataFrame(
                {
                    "a": [datetime(2018, 1, 1), np.nan, np.nan],
                    "b": ["1", "2", "2"],
                    "__index": [1, 2, 3],
                }
            ),
            ps.Series([0, 0, 1]),
        ),
    ]
    for i in matrix:
        dataframe = i[0]
        expected = i[1]
        actual = (
            _generate_id_within_group(dataframe.to_spark(), ["a", "b"], "_temp_0")
            .orderBy("__index")
            .select("_temp_0")
            .to_pandas_on_spark()
        )
        assert (actual["_temp_0"] == expected).all()


@pandas_version
def test_lower(spark_session):
    """This function tests the toggle to use lower case for column names or not"""
    # should match
    df1 = spark_session.createDataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    df2 = spark_session.createDataFrame({"a": [1, 2, 3], "B": [0, 1, 2]})
    compare = VSparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert compare.matches()
    # should not match
    df1 = spark_session.createDataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    df2 = spark_session.createDataFrame({"a": [1, 2, 3], "B": [0, 1, 2]})
    compare = VSparkCompare(
        spark_session, df1, df2, join_columns=["a"], cast_column_names_lower=False
    )
    assert not compare.matches()

    # test join column
    # should match
    df1 = spark_session.createDataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    df2 = spark_session.createDataFrame({"A": [1, 2, 3], "B": [0, 1, 2]})
    compare = VSparkCompare(spark_session, df1, df2, join_columns=["a"])
    assert compare.matches()
    # should fail because "a" is not found in df2
    df1 = spark_session.createDataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    df2 = spark_session.createDataFrame({"A": [1, 2, 3], "B": [0, 1, 2]})
    expected_message = "df2 must have all columns from join_columns"
    with raises(ValueError, match=expected_message):
        compare = VSparkCompare(
            spark_session, df1, df2, join_columns=["a"], cast_column_names_lower=False
        )


@pandas_version
def test_integer_column_names(spark_session):
    """This function tests that integer column names would also work"""
    df1 = spark_session.createDataFrame({1: [1, 2, 3], 2: [0, 1, 2]})
    df2 = spark_session.createDataFrame({1: [1, 2, 3], 2: [0, 1, 2]})
    compare = VSparkCompare(spark_session, df1, df2, join_columns=[1])
    assert compare.matches()


@pandas_version
@mock.patch("datacompy.spark.render")
def test_save_html(mock_render):
    df1 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    compare = VSparkCompare(spark_session, df1, df2, join_columns=["a"])

    m = mock.mock_open()
    with mock.patch("datacompy.spark.open", m, create=True):
        # assert without HTML call
        compare.report()
        assert mock_render.call_count == 4
        m.assert_not_called()

    mock_render.reset_mock()
    m = mock.mock_open()
    with mock.patch("datacompy.spark.open", m, create=True):
        # assert with HTML call
        compare.report(html_file="test.html")
        assert mock_render.call_count == 4
        m.assert_called_with("test.html", "w")


def test_pandas_version(spark_session):
    expected_message = "It seems like you are running Pandas 2+. Please note that Pandas 2+ will only be supported in Spark 4+. See: https://issues.apache.org/jira/browse/SPARK-44101. If you need to use Spark DataFrame with Pandas 2+ then consider using Fugue otherwise downgrade to Pandas 1.5.3"
    df1 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df2 = spark_session.createDataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    with mock.patch("pandas.__version__", "2.0.0"):
        with raises(Exception, match=re.escape(expected_message)):
            VSparkCompare(spark_session, df1, df2, join_columns=["a"])

    with mock.patch("pandas.__version__", "1.5.3"):
        VSparkCompare(spark_session, df1, df2, join_columns=["a"])


@pandas_version
def test_unicode_columns(spark_session):
    df1 = spark_session.createDataFrame(
        [
            {"a": 1, "例": 2, "予測対象日": "test"},
            {"a": 1, "例": 3, "予測対象日": "test"},
        ]
    )
    df2 = spark_session.createDataFrame(
        [
            {"a": 1, "例": 2, "予測対象日": "test"},
            {"a": 1, "例": 3, "予測対象日": "test"},
        ]
    )
    compare = VSparkCompare(spark_session, df1, df2, join_columns=["例"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()
