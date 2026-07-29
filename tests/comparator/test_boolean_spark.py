#
# Copyright 2026 Capital One Services, LLC
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

"""Regression tests for Spark Boolean comparisons."""

from decimal import Decimal

import pytest

pytest.importorskip("pyspark")

from datacompy.comparator import SparkBooleanComparator
from datacompy.spark import SparkSQLCompare, columns_equal
from pyspark.sql.functions import to_date


def _evaluate(dataframe, column):
    """Collect a Column expression into a plain Python list."""
    return [row[0] for row in dataframe.select(column.alias("r")).collect()]


def test_boolean_columns_equal(spark_session):
    df = spark_session.createDataFrame(
        [(True, True), (False, True), (True, False), (False, False)],
        "b1 boolean, b2 boolean",
    )

    assert _evaluate(df, columns_equal(df, "b1", "b2")) == [True, False, False, True]


def test_boolean_columns_equal_with_nulls(spark_session):
    df = spark_session.createDataFrame(
        [(True, True), (None, None), (None, False), (True, None)],
        "b1 boolean, b2 boolean",
    )

    assert _evaluate(df, columns_equal(df, "b1", "b2")) == [True, True, False, False]


def test_all_null_boolean_columns_equal(spark_session):
    df = spark_session.createDataFrame(
        [(None, None), (None, None)], "b1 boolean, b2 boolean"
    )

    assert _evaluate(df, columns_equal(df, "b1", "b2")) == [True, True]


def test_boolean_and_numeric_columns_equal(spark_session):
    """Boolean/numeric avoids implicit coercion so it works under ANSI mode too."""
    df = spark_session.createDataFrame(
        [(True, 1), (False, 0), (True, 0), (True, 2), (None, None)],
        "b boolean, i int",
    )

    expected = [True, True, False, False, True]

    assert _evaluate(df, columns_equal(df, "b", "i")) == expected
    assert _evaluate(df, columns_equal(df, "i", "b")) == expected


def test_boolean_and_decimal_columns_preserve_precision(spark_session):
    """A decimal just past ``double`` precision must not be rounded into a match."""
    df = spark_session.createDataFrame(
        [
            (True, Decimal("1.000000000000000001")),
            (True, Decimal("1.000000000000000000")),
            (False, Decimal("0.000000000000000001")),
            (False, Decimal("0.000000000000000000")),
            (None, None),
        ],
        "b boolean, d decimal(38,18)",
    )

    expected = [False, True, False, True, True]

    assert _evaluate(df, columns_equal(df, "b", "d")) == expected
    assert _evaluate(df, columns_equal(df, "d", "b")) == expected


def test_boolean_and_bigint_columns_preserve_precision(spark_session):
    """Large integers beyond ``double``'s 53-bit mantissa must not collapse."""
    df = spark_session.createDataFrame(
        [(True, 1), (True, 9007199254740993), (False, 0)],
        "b boolean, l bigint",
    )

    expected = [True, False, True]

    assert _evaluate(df, columns_equal(df, "b", "l")) == expected
    assert _evaluate(df, columns_equal(df, "l", "b")) == expected


def test_boolean_and_double_columns_equal(spark_session):
    df = spark_session.createDataFrame(
        [(True, 1.0), (False, 0.0), (False, 1.0), (None, None)], "b boolean, d double"
    )

    assert _evaluate(df, columns_equal(df, "b", "d")) == [True, True, False, True]


def test_boolean_comparator_ignores_non_boolean_columns(spark_session):
    df = spark_session.createDataFrame([(1, 2)], "i1 int, i2 int")

    assert SparkBooleanComparator().compare(df, "i1", "i2") is None


def test_boolean_comparator_declines_boolean_against_string(spark_session):
    """Spark would coerce the string to Boolean; other backends do not match here."""
    df = spark_session.createDataFrame([(True, "true")], "b boolean, s string")

    assert SparkBooleanComparator().compare(df, "b", "s") is None
    assert _evaluate(df, columns_equal(df, "b", "s")) == [False]


def test_boolean_comparator_declines_incomparable_dtypes(spark_session):
    """Claiming these would build a Column that raises at analysis time."""
    df = spark_session.createDataFrame(
        [(True, [1, 2], "2020-01-01")], "b boolean, arr array<int>, ds string"
    ).withColumn("dt", to_date("ds"))

    assert SparkBooleanComparator().compare(df, "b", "arr") is None
    assert SparkBooleanComparator().compare(df, "b", "dt") is None

    # The pipeline resolves to "not equal" rather than raising.
    assert _evaluate(df, columns_equal(df, "b", "arr")) == [False]
    assert _evaluate(df, columns_equal(df, "b", "dt")) == [False]


def test_spark_compare_matches_identical_boolean_values(spark_session):
    df1 = spark_session.createDataFrame([(1, True), (2, False)], "id int, flag boolean")
    df2 = spark_session.createDataFrame([(1, True), (2, False)], "id int, flag boolean")

    comparison = SparkSQLCompare(spark_session, df1, df2, join_columns="id")

    assert comparison.matches()


def test_spark_compare_detects_boolean_mismatch(spark_session):
    df1 = spark_session.createDataFrame([(1, True), (2, False)], "id int, flag boolean")
    df2 = spark_session.createDataFrame(
        [(1, False), (2, False)], "id int, flag boolean"
    )

    comparison = SparkSQLCompare(spark_session, df1, df2, join_columns="id")

    assert not comparison.matches()
    assert comparison.count_matching_rows() == 1


def test_spark_compare_boolean_with_nulls(spark_session):
    df1 = spark_session.createDataFrame(
        [(1, True), (2, None), (3, False)], "id int, flag boolean"
    )
    df2 = spark_session.createDataFrame(
        [(1, True), (2, None), (3, False)], "id int, flag boolean"
    )

    comparison = SparkSQLCompare(spark_session, df1, df2, join_columns="id")

    assert comparison.matches()
    assert comparison.count_matching_rows() == 3


def test_spark_compare_boolean_null_against_value_is_mismatch(spark_session):
    df1 = spark_session.createDataFrame([(1, True), (2, None)], "id int, flag boolean")
    df2 = spark_session.createDataFrame([(1, True), (2, False)], "id int, flag boolean")

    comparison = SparkSQLCompare(spark_session, df1, df2, join_columns="id")

    assert not comparison.matches()
    assert comparison.count_matching_rows() == 1


def test_spark_compare_boolean_with_non_overlapping_join_keys(spark_session):
    df1 = spark_session.createDataFrame(
        [(1, True), (2, False), (3, True)], "id int, flag boolean"
    )
    df2 = spark_session.createDataFrame(
        [(1, True), (2, False), (4, True)], "id int, flag boolean"
    )

    comparison = SparkSQLCompare(spark_session, df1, df2, join_columns="id")

    assert comparison.count_matching_rows() == 2
