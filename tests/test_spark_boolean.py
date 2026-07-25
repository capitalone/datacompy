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


def test_spark_boolean_columns_equal_with_nulls(spark_session):
    dataframe = spark_session.createDataFrame(
        [
            (True, True, True),
            (False, True, False),
            (None, None, True),
            (None, False, False),
        ],
        ["left", "right", "expected"],
    )

    actual = (
        dataframe.withColumn("actual", columns_equal(dataframe, "left", "right"))
        .select("actual")
        .toPandas()["actual"]
    )
    expected = dataframe.select("expected").toPandas()["expected"]

    assert actual.tolist() == expected.tolist()


def test_spark_boolean_numeric_cross_type_in_both_directions(spark_session):
    dataframe = spark_session.createDataFrame(
        [
            (True, 1, True),
            (False, 0, True),
            (True, 0, False),
            (False, 1, False),
        ],
        ["boolean_value", "integer_value", "expected"],
    )

    forward = (
        dataframe.withColumn(
            "actual", columns_equal(dataframe, "boolean_value", "integer_value")
        )
        .select("actual")
        .toPandas()["actual"]
    )
    reverse = (
        dataframe.withColumn(
            "actual", columns_equal(dataframe, "integer_value", "boolean_value")
        )
        .select("actual")
        .toPandas()["actual"]
    )
    expected = dataframe.select("expected").toPandas()["expected"]

    assert forward.tolist() == expected.tolist()
    assert reverse.tolist() == expected.tolist()


def test_spark_boolean_decimal_comparison_preserves_precision(spark_session):
    dataframe = spark_session.createDataFrame(
        [
            (True, Decimal("1.000000000000000001"), False),
            (False, Decimal("0.000000000000000001"), False),
            (True, Decimal("1.000000000000000000"), True),
            (False, Decimal("0.000000000000000000"), True),
            (None, None, True),
            (None, Decimal("0.000000000000000000"), False),
        ],
        ["boolean_value", "decimal_value", "expected"],
    )

    forward = (
        dataframe.withColumn(
            "actual", columns_equal(dataframe, "boolean_value", "decimal_value")
        )
        .select("actual")
        .toPandas()["actual"]
    )
    reverse = (
        dataframe.withColumn(
            "actual", columns_equal(dataframe, "decimal_value", "boolean_value")
        )
        .select("actual")
        .toPandas()["actual"]
    )
    expected = dataframe.select("expected").toPandas()["expected"]

    assert forward.tolist() == expected.tolist()
    assert reverse.tolist() == expected.tolist()


def test_spark_boolean_comparator_ignores_non_boolean_columns(spark_session):
    dataframe = spark_session.createDataFrame([(1, 1)], ["left", "right"])

    assert SparkBooleanComparator().compare(dataframe, "left", "right") is None


def test_spark_compare_matches_identical_boolean_values(spark_session):
    left = spark_session.createDataFrame([(1, True), (2, False)], ["id", "flag"])
    right = spark_session.createDataFrame([(1, True), (2, False)], ["id", "flag"])

    assert SparkSQLCompare(spark_session, left, right, join_columns="id").matches()
