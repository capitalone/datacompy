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

"""Regression tests for Snowflake Boolean comparisons."""

from decimal import Decimal

import pytest

pytest.importorskip("snowflake.snowpark")

from datacompy.comparator import SnowflakeBooleanComparator
from datacompy.snowflake import SnowflakeCompare, columns_equal


def test_snowflake_boolean_columns_equal_with_nulls(snowflake_session):
    dataframe = snowflake_session.createDataFrame(
        [
            (True, True, True),
            (False, True, False),
            (None, None, True),
            (None, False, False),
        ],
        schema=["LEFT_VALUE", "RIGHT_VALUE", "EXPECTED"],
    )

    actual = (
        columns_equal(
            dataframe,
            "LEFT_VALUE",
            "RIGHT_VALUE",
            "ACTUAL",
        )
        .select("ACTUAL")
        .toPandas()["ACTUAL"]
    )
    expected = dataframe.select("EXPECTED").toPandas()["EXPECTED"]

    assert actual.tolist() == expected.tolist()


def test_snowflake_boolean_numeric_cross_type_in_both_directions(
    snowflake_session,
):
    dataframe = snowflake_session.createDataFrame(
        [
            (True, 1, True),
            (False, 0, True),
            (True, 0, False),
            (False, 1, False),
        ],
        schema=["BOOLEAN_VALUE", "INTEGER_VALUE", "EXPECTED"],
    )

    forward = (
        columns_equal(
            dataframe,
            "BOOLEAN_VALUE",
            "INTEGER_VALUE",
            "FORWARD_MATCH",
        )
        .select("FORWARD_MATCH")
        .toPandas()["FORWARD_MATCH"]
    )
    reverse = (
        columns_equal(
            dataframe,
            "INTEGER_VALUE",
            "BOOLEAN_VALUE",
            "REVERSE_MATCH",
        )
        .select("REVERSE_MATCH")
        .toPandas()["REVERSE_MATCH"]
    )
    expected = dataframe.select("EXPECTED").toPandas()["EXPECTED"]

    assert forward.tolist() == expected.tolist()
    assert reverse.tolist() == expected.tolist()


def test_snowflake_boolean_decimal_comparison_preserves_precision(
    snowflake_session,
):
    dataframe = snowflake_session.createDataFrame(
        [
            (True, Decimal("1.000000000000000001"), False),
            (False, Decimal("0.000000000000000001"), False),
            (True, Decimal("1.000000000000000000"), True),
            (False, Decimal("0.000000000000000000"), True),
            (None, None, True),
            (None, Decimal("0.000000000000000000"), False),
        ],
        schema=["BOOLEAN_VALUE", "DECIMAL_VALUE", "EXPECTED"],
    )

    forward = (
        columns_equal(
            dataframe,
            "BOOLEAN_VALUE",
            "DECIMAL_VALUE",
            "FORWARD_MATCH",
        )
        .select("FORWARD_MATCH")
        .toPandas()["FORWARD_MATCH"]
    )
    reverse = (
        columns_equal(
            dataframe,
            "DECIMAL_VALUE",
            "BOOLEAN_VALUE",
            "REVERSE_MATCH",
        )
        .select("REVERSE_MATCH")
        .toPandas()["REVERSE_MATCH"]
    )
    expected = dataframe.select("EXPECTED").toPandas()["EXPECTED"]

    assert forward.tolist() == expected.tolist()
    assert reverse.tolist() == expected.tolist()


def test_snowflake_boolean_comparator_ignores_non_boolean_columns(
    snowflake_session,
):
    dataframe = snowflake_session.createDataFrame(
        [(1, 1)], schema=["LEFT_VALUE", "RIGHT_VALUE"]
    )

    assert (
        SnowflakeBooleanComparator().compare(
            dataframe,
            "LEFT_VALUE",
            "RIGHT_VALUE",
            "MATCH",
        )
        is None
    )


def test_snowflake_compare_matches_identical_boolean_values(snowflake_session):
    left = snowflake_session.createDataFrame(
        [(1, True), (2, False)], schema=["ID", "FLAG"]
    )
    right = snowflake_session.createDataFrame(
        [(1, True), (2, False)], schema=["ID", "FLAG"]
    )

    assert SnowflakeCompare(
        snowflake_session,
        left,
        right,
        join_columns="ID",
    ).matches()
