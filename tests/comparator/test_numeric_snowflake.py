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

import pytest

pytest.importorskip("snowflake.snowpark")

import snowflake.snowpark as sf
from datacompy.comparator.numeric import SnowflakeNumericComparator

# tests for SnowflakeNumericComparator


def test_snowflake_numeric_comparator_exact_match(snowflake_session):
    comparator = SnowflakeNumericComparator()
    df = snowflake_session.createDataFrame(
        [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_numeric_comparator_approximate_match(snowflake_session):
    comparator = SnowflakeNumericComparator()
    df = snowflake_session.createDataFrame(
        [(1.0, 1.001), (2.0, 2.002), (3.0, 3.003)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df,
        col1="col1",
        col2="col2",
        col_match="col_match",
        rtol=1e-3,
        atol=1e-3,
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_numeric_comparator_type_casting(snowflake_session):
    comparator = SnowflakeNumericComparator()
    df = snowflake_session.createDataFrame(
        [(1, 1.0), (2, 2.0), (3, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_numeric_comparator_nan_handling(snowflake_session):
    comparator = SnowflakeNumericComparator()
    df = snowflake_session.createDataFrame(
        [
            (1.0, 1.0),
            (float("nan"), float("nan")),
            (float("nan"), 1.0),
            (1.0, float("nan")),
            (3.0, 3.0),
        ],
        ["col1", "col2"],
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_numeric_comparator_mismatch(snowflake_session):
    comparator = SnowflakeNumericComparator()
    df = snowflake_session.createDataFrame(
        [(1.0, 1.0), (2.0, 2.5), (3.0, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_numeric_comparator_symmetric_tolerance(snowflake_session):
    """Tolerance comparison is symmetric (math.isclose), unlike numpy.isclose."""
    comparator = SnowflakeNumericComparator()
    df = snowflake_session.createDataFrame(
        [(90.0, 100.0), (100.0, 90.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df,
        col1="col1",
        col2="col2",
        col_match="col_match",
        rtol=0.1,
        atol=0,
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_numeric_comparator_max_not_sum_tolerances(snowflake_session):
    """atol and rtol are combined with max, not addition."""
    comparator = SnowflakeNumericComparator()
    df = snowflake_session.createDataFrame([(10.0, 11.0)], ["col1", "col2"])
    result = comparator.compare(
        dataframe=df,
        col1="col1",
        col2="col2",
        col_match="col_match",
        rtol=0.05,
        atol=0.5,
    )
    assert result.select(["col_match"]).collect() == [sf.Row(COL_MATCH=False)]


def test_snowflake_numeric_comparator_atol_dominates(snowflake_session):
    """Near zero rtol contributes nothing, so atol alone sets the bound."""
    comparator = SnowflakeNumericComparator()
    df = snowflake_session.createDataFrame([(0.0, 0.4), (0.0, 0.6)], ["col1", "col2"])
    result = comparator.compare(
        dataframe=df,
        col1="col1",
        col2="col2",
        col_match="col_match",
        rtol=0.05,
        atol=0.5,
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=False),
    ]


def test_snowflake_numeric_comparator_error_handling(snowflake_session):
    comparator = SnowflakeNumericComparator()
    df = snowflake_session.createDataFrame(
        [("a", "x"), ("b", "y"), ("c", "z")], ["col1", "col2"]
    )  # Invalid type for numeric comparison
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result is None
