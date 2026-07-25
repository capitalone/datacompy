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
from datacompy.comparator.boolean import SnowflakeBooleanComparator
from datacompy.snowflake import columns_equal
from snowflake.snowpark.types import (
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# tests for SnowflakeBooleanComparator

BOOLEAN_SCHEMA = StructType(
    [
        StructField("col1", BooleanType()),
        StructField("col2", BooleanType()),
    ]
)


def test_snowflake_boolean_comparator_exact_match(snowflake_session):
    comparator = SnowflakeBooleanComparator()
    df = snowflake_session.createDataFrame(
        [(True, True), (False, False)], schema=BOOLEAN_SCHEMA
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_boolean_comparator_mismatch(snowflake_session):
    comparator = SnowflakeBooleanComparator()
    df = snowflake_session.createDataFrame(
        [(True, False), (False, True)], schema=BOOLEAN_SCHEMA
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=False),
    ]


def test_snowflake_boolean_comparator_null_handling(snowflake_session):
    """Two nulls are equal; a null against a value is not."""
    comparator = SnowflakeBooleanComparator()
    df = snowflake_session.createDataFrame(
        [(None, None), (None, False), (True, None), (True, True)],
        schema=BOOLEAN_SCHEMA,
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_boolean_comparator_all_null(snowflake_session):
    comparator = SnowflakeBooleanComparator()
    df = snowflake_session.createDataFrame(
        [(None, None), (None, None)], schema=BOOLEAN_SCHEMA
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_boolean_comparator_ignores_non_boolean(snowflake_session):
    comparator = SnowflakeBooleanComparator()
    df = snowflake_session.createDataFrame([(1, 2)], ["col1", "col2"])
    assert (
        comparator.compare(
            dataframe=df, col1="col1", col2="col2", col_match="col_match"
        )
        is None
    )


def test_snowflake_boolean_comparator_declines_boolean_against_numeric(
    snowflake_session,
):
    """Boolean/numeric is intentionally left to fall through; see the docstring."""
    comparator = SnowflakeBooleanComparator()
    df = snowflake_session.createDataFrame(
        [(True, 1)],
        schema=StructType(
            [StructField("col1", BooleanType()), StructField("col2", IntegerType())]
        ),
    )
    assert (
        comparator.compare(
            dataframe=df, col1="col1", col2="col2", col_match="col_match"
        )
        is None
    )


def test_snowflake_boolean_comparator_declines_boolean_against_string(
    snowflake_session,
):
    comparator = SnowflakeBooleanComparator()
    df = snowflake_session.createDataFrame(
        [(True, "true")],
        schema=StructType(
            [StructField("col1", BooleanType()), StructField("col2", StringType())]
        ),
    )
    assert (
        comparator.compare(
            dataframe=df, col1="col1", col2="col2", col_match="col_match"
        )
        is None
    )


def test_snowflake_columns_equal_dispatches_to_boolean_comparator(snowflake_session):
    """Exercise the pipeline, not just the comparator in isolation."""
    df = snowflake_session.createDataFrame(
        [(True, True), (True, False), (None, None)], schema=BOOLEAN_SCHEMA
    )
    result = columns_equal(df, "col1", "col2", "col_match")
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=True),
    ]
