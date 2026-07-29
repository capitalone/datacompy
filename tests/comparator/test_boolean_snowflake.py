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

from decimal import Decimal

import pytest

pytest.importorskip("snowflake.snowpark")

import snowflake.snowpark as sf
from datacompy.comparator.boolean import SnowflakeBooleanComparator
from datacompy.snowflake import columns_equal
from snowflake.snowpark.types import (
    BooleanType,
    DecimalType,
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


def test_snowflake_boolean_comparator_null_handling(
    snowflake_session, requires_live_snowflake_session
):
    """Two nulls are equal; a null against a value is not.

    Local testing mode returns ``True`` for every ``eqNullSafe`` row, so this
    can only be verified against a real session.
    """
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


BOOLEAN_NUMERIC_SCHEMA = StructType(
    [
        StructField("col1", BooleanType()),
        StructField("col2", IntegerType()),
    ]
)
NUMERIC_BOOLEAN_SCHEMA = StructType(
    [
        StructField("col1", IntegerType()),
        StructField("col2", BooleanType()),
    ]
)


def test_snowflake_boolean_comparator_against_numeric(snowflake_session):
    """True matches exactly 1 and False exactly 0, regardless of column order."""
    comparator = SnowflakeBooleanComparator()
    rows = [(True, 1), (False, 0), (True, 0), (False, 1), (True, 2), (None, None)]
    expected = [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=True),
    ]

    forward = comparator.compare(
        dataframe=snowflake_session.createDataFrame(
            rows, schema=BOOLEAN_NUMERIC_SCHEMA
        ),
        col1="col1",
        col2="col2",
        col_match="col_match",
    )
    reverse = comparator.compare(
        dataframe=snowflake_session.createDataFrame(
            [(n, b) for b, n in rows], schema=NUMERIC_BOOLEAN_SCHEMA
        ),
        col1="col1",
        col2="col2",
        col_match="col_match",
    )

    assert forward.select(["col_match"]).collect() == expected
    assert reverse.select(["col_match"]).collect() == expected


def test_snowflake_boolean_comparator_numeric_null_handling(snowflake_session):
    """A null against a value is a mismatch in either direction."""
    comparator = SnowflakeBooleanComparator()
    df = snowflake_session.createDataFrame(
        [(None, 1), (True, None), (None, None)], schema=BOOLEAN_NUMERIC_SCHEMA
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_boolean_comparator_decimal_preserves_precision(
    snowflake_session, requires_live_snowflake_session
):
    """A decimal just past double precision must not be rounded into a match.

    Local testing mode truncates ``1.000000000000000001`` to ``1`` when the
    DataFrame is created, destroying the case under test.
    """
    comparator = SnowflakeBooleanComparator()
    df = snowflake_session.createDataFrame(
        [
            (True, Decimal("1.000000000000000001")),
            (True, Decimal("1.000000000000000000")),
            (False, Decimal("0.000000000000000001")),
            (False, Decimal("0.000000000000000000")),
        ],
        schema=StructType(
            [
                StructField("col1", BooleanType()),
                StructField("col2", DecimalType(38, 18)),
            ]
        ),
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=True),
    ]


def test_snowflake_columns_equal_dispatches_boolean_against_numeric(snowflake_session):
    """Exercise the pipeline, not just the comparator in isolation."""
    df = snowflake_session.createDataFrame(
        [(True, 1), (True, 2), (False, 0)], schema=BOOLEAN_NUMERIC_SCHEMA
    )
    result = columns_equal(df, "col1", "col2", "col_match")
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=False),
        sf.Row(COL_MATCH=True),
    ]


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
