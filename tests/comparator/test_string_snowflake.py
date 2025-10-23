import pytest

pytest.importorskip("snowflake.snowpark")

import snowflake.snowpark as sf
from datacompy.comparator.string import SnowflakeStringComparator

# tests for SnowflakeStringComparator


def test_snowflake_string_comparator_exact_match(snowflake_session):
    comparator = SnowflakeStringComparator()
    df = snowflake_session.createDataFrame(
        [("a", "a"), ("b", "b"), ("c", "c")], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(col_match=True),
        sf.Row(col_match=True),
        sf.Row(col_match=True),
    ]


def test_snowflake_string_comparator_case_space_insensitivity(snowflake_session):
    df = snowflake_session.createDataFrame(
        [("a", " a"), ("b", "   B  "), ("c", "C")], ["col1", "col2"]
    )

    comparator = SnowflakeStringComparator(ignore_case=True, ignore_space=True)
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(col_match=True),
        sf.Row(col_match=True),
        sf.Row(col_match=True),
    ]

    comparator = SnowflakeStringComparator(ignore_case=True, ignore_space=False)
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(col_match=False),
        sf.Row(col_match=False),
        sf.Row(col_match=True),
    ]

    comparator = SnowflakeStringComparator(ignore_case=False, ignore_space=True)
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(col_match=True),
        sf.Row(col_match=False),
        sf.Row(col_match=False),
    ]

    comparator = SnowflakeStringComparator(ignore_case=False, ignore_space=False)
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(col_match=False),
        sf.Row(col_match=False),
        sf.Row(col_match=False),
    ]


def test_snowflake_string_comparator_null_handling(snowflake_session):
    comparator = SnowflakeStringComparator()
    df = snowflake_session.createDataFrame(
        [("a", "a"), (None, None), ("c", "c")], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(col_match=True),
        sf.Row(col_match=True),
        sf.Row(col_match=True),
    ]


def test_snowflake_string_comparator_error_handling(snowflake_session):
    comparator = SnowflakeStringComparator()
    df = snowflake_session.createDataFrame(
        [(1, 2), (3, 4), (5, 6)], ["col1", "col2"]
    )  # Invalid type for string comparison
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result is None
