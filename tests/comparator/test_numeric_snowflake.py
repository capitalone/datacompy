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
    comparator = SnowflakeNumericComparator(rtol=1e-3, atol=1e-3)
    df = snowflake_session.createDataFrame(
        [(1.0, 1.001), (2.0, 2.002), (3.0, 3.003)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
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
        [(1.0, 1.0), (float("nan"), float("nan")), (3.0, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        sf.Row(COL_MATCH=True),
        sf.Row(COL_MATCH=True),
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


def test_snowflake_numeric_comparator_error_handling(snowflake_session):
    comparator = SnowflakeNumericComparator()
    df = snowflake_session.createDataFrame(
        [("a", "x"), ("b", "y"), ("c", "z")], ["col1", "col2"]
    )  # Invalid type for numeric comparison
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result is None
