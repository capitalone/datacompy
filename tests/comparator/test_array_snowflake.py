import pytest

pytest.importorskip("snowflake.snowpark")

import snowflake.snowpark as sf
from datacompy.comparator.array import SnowflakeArrayLikeComparator


def test_snowflake_compare_equal_arrays(snowflake_session):
    # Setup
    data = [([1, 2], [1, 2]), ([3, 4], [3, 4])]
    df = snowflake_session.create_dataframe(data, schema=["col1", "col2"])
    comparator = SnowflakeArrayLikeComparator()

    # Execute
    result = comparator.compare(df, "col1", "col2", "match")

    # Assert
    assert result is not None
    matches = result.select("match").collect()
    assert all(row for row in matches)


def test_snowflake_compare_unequal_arrays(snowflake_session):
    # Setup
    data = [([1, 2], [1, 2]), ([3, 4], [3, 5])]
    df = snowflake_session.create_dataframe(data, schema=["col1", "col2"])
    comparator = SnowflakeArrayLikeComparator()

    # Execute
    result = comparator.compare(df, "col1", "col2", "match")

    # Assert
    assert result is not None
    matches = result.select("match").collect()
    assert matches == [
        sf.Row(match=True),
        sf.Row(match=False),
    ]


def test_snowflake_compare_with_nulls(snowflake_session):
    # Setup
    data = [([1, None], [1, None]), ([3, 4], [3, 4])]
    df = snowflake_session.create_dataframe(data, schema=["col1", "col2"])
    comparator = SnowflakeArrayLikeComparator()

    # Execute
    result = comparator.compare(df, "col1", "col2", "match")

    # Assert
    assert result is not None
    matches = result.select("match").collect()
    assert all(row for row in matches)


def test_snowflake_compare_non_array(snowflake_session):
    # integers
    data = [(1, 1), (2, 2)]
    df = snowflake_session.createDataFrame(data, ["col1", "col2"])
    comparator = SnowflakeArrayLikeComparator()
    result = comparator.compare(df, "col1", "col2", "match")
    assert result is None

    # floats
    data = [(1.0, 1.0), (2.0, 2.0)]
    df = snowflake_session.createDataFrame(data, ["col1", "col2"])
    comparator = SnowflakeArrayLikeComparator()
    result = comparator.compare(df, "col1", "col2", "match")
    assert result is None

    # dicts
    data = [({"a": 1}, {"a": 1}), ({"b": 2}, {"b": 2})]
    df = snowflake_session.createDataFrame(data, ["col1", "col2"])
    comparator = SnowflakeArrayLikeComparator()
    result = comparator.compare(df, "col1", "col2", "match")
    assert result is None
