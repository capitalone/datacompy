import numpy as np
import pandas as pd
import polars as pl
import snowflake.snowpark as sf
from datacompy.comparator.array import (
    PandasArrayLikeComparator,
    PolarsArrayLikeComparator,
    SnowflakeArrayLikeComparator,
    SparkArrayLikeComparator,
)


# Pandas
def test_pandas_compare_equal_arrays():
    # Setup
    col1 = pd.Series([np.array([1, 2]), np.array([3, 4])])
    col2 = pd.Series([np.array([1, 2]), np.array([3, 4])])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pd.Series)
    assert result.all()


def test_pandas_compare_equal_lists():
    # Setup
    col1 = pd.Series([[1, 2], [3, 4]])
    col2 = pd.Series([[1, 2], [3, 4]])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pd.Series)
    assert result.all()


def test_pandas_compare_unequal_arrays():
    # Setup
    col1 = pd.Series([np.array([1, 2]), np.array([3, 4])])
    col2 = pd.Series([np.array([1, 2]), np.array([3, 5])])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pd.Series)
    assert not result.all()
    assert result.tolist() == [True, False]


def test_pandas_compare_unequal_lists():
    # Setup
    col1 = pd.Series([[1, 2], [3, 4]])
    col2 = pd.Series([[1, 2], [3, 5]])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pd.Series)
    assert not result.all()
    assert result.tolist() == [True, False]


def test_pandas_compare_with_nans():
    # Setup
    col1 = pd.Series([np.array([1, np.nan]), np.array([3, 4])])
    col2 = pd.Series([np.array([1, np.nan]), np.array([3, 4])])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pd.Series)
    assert result.all()


def test_pandas_compare_different_shapes():
    # Setup
    col1 = pd.Series([np.array([1, 2]), np.array([3, 4]), np.array([3, 4])])
    col2 = pd.Series([np.array([1, 2, 3]), np.array([3, 4, 5])])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert result is None


def test_pandas_compare_non_array_like():
    # integers
    col1 = pd.Series([1, 2])
    col2 = pd.Series([1, 2])
    comparator = PandasArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None

    # floats
    col1 = pd.Series([1.0, 2.0])
    col2 = pd.Series([1.0, 2.0])
    comparator = PandasArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None

    # dict
    col1 = pd.Series([{"a": 1}, {"b": 2}])
    col2 = pd.Series([{"a": 1}, {"b": 2}])
    comparator = PandasArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None


# Polars
def test_polars_compare_equal_arrays():
    # Setup
    col1 = pl.Series([[1, 2], [3, 4]])
    col2 = pl.Series([[1, 2], [3, 6]])
    comparator = PolarsArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pl.Series)
    assert result.to_list() == [True, False]


def test_polars_compare_unequal_arrays():
    # Setup
    col1 = pl.Series([[1, 2], [3, 4]])
    col2 = pl.Series([[1, 2], [3, 5]])
    comparator = PolarsArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pl.Series)
    assert result.to_list() == [True, False]


def test_polars_compare_with_nulls():
    # Setup
    col1 = pl.Series([[1, None], [3, 4]])
    col2 = pl.Series([[1, None], [3, 4]])
    comparator = PolarsArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pl.Series)
    assert result.to_list() == [True, True]


def test_polars_compare_different_shapes():
    # Setup
    col1 = pl.Series([[1, 2], [3, 4], [5, 6]])
    col2 = pl.Series([[1, 2], [3, 4]])
    comparator = PolarsArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert result is None


def test_polars_compare_non_array():
    # integers
    col1 = pl.Series([1, 2, 3])
    col2 = pl.Series([1, 2, 3])
    comparator = PolarsArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None

    # floats
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.0, 2.0, 3.0])
    comparator = PolarsArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None

    # dicts
    col1 = pl.Series([{"a": 1}, {"b": 2}])
    col2 = pl.Series([{"a": 1}, {"b": 2}])
    comparator = PolarsArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None


# PySpark
def test_spark_compare_equal_arrays(spark_session):
    # Setup
    data = [([1, 2], [1, 2]), ([3, 4], [3, 4])]
    df = spark_session.createDataFrame(data, ["col1", "col2"])
    comparator = SparkArrayLikeComparator()

    # Execute
    result = comparator.compare(df, "col1", "col2", "match")

    # Assert
    assert result is not None
    matches = result.select("match").collect()
    assert all(row.match for row in matches)


def test_spark_compare_unequal_arrays(spark_session):
    # Setup
    data = [([1, 2], [1, 2]), ([3, 4], [3, 5])]
    df = spark_session.createDataFrame(data, ["col1", "col2"])
    comparator = SparkArrayLikeComparator()

    # Execute
    result = comparator.compare(df, "col1", "col2", "match")

    # Assert
    assert result is not None
    matches = result.select("match").collect()
    assert matches[0].match is True
    assert matches[1].match is False


def test_spark_compare_with_nulls(spark_session):
    # Setup
    data = [([1, None], [1, None]), ([3, 4], [3, 4])]
    df = spark_session.createDataFrame(data, ["col1", "col2"])
    comparator = SparkArrayLikeComparator()

    # Execute
    result = comparator.compare(df, "col1", "col2", "match")

    # Assert
    assert result is not None
    matches = result.select("match").collect()
    assert all(row.match for row in matches)


def test_spark_compare_non_array(spark_session):
    # integers
    data = [(1, 1), (2, 2)]
    df = spark_session.createDataFrame(data, ["col1", "col2"])
    comparator = SparkArrayLikeComparator()
    result = comparator.compare(df, "col1", "col2", "match")
    assert result is None

    # floats
    data = [(1.0, 1.0), (2.0, 2.0)]
    df = spark_session.createDataFrame(data, ["col1", "col2"])
    comparator = SparkArrayLikeComparator()
    result = comparator.compare(df, "col1", "col2", "match")
    assert result is None

    # dicts
    data = [({"a": 1}, {"a": 1}), ({"b": 2}, {"b": 2})]
    df = spark_session.createDataFrame(data, ["col1", "col2"])
    comparator = SparkArrayLikeComparator()
    result = comparator.compare(df, "col1", "col2", "match")
    assert result is None


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
