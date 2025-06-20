import pandas as pd
import polars as pl
import pyspark.sql as ps
import pytest
import snowflake.snowpark as sf
from datacompy.comparator.string import (
    PandasStringComparator,
    PolarsStringComparator,
    SnowflakeStringComparator,
    SparkStringComparator,
)
from tests.comparator.snowflake_mocks import *  # noqa: F403


# tests for PolarsStringComparator
def test_polars_string_comparator_exact_match():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c"])
    col2 = pl.Series(["a", "b", "c"])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_string_comparator_case_space_insensitivity():
    comparator = PolarsStringComparator(ignore_case=True, ignore_space=True)
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]

    comparator = PolarsStringComparator(ignore_case=True, ignore_space=False)
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, False, False]

    comparator = PolarsStringComparator(ignore_case=False, ignore_space=True)
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [False, True, False]

    comparator = PolarsStringComparator(ignore_case=False, ignore_space=False)
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [False, False, False]


def test_polars_string_comparator_none_handling():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", None])
    col2 = pl.Series(["a", "b", None])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_string_comparator_mismatch():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c"])
    col2 = pl.Series(["a", "b", "d"])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, False]


def test_polars_string_comparator_error_handling():
    comparator = PolarsStringComparator()
    col1 = pl.Series([1, 2, 3])  # Invalid type for string comparison
    col2 = pl.Series([4, 5, 6])  # Invalid type for string comparison
    result = comparator.compare(col1, col2)
    assert result is None

    # different lengths
    col1 = pl.Series(["x", "y", "z"])
    col2 = pl.Series(["x", "y", "z", "c"])
    result = comparator.compare(col1, col2)
    assert result is None


# tests for PandasStringComparator
def test_pandas_string_comparator_exact_match():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c"])
    col2 = pd.Series(["a", "b", "c"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_string_comparator_case_space_insensitivity():
    comparator = PandasStringComparator(ignore_case=True, ignore_space=True)
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]

    comparator = PandasStringComparator(ignore_case=True, ignore_space=False)
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, False, False]

    comparator = PandasStringComparator(ignore_case=False, ignore_space=True)
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [False, True, False]

    comparator = PandasStringComparator(ignore_case=False, ignore_space=False)
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [False, False, False]


def test_pandas_string_comparator_nan_handling():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", float("nan")])
    col2 = pd.Series(["a", "b", float("nan")])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_string_comparator_mismatch():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c"])
    col2 = pd.Series(["a", "b", "d"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, False]


def test_pandas_string_comparator_error_handling():
    comparator = PandasStringComparator()
    col1 = pd.Series([1, 2, 3])  # Invalid type for string comparison
    col2 = pd.Series([4, 5, 6])  # Invalid type for string comparison
    result = comparator.compare(col1, col2)
    assert result is None

    # different lengths
    col1 = pd.Series(["x", "y", "z"])
    col2 = pd.Series(["x", "y", "z", "c"])
    result = comparator.compare(col1, col2)
    assert result is None


# tests for SparkStringComparator
def test_spark_string_comparator_exact_match(spark_session):
    comparator = SparkStringComparator()
    df = spark_session.createDataFrame(
        [("a", "a"), ("b", "b"), ("c", "c")], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]


def test_spark_string_comparator_case_space_insensitivity(spark_session):
    df = spark_session.createDataFrame(
        [("a", " a"), ("b", "   B  "), ("c", "C")], ["col1", "col2"]
    )

    comparator = SparkStringComparator(ignore_case=True, ignore_space=True)
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]

    comparator = SparkStringComparator(ignore_case=True, ignore_space=False)
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=False),
        ps.Row(col_match=False),
        ps.Row(col_match=True),
    ]

    comparator = SparkStringComparator(ignore_case=False, ignore_space=True)
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=False),
        ps.Row(col_match=False),
    ]

    comparator = SparkStringComparator(ignore_case=False, ignore_space=False)
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=False),
        ps.Row(col_match=False),
        ps.Row(col_match=False),
    ]


def test_spark_string_comparator_nan_handling(spark_session):
    comparator = SparkStringComparator()
    df = spark_session.createDataFrame(
        [("a", "a"), (float("nan"), float("nan")), ("c", "c")], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]


def test_spark_string_comparator_error_handling(spark_session):
    comparator = SparkStringComparator()
    df = spark_session.createDataFrame(
        [(1, 2), (3, 4), (5, 6)], ["col1", "col2"]
    )  # Invalid type for string comparison
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result is None


# tests for SnowflakeStringComparator
@pytest.mark.snowflake
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


@pytest.mark.snowflake
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


@pytest.mark.snowflake
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


@pytest.mark.snowflake
def test_snowflake_string_comparator_error_handling(snowflake_session):
    comparator = SnowflakeStringComparator()
    df = snowflake_session.createDataFrame(
        [(1, 2), (3, 4), (5, 6)], ["col1", "col2"]
    )  # Invalid type for string comparison
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result is None
