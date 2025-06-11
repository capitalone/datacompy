import pandas as pd
import polars as pl
import pytest
from datacompy.comparator.numeric import (
    PandasNumericComparator,
    PolarsNumericComparator,
    SnowflakeNumericComparator,
    SparkNumericComparator,
)
from pyspark.sql import Row


# tests for PolarsNumericComparator
def test_polars_numeric_comparator_exact_match():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.0, 2.0, 3.0])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_approximate_match():
    comparator = PolarsNumericComparator(rtol=1e-3, atol=1e-3)
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.001, 2.002, 3.003])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_type_casting():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1, 2, 3])  # Integer type
    col2 = pl.Series([1.0, 2.0, 3.0])  # Float type
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_nan_handling():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, float("nan"), 3.0])
    col2 = pl.Series([1.0, float("nan"), 3.0])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_mismatch():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.0, 2.5, 3.0])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, False, True]


def test_polars_numeric_comparator_error_handling():
    comparator = PolarsNumericComparator()
    col1 = pl.Series(["a", "b", "c"])  # Invalid type for numeric comparison
    col2 = pl.Series(["x", "y", "z"])  # Invalid type for numeric comparison
    result = comparator.compare(col1, col2)
    assert result.to_list() == [False, False, False]

    # different lengths
    col1 = pl.Series([1, 2, 3])
    col2 = pl.Series([1, 2, 3, 4])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [False, False, False]


# tests for PandasNumericComparator
def test_pandas_numeric_comparator_exact_match():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.0, 2.0, 3.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_approximate_match():
    comparator = PandasNumericComparator(rtol=1e-3, atol=1e-3)
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.001, 2.002, 3.003])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_type_casting():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1, 2, 3])  # Integer type
    col2 = pd.Series([1.0, 2.0, 3.0])  # Float type
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_nan_handling():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, float("nan"), 3.0])
    col2 = pd.Series([1.0, float("nan"), 3.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_mismatch():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.0, 2.5, 3.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, False, True]


def test_pandas_numeric_comparator_error_handling():
    comparator = PandasNumericComparator()
    col1 = pd.Series(["a", "b", "c"])  # Invalid type for numeric comparison
    col2 = pd.Series(["x", "y", "z"])  # Invalid type for numeric comparison
    result = comparator.compare(col1, col2)
    assert result.tolist() == [False, False, False]

    # different lengths
    col2 = pd.Series(["x", "y", "z", "c"])  # Invalid type for numeric comparison
    result = comparator.compare(col1, col2)
    assert result.tolist() == [False, False, False]


# tests for SparkNumericComparator
def test_spark_numeric_comparator_exact_match(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=True),
        Row(col_match=True),
        Row(col_match=True),
    ]


def test_spark_numeric_comparator_approximate_match(spark_session):
    comparator = SparkNumericComparator(rtol=1e-3, atol=1e-3)
    df = spark_session.createDataFrame(
        [(1.0, 1.001), (2.0, 2.002), (3.0, 3.003)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=True),
        Row(col_match=True),
        Row(col_match=True),
    ]


def test_spark_numeric_comparator_type_casting(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame([(1, 1.0), (2, 2.0), (3, 3.0)], ["col1", "col2"])
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=True),
        Row(col_match=True),
        Row(col_match=True),
    ]


def test_spark_numeric_comparator_nan_handling(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [(1.0, 1.0), (float("nan"), float("nan")), (3.0, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=True),
        Row(col_match=True),
        Row(col_match=True),
    ]


def test_spark_numeric_comparator_mismatch(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [(1.0, 1.0), (2.0, 2.5), (3.0, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=True),
        Row(col_match=False),
        Row(col_match=True),
    ]


def test_spark_numeric_comparator_error_handling(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [("a", "x"), ("b", "y"), ("c", "z")], ["col1", "col2"]
    )  # Invalid type for numeric comparison
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=False),
        Row(col_match=False),
        Row(col_match=False),
    ]


# tests for SnowflakeNumericComparator
@pytest.mark.snowflake
def test_snowflake_numeric_comparator_exact_match(snowpark_session):
    comparator = SnowflakeNumericComparator()
    df = snowpark_session.createDataFrame(
        [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=True),
        Row(col_match=True),
        Row(col_match=True),
    ]


@pytest.mark.snowflake
def test_snowflake_numeric_comparator_approximate_match(snowpark_session):
    comparator = SnowflakeNumericComparator(rtol=1e-3, atol=1e-3)
    df = snowpark_session.createDataFrame(
        [(1.0, 1.001), (2.0, 2.002), (3.0, 3.003)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=True),
        Row(col_match=True),
        Row(col_match=True),
    ]


@pytest.mark.snowflake
def test_snowflake_numeric_comparator_type_casting(snowpark_session):
    comparator = SnowflakeNumericComparator()
    df = snowpark_session.createDataFrame(
        [(1, 1.0), (2, 2.0), (3, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=True),
        Row(col_match=True),
        Row(col_match=True),
    ]


@pytest.mark.snowflake
def test_snowflake_numeric_comparator_nan_handling(snowpark_session):
    comparator = SnowflakeNumericComparator()
    df = snowpark_session.createDataFrame(
        [(1.0, 1.0), (float("nan"), float("nan")), (3.0, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=True),
        Row(col_match=True),
        Row(col_match=True),
    ]


@pytest.mark.snowflake
def test_snowflake_numeric_comparator_mismatch(snowpark_session):
    comparator = SnowflakeNumericComparator()
    df = snowpark_session.createDataFrame(
        [(1.0, 1.0), (2.0, 2.5), (3.0, 3.0)], ["col1", "col2"]
    )
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=True),
        Row(col_match=False),
        Row(col_match=True),
    ]


@pytest.mark.snowflake
def test_snowflake_numeric_comparator_error_handling(snowpark_session):
    comparator = SnowflakeNumericComparator()
    df = snowpark_session.createDataFrame(
        [("a", "x"), ("b", "y"), ("c", "z")], ["col1", "col2"]
    )  # Invalid type for numeric comparison
    result = comparator.compare(
        dataframe=df, col1="col1", col2="col2", col_match="col_match"
    )
    assert result.select(["col_match"]).collect() == [
        Row(col_match=False),
        Row(col_match=False),
        Row(col_match=False),
    ]
