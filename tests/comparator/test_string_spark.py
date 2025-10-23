import pytest

pytest.importorskip("pyspark")

import pyspark.sql as ps
from datacompy.comparator.string import SparkStringComparator


# tests for SparkStringComparator
def test_spark_string_comparator_exact_match(spark_session):
    comparator = SparkStringComparator()
    df = spark_session.createDataFrame(
        [("a", "a"), ("b", "b"), ("c", "c")], ["col1", "col2"]
    )
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
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
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]

    comparator = SparkStringComparator(ignore_case=True, ignore_space=False)
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=False),
        ps.Row(col_match=False),
        ps.Row(col_match=True),
    ]

    comparator = SparkStringComparator(ignore_case=False, ignore_space=True)
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=False),
        ps.Row(col_match=False),
    ]

    comparator = SparkStringComparator(ignore_case=False, ignore_space=False)
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
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
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
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
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    assert result_col is None
