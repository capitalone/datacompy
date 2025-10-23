import pytest

pytest.importorskip("pyspark")

import pyspark.sql as ps
from datacompy.comparator.numeric import SparkNumericComparator


# tests for SparkNumericComparator
def test_spark_numeric_comparator_exact_match(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], ["col1", "col2"]
    )
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]


def test_spark_numeric_comparator_approximate_match(spark_session):
    comparator = SparkNumericComparator(rtol=1e-3, atol=1e-3)
    df = spark_session.createDataFrame(
        [(1.0, 1.001), (2.0, 2.002), (3.0, 3.003)], ["col1", "col2"]
    )
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]


def test_spark_numeric_comparator_type_casting(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame([(1, 1.0), (2, 2.0), (3, 3.0)], ["col1", "col2"])
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]


def test_spark_numeric_comparator_nan_handling(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [(1.0, 1.0), (float("nan"), float("nan")), (3.0, 3.0)], ["col1", "col2"]
    )
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]


def test_spark_numeric_comparator_mismatch(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [(1.0, 1.0), (2.0, 2.5), (3.0, 3.0)], ["col1", "col2"]
    )
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=False),
        ps.Row(col_match=True),
    ]


def test_spark_numeric_comparator_error_handling(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [("a", "x"), ("b", "y"), ("c", "z")], ["col1", "col2"]
    )  # Invalid type for numeric comparison
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    assert result_col is None
