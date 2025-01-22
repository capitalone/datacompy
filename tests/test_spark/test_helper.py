#
# Copyright 2025 Capital One Services, LLC
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

"""
Testing out the spark helper functionality
"""

import logging
import sys

import pytest

pytest.importorskip("pyspark")
if sys.version_info >= (3, 12):
    pytest.skip("unsupported python version", allow_module_level=True)


from datacompy.spark.helper import (
    compare_by_row,
    format_numeric_fields,
    handle_numeric_strings,
    sort_columns,
    sort_rows,
)
from datacompy.spark.sql import SparkSQLCompare
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def test_detailed_compare_with_string2columns(spark_session):
    # create mock data
    mock_base_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    mock_base_columns = ["name", "age", "pet"]

    mock_compare_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    mock_compare_columns = ["name", "age", "pet"]

    # Create DataFrames
    mock_base_df = spark_session.createDataFrame(mock_base_data, mock_base_columns)
    mock_compare_df = spark_session.createDataFrame(
        mock_compare_data, mock_compare_columns
    )

    # call detailed_compare
    result_compared_data = compare_by_row(
        spark_session=spark_session,
        base_dataframe=mock_base_df,
        compare_dataframe=mock_compare_df,
        string2double_cols=["age"],
    )

    # assert result
    assert isinstance(result_compared_data, SparkSQLCompare)
    assert result_compared_data.matches()


def test_detailed_compare_with_column_to_join(spark_session):
    # create mock data
    mock_base_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    mock_base_columns = ["name", "age", "pet"]

    mock_compare_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    mock_compare_columns = ["name", "age", "pet"]

    # Create DataFrames
    mock_base_df = spark_session.createDataFrame(mock_base_data, mock_base_columns)
    mock_compare_df = spark_session.createDataFrame(
        mock_compare_data, mock_compare_columns
    )

    # call detailed_compare
    result_compared_data = compare_by_row(
        spark_session=spark_session,
        base_dataframe=mock_base_df,
        compare_dataframe=mock_compare_df,
        string2double_cols=[],
    )

    # assert result
    assert result_compared_data.matches()
    assert isinstance(result_compared_data, SparkSQLCompare)


def test_handle_numeric_strings(spark_session):
    # create mock_df
    mock_data = [("bob", "22", "dog"), ("alice", "19", "cat"), ("john", "70", "bunny")]
    mock_columns = ["name", "age", "pet"]
    mock_df = spark_session.createDataFrame(mock_data, mock_columns)

    # create mock field_list
    mock_field_list = ["age"]

    # call handle_numeric_strings
    result_df = handle_numeric_strings(mock_df, mock_field_list)

    # create expected dataframe
    expected_data = [
        ("bob", 22.0, "dog"),
        ("alice", 19.0, "cat"),
        ("john", 70.0, "bunny"),
    ]
    expected_columns = ["name", "age", "pet"]
    expected_df = spark_session.createDataFrame(expected_data, expected_columns)

    # assert calls
    assert result_df.collect() == expected_df.collect()


def test_format_numeric_fields(spark_session):
    # create mock dataframe
    mock_data = [("bob", 22, "dog"), ("alice", 19, "cat"), ("john", 70, "bunny")]
    mock_columns = ["name", "age", "pet"]
    mock_df = spark_session.createDataFrame(mock_data, mock_columns)

    # call format_numeric_fields
    formatted_df = format_numeric_fields(mock_df)

    # create expected dataframe
    expected_data = [
        ("bob", "22.00000", "dog"),
        ("alice", "19.00000", "cat"),
        ("john", "70.00000", "bunny"),
    ]
    expected_columns = ["name", "age", "pet"]
    expected_df = spark_session.createDataFrame(expected_data, expected_columns)

    # assert calls
    assert formatted_df.collect() == expected_df.collect()


def test_sort_rows_failure(spark_session):
    # create mock dataframes
    input_base_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    columns_base = ["name", "age", "pet"]

    input_compare_data = [("19", "cat"), ("70", "bunny"), ("22", "dog")]
    columns_commpare = ["age", "pet"]

    # Create DataFrames
    input_base_df = spark_session.createDataFrame(input_base_data, columns_base)
    input_compare_df = spark_session.createDataFrame(
        input_compare_data, columns_commpare
    )

    # call call_rows
    with pytest.raises(
        Exception, match="name is present in base_df but does not exist in compare_df"
    ):
        sort_rows(input_base_df, input_compare_df)


def test_sort_rows_success(caplog, spark_session):
    caplog.set_level(logging.WARNING)

    # create mock data
    input_base_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    columns_base = ["name", "age", "pet"]

    input_compare_data = [
        ("19", "cat", "alice", "red"),
        ("70", "bunny", "john", "black"),
        ("22", "dog", "bob", "white"),
    ]
    columns_compare = ["age", "pet", "name", "color"]

    # create dataFrames
    input_base_df = spark_session.createDataFrame(input_base_data, columns_base)
    input_compare_df = spark_session.createDataFrame(
        input_compare_data, columns_compare
    )

    # call sort_rows
    sorted_base_df, sorted_compare_df = sort_rows(input_base_df, input_compare_df)

    # create expected base_dataframe
    expected_base_data = [
        ("alice", "19", "cat", 1),
        ("bob", "22", "dog", 2),
        ("john", "70", "bunny", 3),
    ]
    expected_base_schema = StructType(
        [
            StructField("name", StringType(), True),
            StructField("age", StringType(), True),
            StructField("pet", StringType(), True),
            StructField("row", IntegerType(), True),
        ]
    )
    expected_base_df = spark_session.createDataFrame(
        expected_base_data, expected_base_schema
    )

    # create expected compare_dataframe
    expected_compare_data = [
        ("19", "cat", "alice", "red", 1),
        ("22", "dog", "bob", "white", 2),
        ("70", "bunny", "john", "black", 3),
    ]
    expected_compare_schema = StructType(
        [
            StructField("age", StringType(), True),
            StructField("pet", StringType(), True),
            StructField("name", StringType(), True),
            StructField("color", StringType(), True),
            StructField("row", IntegerType(), True),
        ]
    )
    expected_compare_df = spark_session.createDataFrame(
        expected_compare_data, expected_compare_schema
    )

    # assertions
    assert sorted_base_df.collect() == expected_base_df.collect()
    assert sorted_compare_df.collect() == expected_compare_df.collect()
    assert (
        "WARNING: There are columns present in Compare df that do not exist in Base df. The Base df columns will be used for row-wise sorting and may produce unanticipated report output if the extra fields are not null.\n"
        in caplog.text
    )


def test_sort_columns_failure(spark_session):
    # create mock dataframes
    input_base_data = [
        ("row1", "col2", "col3"),
        ("row2", "col2", "col3"),
        ("row3", "col2", "col3"),
    ]
    columns_1 = ["col1", "col2", "col3"]

    input_compare_data = [("row1", "col2"), ("row2", "col2"), ("row3", "col2")]
    columns_2 = ["col1", "col2"]

    # Create DataFrames
    input_base_df = spark_session.createDataFrame(input_base_data, columns_1)
    input_compare_df = spark_session.createDataFrame(input_compare_data, columns_2)

    # call sort_columns
    with pytest.raises(
        Exception, match="col3 is present in base_df but does not exist in compare_df"
    ):
        sort_columns(input_base_df, input_compare_df)


def test_sort_columns_success(spark_session):
    # Create sample DataFrames
    data1 = [(1, "a"), (2, "b"), (3, "c")]
    data2 = [(1, "a"), (2, "b"), (3, "c")]
    columns = ["id", "value"]

    df1 = spark_session.createDataFrame(data1, columns)
    df2 = spark_session.createDataFrame(data2, columns)

    # Test with matching columns
    sorted_df1, sorted_df2 = sort_columns(df1, df2)
    assert sorted_df1.columns == sorted_df2.columns
    assert sorted_df1.collect() == sorted_df2.collect()
