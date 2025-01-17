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
    detailed_compare,
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
    mock_prod_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    mock_prod_columns = ["name", "age", "pet"]

    mock_release_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    mock_release_columns = ["name", "age", "pet"]

    # Create DataFrames
    mock_prod_df = spark_session.createDataFrame(mock_prod_data, mock_prod_columns)
    mock_release_df = spark_session.createDataFrame(
        mock_release_data, mock_release_columns
    )

    # call detailed_compare
    result_compared_data = detailed_compare(
        spark_session, mock_prod_df, mock_release_df, [], ["age"]
    )

    # assert result
    assert result_compared_data.matches()
    assert isinstance(result_compared_data, SparkSQLCompare)


def test_detailed_compare_with_column_to_join(spark_session):
    # create mock data
    mock_prod_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    mock_prod_columns = ["name", "age", "pet"]

    mock_release_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    mock_release_columns = ["name", "age", "pet"]

    # Create DataFrames
    mock_prod_df = spark_session.createDataFrame(mock_prod_data, mock_prod_columns)
    mock_release_df = spark_session.createDataFrame(
        mock_release_data, mock_release_columns
    )

    # call detailed_compare
    result_compared_data = detailed_compare(
        spark_session, mock_prod_df, mock_release_df, ["name"], []
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
    input_prod_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    columns_prod = ["name", "age", "pet"]

    input_release_data = [("19", "cat"), ("70", "bunny"), ("22", "dog")]
    columns_release = ["age", "pet"]

    # Create DataFrames
    input_prod_df = spark_session.createDataFrame(input_prod_data, columns_prod)
    input_release_df = spark_session.createDataFrame(
        input_release_data, columns_release
    )

    # call call_rows
    with pytest.raises(
        Exception, match="name is present in prod_df but does not exist in release_df"
    ):
        sort_rows(input_prod_df, input_release_df)


def test_sort_rows_success(caplog, spark_session):
    caplog.set_level(logging.WARNING)

    # create mock data
    input_prod_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    columns_prod = ["name", "age", "pet"]

    input_release_data = [
        ("19", "cat", "alice", "red"),
        ("70", "bunny", "john", "black"),
        ("22", "dog", "bob", "white"),
    ]
    columns_release = ["age", "pet", "name", "color"]

    # create dataFrames
    input_prod_df = spark_session.createDataFrame(input_prod_data, columns_prod)
    input_release_df = spark_session.createDataFrame(
        input_release_data, columns_release
    )

    # call sort_rows
    sorted_prod_df, sorted_release_df = sort_rows(input_prod_df, input_release_df)

    # create expected prod_dataframe
    expected_prod_data = [
        ("alice", "19", "cat", 1),
        ("bob", "22", "dog", 2),
        ("john", "70", "bunny", 3),
    ]
    expected_prod_schema = StructType(
        [
            StructField("name", StringType(), True),
            StructField("age", StringType(), True),
            StructField("pet", StringType(), True),
            StructField("row", IntegerType(), True),
        ]
    )
    expected_prod_df = spark_session.createDataFrame(
        expected_prod_data, expected_prod_schema
    )

    # create expected release_dataframe
    expected_release_data = [
        ("19", "cat", "alice", "red", 1),
        ("22", "dog", "bob", "white", 2),
        ("70", "bunny", "john", "black", 3),
    ]
    expected_release_schema = StructType(
        [
            StructField("age", StringType(), True),
            StructField("pet", StringType(), True),
            StructField("name", StringType(), True),
            StructField("color", StringType(), True),
            StructField("row", IntegerType(), True),
        ]
    )
    expected_release_df = spark_session.createDataFrame(
        expected_release_data, expected_release_schema
    )

    # assertions
    assert sorted_prod_df.collect() == expected_prod_df.collect()
    assert sorted_release_df.collect() == expected_release_df.collect()
    assert (
        "WARNING: There are columns present in Compare df that do not exist in Base df. The Base df columns will be used for row-wise sorting and may produce unanticipated report output if the extra fields are not null.\n"
        in caplog.text
    )


def test_sort_columns_failure(spark_session):
    # create mock dataframes
    input_prod_data = [
        ("row1", "col2", "col3"),
        ("row2", "col2", "col3"),
        ("row3", "col2", "col3"),
    ]
    columns_1 = ["col1", "col2", "col3"]

    input_release_data = [("row1", "col2"), ("row2", "col2"), ("row3", "col2")]
    columns_2 = ["col1", "col2"]

    # Create DataFrames
    input_prod_df = spark_session.createDataFrame(input_prod_data, columns_1)
    input_release_df = spark_session.createDataFrame(input_release_data, columns_2)

    # call sort_columns
    with pytest.raises(
        Exception, match="col3 is present in prod_df but does not exist in release_df"
    ):
        sort_columns(input_prod_df, input_release_df)


def test_sort_columns_success(spark_session):
    # create mock dataframes
    input_prod_data = [
        ("bob", "22", "dog"),
        ("alice", "19", "cat"),
        ("john", "70", "bunny"),
    ]
    columns_prod = ["name", "age", "pet"]

    input_release_data = [
        ("19", "cat", "alice"),
        ("70", "bunny", "john"),
        ("22", "dog", "bob"),
    ]
    columns_release = ["age", "pet", "name"]

    # create input dataFrames
    input_prod_df = spark_session.createDataFrame(input_prod_data, columns_prod)
    input_release_df = spark_session.createDataFrame(
        input_release_data, columns_release
    )

    # create expected dataFrames
    expected_prod_data = [
        ("alice", "19", "cat"),
        ("bob", "22", "dog"),
        ("john", "70", "bunny"),
    ]
    expected_release_data = [
        ("19", "cat", "alice"),
        ("22", "dog", "bob"),
        ("70", "bunny", "john"),
    ]
    expected_prod_df = spark_session.createDataFrame(expected_prod_data, columns_prod)
    expected_release_df = spark_session.createDataFrame(
        expected_release_data, columns_release
    )

    # call sort_columns
    output_prod_df, output_release_df = sort_columns(input_prod_df, input_release_df)

    # assert the dfs are equal
    assert output_prod_df.collect() == expected_prod_df.collect()
    assert output_release_df.collect() == expected_release_df.collect()
