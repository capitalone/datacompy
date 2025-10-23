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

import pytest

pytest.importorskip("pyspark")

from datacompy.comparator.array import SparkArrayLikeComparator


# PySpark
def test_spark_compare_equal_arrays(spark_session):
    # Setup
    data = [([1, 2], [1, 2]), ([3, 4], [3, 4])]
    df = spark_session.createDataFrame(data, ["col1", "col2"])
    comparator = SparkArrayLikeComparator()

    # Execute
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("match", result_col)

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
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("match", result_col)

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
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("match", result_col)

    # Assert
    assert result is not None
    matches = result.select("match").collect()
    assert all(row.match for row in matches)


def test_spark_compare_non_array(spark_session):
    # integers
    data = [(1, 1), (2, 2)]
    df = spark_session.createDataFrame(data, ["col1", "col2"])
    comparator = SparkArrayLikeComparator()
    result = comparator.compare(dataframe=df, col1="col1", col2="col2")
    assert result is None

    # floats
    data = [(1.0, 1.0), (2.0, 2.0)]
    df = spark_session.createDataFrame(data, ["col1", "col2"])
    comparator = SparkArrayLikeComparator()
    result = comparator.compare(dataframe=df, col1="col1", col2="col2")
    assert result is None

    # dicts
    data = [({"a": 1}, {"a": 1}), ({"b": 2}, {"b": 2})]
    df = spark_session.createDataFrame(data, ["col1", "col2"])
    comparator = SparkArrayLikeComparator()
    result = comparator.compare(dataframe=df, col1="col1", col2="col2")
    assert result is None
