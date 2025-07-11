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

import snowflake.snowpark.types as spt
from datacompy.comparator.utility import (
    get_snowflake_column_dtypes,
    get_spark_column_dtypes,
)


def test_get_spark_column_dtypes(spark_session):
    data = [(1, "a"), (2, "b")]
    df = spark_session.createDataFrame(data, ["num", "str"])
    dtype1, dtype2 = get_spark_column_dtypes(df, "num", "str")
    assert dtype1 == "bigint"
    assert dtype2 == "string"


def test_get_snowflake_column_dtypes(snowflake_session):
    schema = spt.StructType(
        [
            spt.StructField("col1", spt.IntegerType()),
            spt.StructField("col2", spt.StringType()),
        ]
    )
    # Mock snowflake dataframe
    df = snowflake_session.create_dataframe([[1, "a"], [2, "b"]], schema=schema)
    dtype1, dtype2 = get_snowflake_column_dtypes(df, "col1", "col2")
    assert str(dtype1) == "LongType()"
    assert str(dtype2) == "StringType()"


def test_get_spark_column_dtypes_case_insensitive(spark_session):
    data = [(1, "a"), (2, "b")]
    df = spark_session.createDataFrame(data, ["NUM", "STR"])
    dtype1, dtype2 = get_spark_column_dtypes(df, "num", "str")
    assert dtype1 == "bigint"
    assert dtype2 == "string"


def test_get_snowflake_column_dtypes_case_insensitive(snowflake_session):
    schema = spt.StructType(
        [
            spt.StructField("COL1", spt.IntegerType()),
            spt.StructField("COL2", spt.StringType()),
        ]
    )
    df = snowflake_session.create_dataframe([[1, "a"], [2, "b"]], schema=schema)
    dtype1, dtype2 = get_snowflake_column_dtypes(df, "col1", "col2")
    assert str(dtype1) == "LongType()"
    assert str(dtype2) == "StringType()"
