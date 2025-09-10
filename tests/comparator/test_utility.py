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

from datetime import date, datetime
from decimal import Decimal

import pytest
import snowflake.snowpark.types as spt
from datacompy.comparator.utility import (
    get_snowflake_column_dtypes,
    get_spark_column_dtypes,
)
from pyspark.sql.types import (
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def test_get_spark_column_dtypes(spark_session):
    schema = StructType(
        [
            StructField("str_col", StringType(), True),
            StructField("int_col", IntegerType(), True),
            StructField("float_col", FloatType(), True),
            StructField("double_col", DoubleType(), True),
            StructField("decimal_col", DecimalType(10, 2), True),
            StructField("date_col", DateType(), True),
            StructField("timestamp_col", TimestampType(), True),
        ]
    )

    data = [
        (
            "test",
            1,
            1.5,
            2.0,
            Decimal("10.25"),
            date(2023, 1, 1),
            datetime(2023, 1, 1, 12, 0),
        )
    ]
    df = spark_session.createDataFrame(data, schema)

    # Test each datatype
    str_type, int_type = get_spark_column_dtypes(df, "str_col", "int_col")
    assert str_type == "string"
    assert int_type == "int"

    float_type, double_type = get_spark_column_dtypes(df, "float_col", "double_col")
    assert float_type == "float"
    assert double_type == "double"

    decimal_type, date_type = get_spark_column_dtypes(df, "decimal_col", "date_col")
    assert decimal_type.startswith("decimal")
    assert date_type == "date"

    timestamp_type, _ = get_spark_column_dtypes(df, "timestamp_col", "str_col")
    assert timestamp_type == "timestamp"


@pytest.mark.snowflake
def test_get_snowflake_column_dtypes(snowflake_session):
    schema = spt.StructType(
        [
            spt.StructField("str_col", spt.StringType()),
            spt.StructField("int_col", spt.IntegerType()),
            spt.StructField("float_col", spt.FloatType()),
            spt.StructField("double_col", spt.DoubleType()),
            spt.StructField("decimal_col", spt.DecimalType(10, 2)),
            spt.StructField("date_col", spt.DateType()),
            spt.StructField("timestamp_col", spt.TimestampType()),
            spt.StructField("tinyint_col", spt.ByteType()),
            spt.StructField("short_col", spt.ShortType()),
            spt.StructField("long_col", spt.LongType()),
        ]
    )

    data = [
        (
            "test",
            1,
            1.5,
            2.0,
            Decimal("10.25"),
            date(2023, 1, 1),
            datetime(2023, 1, 1, 12, 0),
            1,
            1,
            1,
        )
    ]

    df = snowflake_session.create_dataframe(data, schema=schema)

    # Test each datatype
    str_type, int_type = get_snowflake_column_dtypes(df, "str_col", "int_col")
    assert "string" in str_type
    assert int_type == "bigint"

    float_type, double_type = get_snowflake_column_dtypes(df, "float_col", "double_col")
    assert float_type == "double"
    assert double_type == "double"

    decimal_type, date_type = get_snowflake_column_dtypes(df, "decimal_col", "date_col")
    assert "decimal" in decimal_type
    assert date_type == "date"

    timestamp_type, tinyint_type = get_snowflake_column_dtypes(
        df, "timestamp_col", "tinyint_col"
    )
    assert timestamp_type == "timestamp"
    assert tinyint_type == "bigint"

    short_type, long_type = get_snowflake_column_dtypes(df, "short_col", "long_col")
    assert short_type == "bigint"
    assert long_type == "bigint"


def test_get_spark_column_dtypes_case_insensitive(spark_session):
    data = [(1, "a"), (2, "b")]
    df = spark_session.createDataFrame(data, ["NUM", "STR"])
    dtype1, dtype2 = get_spark_column_dtypes(df, "num", "str")
    assert dtype1 == "bigint"
    assert dtype2 == "string"


@pytest.mark.snowflake
def test_get_snowflake_column_dtypes_case_insensitive(snowflake_session):
    schema = spt.StructType(
        [
            spt.StructField("COL1", spt.IntegerType()),
            spt.StructField("COL2", spt.StringType()),
        ]
    )
    df = snowflake_session.create_dataframe([[1, "a"], [2, "b"]], schema=schema)
    dtype1, dtype2 = get_snowflake_column_dtypes(df, "col1", "col2")
    assert dtype1 == "bigint"
    assert "string" in dtype2
