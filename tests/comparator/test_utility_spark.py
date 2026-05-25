#
# Copyright 2026 Capital One Services, LLC
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

pytest.importorskip("pyspark")

from datacompy.comparator.utility import get_spark_column_dtypes
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


@pytest.mark.pyspark
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


@pytest.mark.pyspark
def test_get_spark_column_dtypes_case_insensitive(spark_session):
    data = [(1, "a"), (2, "b")]
    df = spark_session.createDataFrame(data, ["NUM", "STR"])
    dtype1, dtype2 = get_spark_column_dtypes(df, "num", "str")
    assert dtype1 == "bigint"
    assert dtype2 == "string"
