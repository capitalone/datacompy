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

from datacompy.comparator.utility import (
    get_spark_column_dtypes,
    get_spark_functions,
    get_spark_window,
    is_spark_connect_dataframe,
    is_spark_connect_object,
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


@pytest.mark.pyspark
def test_is_spark_connect_object_connect_branch():
    """A Spark Connect Column is built without any session or SparkContext."""
    pytest.importorskip("grpc")

    import pandas as pd
    from pyspark.sql.connect import functions as connect_functions

    assert is_spark_connect_object(connect_functions.col("value"))
    assert not is_spark_connect_object(pd.DataFrame({"a": [1]}))
    assert not is_spark_connect_object(object())


@pytest.mark.pyspark
def test_is_spark_connect_dataframe_is_narrower_than_is_connect_object():
    """Only a Connect DataFrame passes, so type validation stays meaningful."""
    pytest.importorskip("grpc")

    import pandas as pd
    from pyspark.sql.connect import functions as connect_functions

    column = connect_functions.col("value")

    # A Connect Column is a Spark Connect object, but it is not a DataFrame.
    assert is_spark_connect_object(column)
    assert not is_spark_connect_dataframe(column)

    assert not is_spark_connect_dataframe(pd.DataFrame({"a": [1]}))
    assert not is_spark_connect_dataframe(object())


@pytest.mark.pyspark
def test_get_spark_helpers_connect_branch():
    """Spark Connect objects resolve to the Spark Connect implementations."""
    pytest.importorskip("grpc")

    from pyspark.sql.connect import functions as connect_functions
    from pyspark.sql.connect.window import Window as ConnectWindow

    column = connect_functions.col("value")

    assert get_spark_functions(column) is connect_functions
    assert get_spark_window(column) is ConnectWindow


@pytest.mark.pyspark
def test_get_spark_helpers_match_the_session(spark_session):
    """The helpers must agree with the flavour of the session under test.

    This runs under both lanes -- the default classic run and the Spark Connect
    run driven by ``pytest-connect.ini`` -- so it asserts against whichever
    flavour ``spark_session`` actually is. The negative assertions matter:
    without them an implementation that always returned one flavour would pass.
    """
    pytest.importorskip("grpc")

    import pyspark.sql.functions as classic_functions
    from pyspark.sql import Window as ClassicWindow
    from pyspark.sql.connect import functions as connect_functions
    from pyspark.sql.connect.window import Window as ConnectWindow

    df = spark_session.range(1)

    # For a DataFrame the broad and the narrow check must agree, whichever
    # lane this is running in.
    assert is_spark_connect_dataframe(df) == is_spark_connect_object(df)

    if is_spark_connect_object(df):
        assert get_spark_functions(df) is connect_functions
        assert get_spark_functions(df) is not classic_functions
        assert get_spark_window(df) is ConnectWindow
        assert get_spark_window(df) is not ClassicWindow
    else:
        assert get_spark_functions(df) is classic_functions
        assert get_spark_functions(df) is not connect_functions
        assert get_spark_window(df) is ClassicWindow
        assert get_spark_window(df) is not ConnectWindow


@pytest.mark.pyspark
def test_get_spark_helpers_default_to_classic():
    """Anything that is not a Spark Connect object falls back to classic."""
    import pyspark.sql.functions as classic_functions
    from pyspark.sql import Window as ClassicWindow

    assert get_spark_functions(object()) is classic_functions
    assert get_spark_window(object()) is ClassicWindow
