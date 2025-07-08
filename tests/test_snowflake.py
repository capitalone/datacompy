#
# Copyright 2025 Capital One Services, LLC
#
# Licensed under the Apache License, Version 2.0 (the "LICENSE");
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
Testing out the datacompy functionality
"""

import io
import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from decimal import Decimal
from io import StringIO
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pytest import raises

pytest.importorskip("snowflake")
pytest.importorskip("pyspark")


from datacompy.snowflake import (
    SnowflakeCompare,
    _generate_id_within_group,
    calculate_max_diff,
    columns_equal,
    temp_column_name,
)
from pandas.testing import assert_frame_equal, assert_series_equal
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.types import (
    DecimalType,
    StringType,
    StructField,
    StructType,
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

pd.DataFrame.iteritems = pd.DataFrame.items  # Pandas 2+ compatability
np.bool = np.bool_  # Numpy 1.24.3+ comptability


def test_numeric_columns_equal_abs(snowpark_session):
    data = """A|B|EXPECTED
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""

    df = snowpark_session.createDataFrame(pd.read_csv(StringIO(data), sep="|"))
    actual_out = columns_equal(df, "A", "B", "ACTUAL", abs_tol=0.2).toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_numeric_columns_equal_rel(snowpark_session):
    data = """A|B|EXPECTED
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""
    df = snowpark_session.createDataFrame(pd.read_csv(StringIO(data), sep="|"))
    actual_out = columns_equal(df, "A", "B", "ACTUAL", rel_tol=0.2).toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_string_columns_equal(snowpark_session):
    data = """A|B|EXPECTED
Hi|Hi|True
Yo|Yo|True
Hey|Hey |False
résumé|resume|False
résumé|résumé|True
💩|💩|True
💩|🤔|False
 | |True
  | |False
datacompy|DataComPy|False
something||False
|something|False
||True"""
    df = snowpark_session.createDataFrame(pd.read_csv(StringIO(data), sep="|"))
    actual_out = columns_equal(df, "A", "B", "ACTUAL", rel_tol=0.2).toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_string_columns_equal_with_ignore_spaces(snowpark_session):
    data = """A|B|EXPECTED
Hi|Hi|True
Yo|Yo|True
Hey|Hey |True
résumé|resume|False
résumé|résumé|True
💩|💩|True
💩|🤔|False
 | |True
  |       |True
datacompy|DataComPy|False
something||False
|something|False
||True"""
    df = snowpark_session.createDataFrame(pd.read_csv(StringIO(data), sep="|"))
    actual_out = columns_equal(
        df, "A", "B", "ACTUAL", rel_tol=0.2, ignore_spaces=True
    ).toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_date_columns_equal(snowpark_session):
    data = """A|B|EXPECTED
2017-01-01|2017-01-01|True
2017-01-02|2017-01-02|True
2017-10-01|2017-10-10|False
2017-01-01||False
|2017-01-01|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = snowpark_session.createDataFrame(pdf)
    # First compare just the strings
    actual_out = columns_equal(df, "A", "B", "ACTUAL", rel_tol=0.2).toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    pdf["A"] = pd.to_datetime(pdf["A"])
    pdf["B"] = pd.to_datetime(pdf["B"])
    df = snowpark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "A", "B", "ACTUAL", rel_tol=0.2).toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)
    # and reverse
    actual_out_rev = columns_equal(df, "B", "A", "ACTUAL", rel_tol=0.2).toPandas()[
        "ACTUAL"
    ]
    assert_series_equal(expect_out, actual_out_rev, check_names=False)


def test_date_columns_equal_with_ignore_spaces(snowpark_session):
    data = """A|B|EXPECTED
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    pdf = pd.read_csv(io.StringIO(data), sep="|")
    df = snowpark_session.createDataFrame(pdf)
    # First compare just the strings
    actual_out = columns_equal(
        df, "A", "B", "ACTUAL", rel_tol=0.2, ignore_spaces=True
    ).toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    try:  # pandas 2
        pdf["A"] = pd.to_datetime(pdf["A"], format="mixed")
        pdf["B"] = pd.to_datetime(pdf["B"], format="mixed")
    except ValueError:  # pandas 1.5
        pdf["A"] = pd.to_datetime(pdf["A"])
        pdf["B"] = pd.to_datetime(pdf["B"])
    df = snowpark_session.createDataFrame(pdf)
    actual_out = columns_equal(
        df, "A", "B", "ACTUAL", rel_tol=0.2, ignore_spaces=True
    ).toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)
    # and reverse
    actual_out_rev = columns_equal(
        df, "B", "A", "ACTUAL", rel_tol=0.2, ignore_spaces=True
    ).toPandas()["ACTUAL"]
    assert_series_equal(expect_out, actual_out_rev, check_names=False)


def test_columns_equal_same_type_dif_length(snowpark_session):
    schema = StructType(
        [
            StructField("NAME", StringType(length=20)),
            StructField("DECIMAL_VAL", DecimalType(precision=7, scale=5)),
            StructField("NAME_COPY", StringType(is_max_size=True)),
            StructField("DECIMAL_VAL_COPY", DecimalType(precision=20, scale=10)),
        ]
    )
    data = [
        ["Alice", 10.44556, "Alice", 10.44556],
        ["Bob", 2.33445, "Bob", 2.33445],
        ["Charlie", 5.2234, "Charlie", 5.2234],
    ]

    df = snowpark_session.create_dataframe(data, schema=schema)
    assert (
        columns_equal(df, "NAME", "NAME_COPY", "NAME_ACTUAL")
        .toPandas()["NAME_ACTUAL"]
        .all()
    )
    assert (
        columns_equal(df, "DECIMAL_VAL", "DECIMAL_VAL_COPY", "DECIMAL_VAL_ACTUAL")
        .toPandas()["DECIMAL_VAL_ACTUAL"]
        .all()
    )


def test_date_columns_unequal(snowpark_session):
    """I want datetime fields to match with dates stored as strings"""
    data = [{"A": "2017-01-01", "B": "2017-01-02"}, {"A": "2017-01-01"}]
    pdf = pd.DataFrame(data)
    pdf["A_DT"] = pd.to_datetime(pdf["A"])
    pdf["B_DT"] = pd.to_datetime(pdf["B"])
    df = snowpark_session.createDataFrame(pdf)
    assert columns_equal(df, "A", "A_DT", "ACTUAL").toPandas()["ACTUAL"].all()
    assert columns_equal(df, "B", "B_DT", "ACTUAL").toPandas()["ACTUAL"].all()
    assert columns_equal(df, "A_DT", "A", "ACTUAL").toPandas()["ACTUAL"].all()
    assert columns_equal(df, "B_DT", "B", "ACTUAL").toPandas()["ACTUAL"].all()
    assert not columns_equal(df, "B_DT", "A", "ACTUAL").toPandas()["ACTUAL"].any()
    assert not columns_equal(df, "A_DT", "B", "ACTUAL").toPandas()["ACTUAL"].any()
    assert not columns_equal(df, "A", "B_DT", "ACTUAL").toPandas()["ACTUAL"].any()
    assert not columns_equal(df, "B", "A_DT", "ACTUAL").toPandas()["ACTUAL"].any()


def test_bad_date_columns(snowpark_session):
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    data = [
        {"A": "2017-01-01", "B": "2017-01-01"},
        {"A": "2017-01-01", "B": "217-01-01"},
    ]
    pdf = pd.DataFrame(data)
    pdf["A_DT"] = pd.to_datetime(pdf["A"])
    df = snowpark_session.createDataFrame(pdf)
    assert not columns_equal(df, "A_DT", "B", "ACTUAL").toPandas()["ACTUAL"].all()
    assert columns_equal(df, "A_DT", "B", "ACTUAL").toPandas()["ACTUAL"].any()


def test_rounded_date_columns(snowpark_session):
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    data = [
        {"A": "2017-01-01", "B": "2017-01-01 00:00:00.000000", "EXP": True},
        {"A": "2017-01-01", "B": "2017-01-01 00:00:00.123456", "EXP": False},
        {"A": "2017-01-01", "B": "2017-01-01 00:00:01.000000", "EXP": False},
        {"A": "2017-01-01", "B": "2017-01-01 00:00:00", "EXP": True},
    ]
    pdf = pd.DataFrame(data)
    pdf["A_DT"] = pd.to_datetime(pdf["A"])
    df = snowpark_session.createDataFrame(pdf)
    actual = columns_equal(df, "A_DT", "B", "ACTUAL").toPandas()["ACTUAL"]
    expected = df.select("EXP").toPandas()["EXP"]
    assert_series_equal(actual, expected, check_names=False)


def test_decimal_float_columns_equal(snowpark_session):
    data = [
        {"A": Decimal("1"), "B": 1, "EXPECTED": True},
        {"A": Decimal("1.3"), "B": 1.3, "EXPECTED": True},
        {"A": Decimal("1.000003"), "B": 1.000003, "EXPECTED": True},
        {"A": Decimal("1.000000004"), "B": 1.000000003, "EXPECTED": False},
        {"A": Decimal("1.3"), "B": 1.2, "EXPECTED": False},
        {"A": np.nan, "B": np.nan, "EXPECTED": True},
        {"A": np.nan, "B": 1, "EXPECTED": False},
        {"A": Decimal("1"), "B": np.nan, "EXPECTED": False},
    ]
    pdf = pd.DataFrame(data)
    df = snowpark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "A", "B", "ACTUAL").toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_float_columns_equal_rel(snowpark_session):
    data = [
        {"A": Decimal("1"), "B": 1, "EXPECTED": True},
        {"A": Decimal("1.3"), "B": 1.3, "EXPECTED": True},
        {"A": Decimal("1.000003"), "B": 1.000003, "EXPECTED": True},
        {"A": Decimal("1.000000004"), "B": 1.000000003, "EXPECTED": True},
        {"A": Decimal("1.3"), "B": 1.2, "EXPECTED": False},
        {"A": np.nan, "B": np.nan, "EXPECTED": True},
        {"A": np.nan, "B": 1, "EXPECTED": False},
        {"A": Decimal("1"), "B": np.nan, "EXPECTED": False},
    ]
    pdf = pd.DataFrame(data)
    df = snowpark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "A", "B", "ACTUAL", abs_tol=0.001).toPandas()[
        "ACTUAL"
    ]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_columns_equal(snowpark_session):
    data = [
        {"A": Decimal("1"), "B": Decimal("1"), "EXPECTED": True},
        {"A": Decimal("1.3"), "B": Decimal("1.3"), "EXPECTED": True},
        {"A": Decimal("1.000003"), "B": Decimal("1.000003"), "EXPECTED": True},
        {
            "A": Decimal("1.000000004"),
            "B": Decimal("1.000000003"),
            "EXPECTED": False,
        },
        {"A": Decimal("1.3"), "B": Decimal("1.2"), "EXPECTED": False},
        {"A": np.nan, "B": np.nan, "EXPECTED": True},
        {"A": np.nan, "B": Decimal("1"), "EXPECTED": False},
        {"A": Decimal("1"), "B": np.nan, "EXPECTED": False},
    ]
    pdf = pd.DataFrame(data)
    df = snowpark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "A", "B", "ACTUAL").toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_columns_equal_rel(snowpark_session):
    data = [
        {"A": Decimal("1"), "B": Decimal("1"), "EXPECTED": True},
        {"A": Decimal("1.3"), "B": Decimal("1.3"), "EXPECTED": True},
        {"A": Decimal("1.000003"), "B": Decimal("1.000003"), "EXPECTED": True},
        {
            "A": Decimal("1.000000004"),
            "B": Decimal("1.000000003"),
            "EXPECTED": True,
        },
        {"A": Decimal("1.3"), "B": Decimal("1.2"), "EXPECTED": False},
        {"A": np.nan, "B": np.nan, "EXPECTED": True},
        {"A": np.nan, "B": Decimal("1"), "EXPECTED": False},
        {"A": Decimal("1"), "B": np.nan, "EXPECTED": False},
    ]
    pdf = pd.DataFrame(data)
    df = snowpark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "A", "B", "ACTUAL", abs_tol=0.001).toPandas()[
        "ACTUAL"
    ]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_infinity_and_beyond(snowpark_session):
    # https://spark.apache.org/docs/latest/sql-ref-datatypes.html#positivenegative-infinity-semantics
    # Positive/negative infinity multiplied by 0 returns NaN.
    # Positive infinity sorts lower than NaN and higher than any other values.
    # Negative infinity sorts lower than any other values.
    data = [
        {"A": np.inf, "B": np.inf, "EXPECTED": True},
        {"A": -np.inf, "B": -np.inf, "EXPECTED": True},
        {"A": -np.inf, "B": np.inf, "EXPECTED": True},
        {"A": np.inf, "B": -np.inf, "EXPECTED": True},
        {"A": 1, "B": 1, "EXPECTED": True},
        {"A": 1, "B": 0, "EXPECTED": False},
    ]
    pdf = pd.DataFrame(data)
    df = snowpark_session.createDataFrame(pdf)
    actual_out = columns_equal(df, "A", "B", "ACTUAL").toPandas()["ACTUAL"]
    expect_out = df.select("EXPECTED").toPandas()["EXPECTED"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_compare_table_setter_bad(snowpark_session):
    # Invalid table name
    with raises(
        ValueError, match=re.escape("invalid_table_name_1 is not a valid table name.")
    ):
        SnowflakeCompare(
            snowpark_session, "invalid_table_name_1", "invalid_table_name_2", ["A"]
        )
    # Valid table name but table does not exist
    with raises(SnowparkSQLException):
        SnowflakeCompare(
            snowpark_session, "non.existant.table_1", "non.existant.table_2", ["A"]
        )


@mock.patch(
    "datacompy.snowflake.SnowflakeCompare._validate_dataframe", new=mock.MagicMock()
)
@mock.patch("datacompy.snowflake.SnowflakeCompare._compare", new=mock.MagicMock())
def test_compare_table_unique_names(snowpark_session):
    # Assert that two tables with the same name but from a different DB/Schema have unique names
    # Same schema/name, different DB
    compare = SnowflakeCompare(
        snowpark_session,
        "test_db1.test_schema.test_name",
        "test_db2.test_schema.test_name",
        ["A"],
    )
    assert compare.df1_name != compare.df2_name

    # Same db/name, different schema
    compare = SnowflakeCompare(
        snowpark_session,
        "test_db.test_schema1.test_name",
        "test_db.test_schema2.test_name",
        ["A"],
    )
    assert compare.df1_name != compare.df2_name


def test_compare_table_setter_good(snowpark_session):
    data = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
    10000001234,123.4,George Michael Bluth,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Robert Loblaw,345.12,
    10000001238,1.05,Loose Seal Bluth,111,
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """
    df = pd.read_csv(StringIO(data), sep=",")
    database = snowpark_session.get_current_database().replace('"', "")
    schema = snowpark_session.get_current_schema().replace('"', "")
    full_table_name = f"{database}.{schema}"
    toy_table_name_1 = "DC_TOY_TABLE_1"
    toy_table_name_2 = "DC_TOY_TABLE_2"
    full_toy_table_name_1 = f"{full_table_name}.{toy_table_name_1}"
    full_toy_table_name_2 = f"{full_table_name}.{toy_table_name_2}"

    snowpark_session.write_pandas(
        df, toy_table_name_1, table_type="temp", auto_create_table=True, overwrite=True
    )
    snowpark_session.write_pandas(
        df, toy_table_name_2, table_type="temp", auto_create_table=True, overwrite=True
    )

    compare = SnowflakeCompare(
        snowpark_session,
        full_toy_table_name_1,
        full_toy_table_name_2,
        join_columns=["ACCT_ID"],
    )
    assert compare.df1.toPandas().equals(df)
    assert compare.join_columns == ["ACCT_ID"]


def test_compare_df_setter_bad(snowpark_session):
    pdf = pd.DataFrame([{"A": 1, "C": 2}, {"A": 2, "C": 2}])
    df = snowpark_session.createDataFrame(pdf)
    with raises(TypeError, match=r"DF1 must be a valid sp\.Dataframe"):
        SnowflakeCompare(snowpark_session, 3, 2, ["A"])
    with raises(ValueError, match="DF1 must have all columns from join_columns"):
        SnowflakeCompare(snowpark_session, df, df.select("*"), ["B"])
    pdf = pd.DataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 3}])
    df_dupe = snowpark_session.createDataFrame(pdf)
    pd.testing.assert_frame_equal(
        SnowflakeCompare(
            snowpark_session, df_dupe, df_dupe.select("*"), ["A", "B"]
        ).df1.toPandas(),
        pdf,
        check_dtype=False,
    )


def test_compare_df_setter_good(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 2}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 3}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["A"])
    assert compare.df1.toPandas().equals(df1.toPandas())
    assert compare.join_columns == ["A"]
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["A", "B"])
    assert compare.df1.toPandas().equals(df1.toPandas())
    assert compare.join_columns == ["A", "B"]


def test_compare_df_setter_different_cases(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 2}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 3}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["A"])
    assert compare.df1.toPandas().equals(df1.toPandas())


def test_columns_overlap(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 2}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 3}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["A"])
    assert compare.df1_unq_columns() == set()
    assert compare.df2_unq_columns() == set()
    assert compare.intersect_columns() == {"A", "B"}


def test_columns_no_overlap(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2, "C": "HI"}, {"A": 2, "B": 2, "C": "YO"}]
    )
    df2 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2, "D": "OH"}, {"A": 2, "B": 3, "D": "YA"}]
    )
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["A"])
    assert compare.df1_unq_columns() == {"C"}
    assert compare.df2_unq_columns() == {"D"}
    assert compare.intersect_columns() == {"A", "B"}


def test_columns_maintain_order_through_set_operations(snowpark_session):
    pdf1 = pd.DataFrame(
        {
            "JOIN": ["A", "B"],
            "F": [0, 0],
            "G": [1, 2],
            "B": [2, 2],
            "H": [3, 3],
            "A": [4, 4],
            "C": [-2, -3],
        }
    )
    pdf2 = pd.DataFrame(
        {
            "JOIN": ["A", "B"],
            "E": [0, 1],
            "H": [1, 2],
            "B": [2, 3],
            "A": [-1, -1],
            "G": [4, 4],
            "D": [-3, -2],
        }
    )
    df1 = snowpark_session.createDataFrame(pdf1)
    df2 = snowpark_session.createDataFrame(pdf2)
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["JOIN"])
    assert list(compare.df1_unq_columns()) == ["F", "C"]
    assert list(compare.df2_unq_columns()) == ["E", "D"]
    assert list(compare.intersect_columns()) == ["JOIN", "G", "B", "H", "A"]


def test_10k_rows(snowpark_session):
    rng = np.random.default_rng()
    pdf = pd.DataFrame(rng.integers(0, 100, size=(10000, 2)), columns=["B", "C"])
    pdf.reset_index(inplace=True)
    pdf.columns = ["A", "B", "C"]
    pdf2 = pdf.copy()
    pdf2["B"] = pdf2["B"] + 0.1
    df1 = snowpark_session.createDataFrame(pdf)
    df2 = snowpark_session.createDataFrame(pdf2)
    compare_tol = SnowflakeCompare(snowpark_session, df1, df2, ["A"], abs_tol=0.2)
    assert compare_tol.matches()
    assert compare_tol.df1_unq_rows.count() == 0
    assert compare_tol.df2_unq_rows.count() == 0
    assert compare_tol.intersect_columns() == {"A", "B", "C"}
    assert compare_tol.all_columns_match()
    assert compare_tol.all_rows_overlap()
    assert compare_tol.intersect_rows_match()

    compare_no_tol = SnowflakeCompare(snowpark_session, df1, df2, ["A"])
    assert not compare_no_tol.matches()
    assert compare_no_tol.df1_unq_rows.count() == 0
    assert compare_no_tol.df2_unq_rows.count() == 0
    assert compare_no_tol.intersect_columns() == {"A", "B", "C"}
    assert compare_no_tol.all_columns_match()
    assert compare_no_tol.all_rows_overlap()
    assert not compare_no_tol.intersect_rows_match()


def test_10k_rows_abs_tol_per_column(snowpark_session):
    rng = np.random.default_rng()
    pdf = pd.DataFrame(rng.integers(0, 100, size=(10000, 2)), columns=["B", "C"])
    pdf.reset_index(inplace=True)
    pdf.columns = ["A", "B", "C"]
    pdf2 = pdf.copy()
    pdf2["B"] = pdf2["B"] + 0.1
    df1 = snowpark_session.createDataFrame(pdf)
    df2 = snowpark_session.createDataFrame(pdf2)
    compare_tol = SnowflakeCompare(
        snowpark_session, df1, df2, ["A"], abs_tol={"B": 0.2}
    )
    assert compare_tol.matches()
    assert compare_tol.df1_unq_rows.count() == 0
    assert compare_tol.df2_unq_rows.count() == 0
    assert compare_tol.intersect_columns() == {"A", "B", "C"}
    assert compare_tol.all_columns_match()
    assert compare_tol.all_rows_overlap()
    assert compare_tol.intersect_rows_match()


def test_10k_rows_abs_tol_per_column_default(snowpark_session):
    rng = np.random.default_rng()
    pdf = pd.DataFrame(rng.integers(0, 100, size=(10000, 2)), columns=["B", "C"])
    pdf.reset_index(inplace=True)
    pdf.columns = ["A", "B", "C"]
    pdf2 = pdf.copy()
    pdf2["B"] = pdf2["B"] + 0.1
    pdf2["C"] = pdf2["C"] + 0.3
    df1 = snowpark_session.createDataFrame(pdf)
    df2 = snowpark_session.createDataFrame(pdf2)
    compare_tol = SnowflakeCompare(
        snowpark_session, df1, df2, ["A"], abs_tol={"c": 0.0, "default": 0.2}
    )
    assert not compare_tol.matches()
    assert compare_tol.df1_unq_rows.count() == 0
    assert compare_tol.df2_unq_rows.count() == 0
    assert compare_tol.intersect_columns() == {"A", "B", "C"}
    assert compare_tol.all_columns_match()
    assert compare_tol.all_rows_overlap()
    assert not compare_tol.intersect_rows_match()


def test_10k_rows_rel_tol_per_column(snowpark_session):
    rng = np.random.default_rng()
    pdf = pd.DataFrame(rng.integers(0, 100, size=(10000, 2)), columns=["B", "C"])
    pdf.reset_index(inplace=True)
    pdf.columns = ["A", "B", "C"]
    pdf2 = pdf.copy()
    pdf2["B"] = pdf2["B"] + 0.1
    df1 = snowpark_session.createDataFrame(pdf)
    df2 = snowpark_session.createDataFrame(pdf2)
    compare_tol = SnowflakeCompare(
        snowpark_session, df1, df2, ["A"], rel_tol={"B": 1.0}
    )
    assert compare_tol.matches()
    assert compare_tol.df1_unq_rows.count() == 0
    assert compare_tol.df2_unq_rows.count() == 0
    assert compare_tol.intersect_columns() == {"A", "B", "C"}
    assert compare_tol.all_columns_match()
    assert compare_tol.all_rows_overlap()
    assert compare_tol.intersect_rows_match()


def test_10k_rows_rel_tol_per_column_default(snowpark_session):
    rng = np.random.default_rng()
    pdf = pd.DataFrame(rng.integers(0, 100, size=(10000, 2)), columns=["B", "C"])
    pdf.reset_index(inplace=True)
    pdf.columns = ["A", "B", "C"]
    pdf2 = pdf.copy()
    pdf2["B"] = pdf2["B"] + 0.1
    pdf2["C"] = pdf2["C"] + 0.1
    df1 = snowpark_session.createDataFrame(pdf)
    df2 = snowpark_session.createDataFrame(pdf2)
    compare_tol = SnowflakeCompare(
        snowpark_session, df1, df2, ["A"], rel_tol={"c": 0.0, "default": 1}
    )
    assert not compare_tol.matches()
    assert compare_tol.df1_unq_rows.count() == 0
    assert compare_tol.df2_unq_rows.count() == 0
    assert compare_tol.intersect_columns() == {"A", "B", "C"}
    assert compare_tol.all_columns_match()
    assert compare_tol.all_rows_overlap()
    assert not compare_tol.intersect_rows_match()


def test_subset(snowpark_session, caplog):
    caplog.set_level(logging.DEBUG)
    df1 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2, "C": "HI"}, {"A": 2, "B": 2, "C": "YO"}]
    )
    df2 = snowpark_session.createDataFrame([{"A": 1, "C": "HI"}])
    comp = SnowflakeCompare(snowpark_session, df1, df2, ["A"])
    assert comp.subset()


def test_not_subset(snowpark_session, caplog):
    caplog.set_level(logging.INFO)
    df1 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2, "C": "HI"}, {"A": 2, "B": 2, "C": "YO"}]
    )
    df2 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2, "C": "HI"}, {"A": 2, "B": 2, "C": "GREAT"}]
    )
    comp = SnowflakeCompare(snowpark_session, df1, df2, ["A"])
    assert not comp.subset()
    assert "C: 1 / 2 (50.00%) match" in caplog.text


def test_large_subset(snowpark_session):
    rng = np.random.default_rng()
    pdf = pd.DataFrame(rng.integers(0, 100, size=(10000, 2)), columns=["B", "C"])
    pdf.reset_index(inplace=True)
    pdf.columns = ["A", "B", "C"]
    pdf2 = pdf[["A", "B"]].head(50).copy()
    df1 = snowpark_session.createDataFrame(pdf)
    df2 = snowpark_session.createDataFrame(pdf2)
    comp = SnowflakeCompare(snowpark_session, df1, df2, ["A"])
    assert not comp.matches()
    assert comp.subset()


def test_string_joiner(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"AB": 1, "BC": 2}, {"AB": 2, "BC": 2}])
    df2 = snowpark_session.createDataFrame([{"AB": 1, "BC": 2}, {"AB": 2, "BC": 2}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, "AB")
    assert compare.matches()


def test_decimal_with_joins(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"A": Decimal("1"), "B": 2}, {"A": Decimal("2"), "B": 2}]
    )
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 2}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, "A")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_nulls(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"A": 1, "B": Decimal("2")}, {"A": 2, "B": Decimal("2")}]
    )
    df2 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2}, {"A": 2, "B": 2}, {"A": 3, "B": 2}]
    )
    compare = SnowflakeCompare(snowpark_session, df1, df2, "A")
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_joins(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": "HI", "B": 2}, {"A": "BYE", "B": 2}])
    df2 = snowpark_session.createDataFrame([{"A": "HI", "B": 2}, {"A": "BYE", "B": 2}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, "A")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_temp_column_name(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": "HI", "B": 2}, {"A": "BYE", "B": 2}])
    df2 = snowpark_session.createDataFrame(
        [{"A": "HI", "B": 2}, {"A": "BYE", "B": 2}, {"A": "back fo mo", "B": 3}]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_TEMP_0"


def test_temp_column_name_one_has(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"_TEMP_0": "HI", "B": 2}, {"_TEMP_0": "BYE", "B": 2}]
    )
    df2 = snowpark_session.createDataFrame(
        [{"A": "HI", "B": 2}, {"A": "BYE", "B": 2}, {"A": "back fo mo", "B": 3}]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_TEMP_1"


def test_temp_column_name_both_have_temp_1(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"_TEMP_0": "HI", "B": 2}, {"_TEMP_0": "BYE", "B": 2}]
    )
    df2 = snowpark_session.createDataFrame(
        [
            {"_TEMP_0": "HI", "B": 2},
            {"_TEMP_0": "BYE", "B": 2},
            {"A": "back fo mo", "B": 3},
        ]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_TEMP_1"


def test_temp_column_name_both_have_temp_2(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"_TEMP_0": "HI", "B": 2}, {"_TEMP_0": "BYE", "B": 2}]
    )
    df2 = snowpark_session.createDataFrame(
        [
            {"_TEMP_0": "HI", "B": 2},
            {"_TEMP_1": "BYE", "B": 2},
            {"A": "back fo mo", "B": 3},
        ]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_TEMP_2"


def test_temp_column_name_one_already(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"_TEMP_1": "HI", "B": 2}, {"_TEMP_1": "BYE", "B": 2}]
    )
    df2 = snowpark_session.createDataFrame(
        [
            {"_TEMP_1": "HI", "B": 2},
            {"_TEMP_1": "BYE", "B": 2},
            {"A": "back fo mo", "B": 3},
        ]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_TEMP_0"


# Duplicate testing!


def test_simple_dupes_one_field(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 2}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 2}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, join_columns=["A"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    compare.report()


def test_simple_dupes_two_fields(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 2, "C": 2}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 2, "C": 2}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, join_columns=["A", "B"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    compare.report()


def test_simple_dupes_one_field_two_vals_1(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 0}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 0}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, join_columns=["A"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    compare.report()


def test_simple_dupes_one_field_two_vals_2(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 0}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 0}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, join_columns=["A"])
    assert not compare.matches()
    assert compare.df1_unq_rows.count() == 1
    assert compare.df2_unq_rows.count() == 1
    assert compare.intersect_rows.count() == 1
    # Just render the report to make sure it renders.
    compare.report()


def test_simple_dupes_one_field_three_to_two_vals(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2}, {"A": 1, "B": 0}, {"A": 1, "B": 0}]
    )
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 0}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, join_columns=["A"])
    assert not compare.matches()
    assert compare.df1_unq_rows.count() == 1
    assert compare.df2_unq_rows.count() == 0
    assert compare.intersect_rows.count() == 2
    # Just render the report to make sure it renders.
    compare.report()
    assert "(First 1 Columns)" in compare.report(column_count=1)
    assert "(First 2 Columns)" in compare.report(column_count=2)


def test_dupes_from_real_data(snowpark_session):
    data = """ACCT_ID,ACCT_SFX_NUM,TRXN_POST_DT,TRXN_POST_SEQ_NUM,TRXN_AMT,TRXN_DT,DEBIT_CR_CD,CASH_ADV_TRXN_COMN_CNTRY_CD,MRCH_CATG_CD,MRCH_PSTL_CD,VISA_MAIL_PHN_CD,VISA_RQSTD_PMT_SVC_CD,MC_PMT_FACILITATOR_IDN_NUM
100,0,2017-06-17,1537019,30.64,2017-06-15,D,CAN,5812,M2N5P5,,,0.0
200,0,2017-06-24,1022477,485.32,2017-06-22,D,USA,4511,7114,7.0,1,
100,0,2017-06-17,1537039,2.73,2017-06-16,D,CAN,5812,M4J 1M9,,,0.0
200,0,2017-06-29,1049223,22.41,2017-06-28,D,USA,4789,21211,,A,
100,0,2017-06-17,1537029,34.05,2017-06-16,D,CAN,5812,M4E 2C7,,,0.0
200,0,2017-06-29,1049213,9.12,2017-06-28,D,CAN,5814,0,,,
100,0,2017-06-19,1646426,165.21,2017-06-17,D,CAN,5411,M4M 3H9,,,0.0
200,0,2017-06-30,1233082,28.54,2017-06-29,D,USA,4121,94105,7.0,G,
100,0,2017-06-19,1646436,17.87,2017-06-18,D,CAN,5812,M4J 1M9,,,0.0
200,0,2017-06-30,1233092,24.39,2017-06-29,D,USA,4121,94105,7.0,G,
100,0,2017-06-19,1646446,5.27,2017-06-17,D,CAN,5200,M4M 3G6,,,0.0
200,0,2017-06-30,1233102,61.8,2017-06-30,D,CAN,4121,0,,,
100,0,2017-06-20,1607573,41.99,2017-06-19,D,CAN,5661,M4C1M9,,,0.0
200,0,2017-07-01,1009403,2.31,2017-06-29,D,USA,5814,22102,,F,
100,0,2017-06-20,1607553,86.88,2017-06-19,D,CAN,4812,H2R3A8,,,0.0
200,0,2017-07-01,1009423,5.5,2017-06-29,D,USA,5812,2903,,F,
100,0,2017-06-20,1607563,25.17,2017-06-19,D,CAN,5641,M4C 1M9,,,0.0
200,0,2017-07-01,1009433,214.12,2017-06-29,D,USA,3640,20170,,A,
100,0,2017-06-20,1607593,1.67,2017-06-19,D,CAN,5814,M2N 6L7,,,0.0
200,0,2017-07-01,1009393,2.01,2017-06-29,D,USA,5814,22102,,F,"""
    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data), sep=","))
    df2 = df1.select("*")
    compare_acct = SnowflakeCompare(
        snowpark_session, df1, df2, join_columns=["ACCT_ID"]
    )
    assert compare_acct.matches()
    compare_acct.report()

    compare_unq = SnowflakeCompare(
        snowpark_session,
        df1,
        df2,
        join_columns=["ACCT_ID", "ACCT_SFX_NUM", "TRXN_POST_DT", "TRXN_POST_SEQ_NUM"],
    )
    assert compare_unq.matches()
    compare_unq.report()


def test_table_compare_from_real_data(snowpark_session):
    data = """ACCT_ID,ACCT_SFX_NUM,TRXN_POST_DT,TRXN_POST_SEQ_NUM,TRXN_AMT,TRXN_DT,DEBIT_CR_CD,CASH_ADV_TRXN_COMN_CNTRY_CD,MRCH_CATG_CD,MRCH_PSTL_CD,VISA_MAIL_PHN_CD,VISA_RQSTD_PMT_SVC_CD,MC_PMT_FACILITATOR_IDN_NUM
100,0,2017-06-17,1537019,30.64,2017-06-15,D,CAN,5812,M2N5P5,,,0.0
200,0,2017-06-24,1022477,485.32,2017-06-22,D,USA,4511,7114,7.0,1,
100,0,2017-06-17,1537039,2.73,2017-06-16,D,CAN,5812,M4J 1M9,,,0.0
200,0,2017-06-29,1049223,22.41,2017-06-28,D,USA,4789,21211,,A,
100,0,2017-06-17,1537029,34.05,2017-06-16,D,CAN,5812,M4E 2C7,,,0.0
200,0,2017-06-29,1049213,9.12,2017-06-28,D,CAN,5814,0,,,
100,0,2017-06-19,1646426,165.21,2017-06-17,D,CAN,5411,M4M 3H9,,,0.0
200,0,2017-06-30,1233082,28.54,2017-06-29,D,USA,4121,94105,7.0,G,
100,0,2017-06-19,1646436,17.87,2017-06-18,D,CAN,5812,M4J 1M9,,,0.0
200,0,2017-06-30,1233092,24.39,2017-06-29,D,USA,4121,94105,7.0,G,
100,0,2017-06-19,1646446,5.27,2017-06-17,D,CAN,5200,M4M 3G6,,,0.0
200,0,2017-06-30,1233102,61.8,2017-06-30,D,CAN,4121,0,,,
100,0,2017-06-20,1607573,41.99,2017-06-19,D,CAN,5661,M4C1M9,,,0.0
200,0,2017-07-01,1009403,2.31,2017-06-29,D,USA,5814,22102,,F,
100,0,2017-06-20,1607553,86.88,2017-06-19,D,CAN,4812,H2R3A8,,,0.0
200,0,2017-07-01,1009423,5.5,2017-06-29,D,USA,5812,2903,,F,
100,0,2017-06-20,1607563,25.17,2017-06-19,D,CAN,5641,M4C 1M9,,,0.0
200,0,2017-07-01,1009433,214.12,2017-06-29,D,USA,3640,20170,,A,
100,0,2017-06-20,1607593,1.67,2017-06-19,D,CAN,5814,M2N 6L7,,,0.0
200,0,2017-07-01,1009393,2.01,2017-06-29,D,USA,5814,22102,,F,"""
    df = pd.read_csv(StringIO(data), sep=",")
    database = snowpark_session.get_current_database().replace('"', "")
    schema = snowpark_session.get_current_schema().replace('"', "")
    full_table_name = f"{database}.{schema}"
    toy_table_name_1 = "DC_TOY_TABLE_1"
    toy_table_name_2 = "DC_TOY_TABLE_2"
    full_toy_table_name_1 = f"{full_table_name}.{toy_table_name_1}"
    full_toy_table_name_2 = f"{full_table_name}.{toy_table_name_2}"

    snowpark_session.write_pandas(
        df, toy_table_name_1, table_type="temp", auto_create_table=True, overwrite=True
    )
    snowpark_session.write_pandas(
        df, toy_table_name_2, table_type="temp", auto_create_table=True, overwrite=True
    )

    compare_acct = SnowflakeCompare(
        snowpark_session,
        full_toy_table_name_1,
        full_toy_table_name_2,
        join_columns=["ACCT_ID"],
    )
    assert compare_acct.matches()
    compare_acct.report()

    compare_unq = SnowflakeCompare(
        snowpark_session,
        full_toy_table_name_1,
        full_toy_table_name_2,
        join_columns=["ACCT_ID", "ACCT_SFX_NUM", "TRXN_POST_DT", "TRXN_POST_SEQ_NUM"],
    )
    assert compare_unq.matches()
    compare_unq.report()


def test_strings_with_joins_with_ignore_spaces(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"A": "HI", "B": " A"}, {"A": "BYE", "B": "A"}]
    )
    df2 = snowpark_session.createDataFrame(
        [{"A": "HI", "B": "A"}, {"A": "BYE", "B": "A "}]
    )
    compare = SnowflakeCompare(snowpark_session, df1, df2, "A", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = SnowflakeCompare(snowpark_session, df1, df2, "A", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_joins_with_ignore_spaces(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": " A"}, {"A": 2, "B": "A"}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": "A"}, {"A": 2, "B": "A "}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, "A", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = SnowflakeCompare(snowpark_session, df1, df2, "A", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_joins_with_ignore_spaces(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": " A"}, {"A": 2, "B": "A"}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": "A"}, {"A": 2, "B": "A "}])

    compare = SnowflakeCompare(snowpark_session, df1, df2, "A", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_joins_with_insensitive_lowercase_cols(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"a": 1, "B": "A"}, {"a": 2, "B": "A"}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": "A"}, {"A": 2, "B": "A"}])

    compare = SnowflakeCompare(snowpark_session, df1, df2, "A")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()

    df1 = snowpark_session.createDataFrame([{"A": 1, "B": "A"}, {"A": 2, "B": "A"}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": "A"}, {"A": 2, "B": "A"}])

    compare = SnowflakeCompare(snowpark_session, df1, df2, "a")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_joins_with_sensitive_lowercase_cols(snowpark_session):
    df1 = snowpark_session.createDataFrame([{'"a"': 1, "B": "A"}, {'"a"': 2, "B": "A"}])
    df2 = snowpark_session.createDataFrame([{'"a"': 1, "B": "A"}, {'"a"': 2, "B": "A"}])

    compare = SnowflakeCompare(snowpark_session, df1, df2, '"a"')
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_full_join_counts_all_matches(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 2}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 2}])
    compare = SnowflakeCompare(
        snowpark_session, df1, df2, ["A", "B"], ignore_spaces=False
    )
    assert compare.count_matching_rows() == 2


def test_strings_with_ignore_spaces_and_join_columns(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"A": "HI", "B": "A"}, {"A": "BYE", "B": "A"}]
    )
    df2 = snowpark_session.createDataFrame(
        [{"A": " HI ", "B": "A"}, {"A": " BYE ", "B": "A"}]
    )
    compare = SnowflakeCompare(snowpark_session, df1, df2, "A", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.count_matching_rows() == 0

    compare = SnowflakeCompare(snowpark_session, df1, df2, "A", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2


def test_integers_with_ignore_spaces_and_join_columns(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": "A"}, {"A": 2, "B": "A"}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": "A"}, {"A": 2, "B": "A"}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, "A", ignore_spaces=False)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2

    compare = SnowflakeCompare(snowpark_session, df1, df2, "A", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2


def test_sample_mismatch(snowpark_session):
    data1 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """

    data2 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
    10000001234,123.4,George Michael Bluth,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Robert Loblaw,345.12,
    10000001238,1.05,Loose Seal Bluth,111,
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """

    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))

    compare = SnowflakeCompare(snowpark_session, df1, df2, "ACCT_ID")

    output = compare.sample_mismatch(column="NAME", sample_count=1).toPandas()
    assert output.shape[0] == 1
    assert (output.NAME_DF1 != output.NAME_DF2).all()

    output = compare.sample_mismatch(column="NAME", sample_count=2).toPandas()
    assert output.shape[0] == 2
    assert (output.NAME_DF1 != output.NAME_DF2).all()

    output = compare.sample_mismatch(column="NAME", sample_count=3).toPandas()
    assert output.shape[0] == 2
    assert (output.NAME_DF1 != output.NAME_DF2).all()


def test_all_mismatch_not_ignore_matching_cols_no_cols_matching(snowpark_session):
    data1 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """

    data2 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
    10000001234,123.4,George Michael Bluth,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Robert Loblaw,345.12,
    10000001238,1.05,Loose Seal Bluth,111,
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """
    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = SnowflakeCompare(snowpark_session, df1, df2, "ACCT_ID")

    output = compare.all_mismatch().toPandas()
    assert output.shape[0] == 4
    assert output.shape[1] == 9

    assert (output.NAME_DF1 != output.NAME_DF2).values.sum() == 2
    assert (~(output.NAME_DF1 != output.NAME_DF2)).values.sum() == 2

    assert (output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2).values.sum() == 1
    assert (~(output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2)).values.sum() == 3

    assert (output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2).values.sum() == 3
    assert (~(output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2)).values.sum() == 1

    assert (output.DATE_FLD_DF1 != output.DATE_FLD_DF2).values.sum() == 4
    assert (~(output.DATE_FLD_DF1 != output.DATE_FLD_DF2)).values.sum() == 0


def test_all_mismatch_not_ignore_matching_cols_some_cols_matching(snowpark_session):
    # Columns dollar_amt and name are matching
    data1 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1,2017-01-01
        10000001236,1345,George Bluth,,2017-01-01
        10000001237,123456,Bob Loblaw,345.12,2017-01-01
        10000001239,1.05,Lucille Bluth,,2017-01-01
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """

    data2 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.155,
        10000001235,0.45,Michael Bluth,,
        10000001236,1345,George Bluth,1,
        10000001237,123456,Bob Loblaw,345.12,
        10000001238,1.05,Lucille Bluth,111,
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """
    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = SnowflakeCompare(snowpark_session, df1, df2, "ACCT_ID")

    output = compare.all_mismatch().toPandas()
    assert output.shape[0] == 4
    assert output.shape[1] == 9

    assert (output.NAME_DF1 != output.NAME_DF2).values.sum() == 0
    assert (~(output.NAME_DF1 != output.NAME_DF2)).values.sum() == 4

    assert (output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2).values.sum() == 0
    assert (~(output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2)).values.sum() == 4

    assert (output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2).values.sum() == 3
    assert (~(output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2)).values.sum() == 1

    assert (output.DATE_FLD_DF1 != output.DATE_FLD_DF2).values.sum() == 4
    assert (~(output.DATE_FLD_DF1 != output.DATE_FLD_DF2)).values.sum() == 0


def test_all_mismatch_ignore_matching_cols_some_cols_matching_diff_rows(
    snowpark_session,
):
    # Case where there are rows on either dataset which don't match up.
    # Columns dollar_amt and name are matching
    data1 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1,2017-01-01
        10000001236,1345,George Bluth,,2017-01-01
        10000001237,123456,Bob Loblaw,345.12,2017-01-01
        10000001239,1.05,Lucille Bluth,,2017-01-01
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        10000001241,1111.05,Lucille Bluth,
        """

    data2 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.155,
        10000001235,0.45,Michael Bluth,,
        10000001236,1345,George Bluth,1,
        10000001237,123456,Bob Loblaw,345.12,
        10000001238,1.05,Lucille Bluth,111,
        """
    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = SnowflakeCompare(snowpark_session, df1, df2, "ACCT_ID")

    output = compare.all_mismatch(ignore_matching_cols=True).toPandas()

    assert output.shape[0] == 4
    assert output.shape[1] == 5

    assert (output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2).values.sum() == 3
    assert (~(output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2)).values.sum() == 1

    assert (output.DATE_FLD_DF1 != output.DATE_FLD_DF2).values.sum() == 4
    assert (~(output.DATE_FLD_DF1 != output.DATE_FLD_DF2)).values.sum() == 0

    assert not ("NAME_DF1" in output and "NAME_DF2" in output)
    assert not ("DOLLAR_AMT_DF1" in output and "DOLLAR_AMT_DF1" in output)


def test_all_mismatch_ignore_matching_cols_some_cols_matching(snowpark_session):
    # Columns dollar_amt and name are matching
    data1 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1,2017-01-01
        10000001236,1345,George Bluth,,2017-01-01
        10000001237,123456,Bob Loblaw,345.12,2017-01-01
        10000001239,1.05,Lucille Bluth,,2017-01-01
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """

    data2 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.155,
        10000001235,0.45,Michael Bluth,,
        10000001236,1345,George Bluth,1,
        10000001237,123456,Bob Loblaw,345.12,
        10000001238,1.05,Lucille Bluth,111,
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """
    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = SnowflakeCompare(snowpark_session, df1, df2, "ACCT_ID")

    output = compare.all_mismatch(ignore_matching_cols=True).toPandas()

    assert output.shape[0] == 4
    assert output.shape[1] == 5

    assert (output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2).values.sum() == 3
    assert (~(output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2)).values.sum() == 1

    assert (output.DATE_FLD_DF1 != output.DATE_FLD_DF2).values.sum() == 4
    assert (~(output.DATE_FLD_DF1 != output.DATE_FLD_DF2)).values.sum() == 0

    assert not ("NAME_DF1" in output and "NAME_DF2" in output)
    assert not ("DOLLAR_AMT_DF1" in output and "DOLLAR_AMT_DF1" in output)


def test_all_mismatch_ignore_matching_cols_no_cols_matching(snowpark_session):
    data1 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1,2017-01-01
        10000001236,1345,George Bluth,,2017-01-01
        10000001237,123456,Bob Loblaw,345.12,2017-01-01
        10000001239,1.05,Lucille Bluth,,2017-01-01
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """

    data2 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.4,George Michael Bluth,14530.155,
        10000001235,0.45,Michael Bluth,,
        10000001236,1345,George Bluth,1,
        10000001237,123456,Robert Loblaw,345.12,
        10000001238,1.05,Loose Seal Bluth,111,
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """
    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = SnowflakeCompare(snowpark_session, df1, df2, "ACCT_ID")

    output = compare.all_mismatch().toPandas()
    assert output.shape[0] == 4
    assert output.shape[1] == 9

    assert (output.NAME_DF1 != output.NAME_DF2).values.sum() == 2
    assert (~(output.NAME_DF1 != output.NAME_DF2)).values.sum() == 2

    assert (output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2).values.sum() == 1
    assert (~(output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2)).values.sum() == 3

    assert (output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2).values.sum() == 3
    assert (~(output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2)).values.sum() == 1

    assert (output.DATE_FLD_DF1 != output.DATE_FLD_DF2).values.sum() == 4
    assert (~(output.DATE_FLD_DF1 != output.DATE_FLD_DF2)).values.sum() == 0


def test_all_mismatch_ignore_matching_cols_no_cols_matching_abs_tol_float(
    snowpark_session,
):
    data1 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1,2017-01-01
        10000001236,1345,George Bluth,,2017-01-01
        10000001237,123456,Bob Loblaw,345.12,2017-01-01
        10000001239,1.05,Lucille Bluth,,2017-01-01
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """

    data2 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.4,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1.04,2017-01-01
        10000001236,1345,George Bluth,1,
        10000001237,123456,Robert Loblaw,345.12,
        10000001238,1.05,Loose Seal Bluth,111,
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """
    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = SnowflakeCompare(snowpark_session, df1, df2, "ACCT_ID", 0.05)

    output = compare.all_mismatch().toPandas()
    assert output.shape[0] == 2
    assert output.shape[1] == 9

    assert (output.NAME_DF1 != output.NAME_DF2).values.sum() == 1
    assert (~(output.NAME_DF1 != output.NAME_DF2)).values.sum() == 1

    assert (output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2).values.sum() == 0
    assert (~(output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2)).values.sum() == 2

    assert (output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2).values.sum() == 1
    assert (~(output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2)).values.sum() == 1

    assert (output.DATE_FLD_DF1 != output.DATE_FLD_DF2).values.sum() == 2
    assert (~(output.DATE_FLD_DF1 != output.DATE_FLD_DF2)).values.sum() == 0


def test_all_mismatch_ignore_matching_cols_no_cols_matching_abs_tol_dict(
    snowpark_session,
):
    data1 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1,2017-01-01
        10000001236,1345,George Bluth,,2017-01-01
        10000001237,123456,Bob Loblaw,345.12,2017-01-01
        10000001239,1.05,Lucille Bluth,,2017-01-01
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """

    data2 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.4,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1.05,2017-01-01
        10000001236,1345,George Bluth,1,
        10000001237,123456,Robert Loblaw,345.12,
        10000001238,1.05,Loose Seal Bluth,111,
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """
    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = SnowflakeCompare(
        snowpark_session, df1, df2, "ACCT_ID", {"DOLLAR_AMT": 0.05}
    )

    output = compare.all_mismatch().toPandas()
    assert output.shape[0] == 3
    assert output.shape[1] == 9

    assert (output.NAME_DF1 != output.NAME_DF2).values.sum() == 1
    assert (~(output.NAME_DF1 != output.NAME_DF2)).values.sum() == 2

    assert (output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2).values.sum() == 0
    assert (~(output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2)).values.sum() == 3

    assert (output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2).values.sum() == 2
    assert (~(output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2)).values.sum() == 1

    assert (output.DATE_FLD_DF1 != output.DATE_FLD_DF2).values.sum() == 2
    assert (~(output.DATE_FLD_DF1 != output.DATE_FLD_DF2)).values.sum() == 1


def test_all_mismatch_ignore_matching_cols_no_cols_matching_rel_tol_float(
    snowpark_session,
):
    data1 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1,2017-01-01
        10000001236,1345,George Bluth,,2017-01-01
        10000001237,123456,Bob Loblaw,345.12,2017-01-01
        10000001239,1.05,Lucille Bluth,,2017-01-01
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """

    data2 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.4,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1.04,2017-01-01
        10000001236,1345,George Bluth,1,
        10000001237,123456,Robert Loblaw,345.12,
        10000001238,1.05,Loose Seal Bluth,111,
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """
    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = SnowflakeCompare(snowpark_session, df1, df2, "ACCT_ID", rel_tol=0.1)

    output = compare.all_mismatch().toPandas()
    assert output.shape[0] == 2
    assert output.shape[1] == 9

    assert (output.NAME_DF1 != output.NAME_DF2).values.sum() == 1
    assert (~(output.NAME_DF1 != output.NAME_DF2)).values.sum() == 1

    assert (output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2).values.sum() == 0
    assert (~(output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2)).values.sum() == 2

    assert (output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2).values.sum() == 1
    assert (~(output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2)).values.sum() == 1

    assert (output.DATE_FLD_DF1 != output.DATE_FLD_DF2).values.sum() == 2
    assert (~(output.DATE_FLD_DF1 != output.DATE_FLD_DF2)).values.sum() == 0


def test_all_mismatch_ignore_matching_cols_no_cols_matching_rel_tol_dict(
    snowpark_session,
):
    data1 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.45,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1,2017-01-01
        10000001236,1345,George Bluth,,2017-01-01
        10000001237,123456,Bob Loblaw,345.12,2017-01-01
        10000001239,1.05,Lucille Bluth,,2017-01-01
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """

    data2 = """ACCT_ID,DOLLAR_AMT,NAME,FLOAT_FLD,DATE_FLD
        10000001234,123.4,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1.05,2017-01-01
        10000001236,1345,George Bluth,1,
        10000001237,123456,Robert Loblaw,345.12,
        10000001238,1.05,Loose Seal Bluth,111,
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """
    df1 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data1), sep=","))
    df2 = snowpark_session.createDataFrame(pd.read_csv(StringIO(data2), sep=","))
    compare = SnowflakeCompare(
        snowpark_session,
        df1,
        df2,
        "ACCT_ID",
        abs_tol={"FLOAT_FLD": 0.01},
        rel_tol={"DOLLAR_AMT": 0.10, "FLOAT_FLD": 0.02},
    )

    output = compare.all_mismatch().toPandas()
    r = compare.report()

    assert output.shape[0] == 3
    assert output.shape[1] == 9

    assert (output.NAME_DF1 != output.NAME_DF2).values.sum() == 1
    assert (~(output.NAME_DF1 != output.NAME_DF2)).values.sum() == 2

    assert (output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2).values.sum() == 0
    assert (~(output.DOLLAR_AMT_DF1 != output.DOLLAR_AMT_DF2)).values.sum() == 3

    assert (output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2).values.sum() == 2
    assert (~(output.FLOAT_FLD_DF1 != output.FLOAT_FLD_DF2)).values.sum() == 1

    assert (output.DATE_FLD_DF1 != output.DATE_FLD_DF2).values.sum() == 2
    assert (~(output.DATE_FLD_DF1 != output.DATE_FLD_DF2)).values.sum() == 1


@pytest.mark.parametrize(
    "column, expected",
    [
        ("BASE", 0),
        ("FLOATS", 0.2),
        ("DECIMALS", 0.1),
        ("NULL_FLOATS", 0.1),
        ("STRINGS", 0.1),
        ("INFINITY", np.inf),
    ],
)
def test_calculate_max_diff(snowpark_session, column, expected):
    pdf = pd.DataFrame(
        {
            "BASE": [1, 1, 1, 1, 1],
            "FLOATS": [1.1, 1.1, 1.1, 1.2, 0.9],
            "DECIMALS": [
                Decimal("1.1"),
                Decimal("1.1"),
                Decimal("1.1"),
                Decimal("1.1"),
                Decimal("1.1"),
            ],
            "NULL_FLOATS": [np.nan, 1.1, 1, 1, 1],
            "STRINGS": ["1", "1", "1", "1.1", "1"],
            "INFINITY": [1, 1, 1, 1, np.inf],
        }
    )
    MAX_DIFF_DF = snowpark_session.createDataFrame(pdf)
    assert np.isclose(
        calculate_max_diff(MAX_DIFF_DF, "BASE", column).result()[0][0],
        expected,
    )


def test_dupes_with_nulls_strings(snowpark_session):
    pdf1 = pd.DataFrame(
        {
            "FLD_1": [1, 2, 2, 3, 3, 4, 5, 5],
            "FLD_2": ["A", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "FLD_3": [1, 2, 2, 3, 3, 4, 5, 5],
        }
    )
    pdf2 = pd.DataFrame(
        {
            "FLD_1": [1, 2, 3, 4, 5],
            "FLD_2": ["A", np.nan, np.nan, np.nan, np.nan],
            "FLD_3": [1, 2, 3, 4, 5],
        }
    )
    df1 = snowpark_session.createDataFrame(pdf1)
    df2 = snowpark_session.createDataFrame(pdf2)
    comp = SnowflakeCompare(snowpark_session, df1, df2, join_columns=["FLD_1", "FLD_2"])
    assert comp.subset()


def test_dupes_with_nulls_ints(snowpark_session):
    pdf1 = pd.DataFrame(
        {
            "FLD_1": [1, 2, 2, 3, 3, 4, 5, 5],
            "FLD_2": [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "FLD_3": [1, 2, 2, 3, 3, 4, 5, 5],
        }
    )
    pdf2 = pd.DataFrame(
        {
            "FLD_1": [1, 2, 3, 4, 5],
            "FLD_2": [1, np.nan, np.nan, np.nan, np.nan],
            "FLD_3": [1, 2, 3, 4, 5],
        }
    )
    df1 = snowpark_session.createDataFrame(pdf1)
    df2 = snowpark_session.createDataFrame(pdf2)
    comp = SnowflakeCompare(snowpark_session, df1, df2, join_columns=["FLD_1", "FLD_2"])
    assert comp.subset()


def test_generate_id_within_group(snowpark_session):
    matrix = [
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "__INDEX": [1, 2, 3]}),
            pd.Series([0, 0, 0]),
        ),
        (
            pd.DataFrame(
                {
                    "A": ["A", "A", "DATACOMPY_NULL"],
                    "B": [1, 1, 2],
                    "__INDEX": [1, 2, 3],
                }
            ),
            pd.Series([0, 1, 0]),
        ),
        (
            pd.DataFrame({"A": [-999, 2, 3], "B": [1, 2, 3], "__INDEX": [1, 2, 3]}),
            pd.Series([0, 0, 0]),
        ),
        (
            pd.DataFrame(
                {"A": [1, np.nan, np.nan], "B": [1, 2, 2], "__INDEX": [1, 2, 3]}
            ),
            pd.Series([0, 0, 1]),
        ),
        (
            pd.DataFrame(
                {"A": ["1", np.nan, np.nan], "B": ["1", "2", "2"], "__INDEX": [1, 2, 3]}
            ),
            pd.Series([0, 0, 1]),
        ),
        (
            pd.DataFrame(
                {
                    "A": [datetime(2018, 1, 1), np.nan, np.nan],
                    "B": ["1", "2", "2"],
                    "__INDEX": [1, 2, 3],
                }
            ),
            pd.Series([0, 0, 1]),
        ),
    ]
    for i in matrix:
        dataframe = i[0]
        expected = i[1]
        actual = (
            _generate_id_within_group(
                snowpark_session.createDataFrame(dataframe), ["A", "B"], "_TEMP_0"
            )
            .orderBy("__INDEX")
            .select("_TEMP_0")
            .toPandas()
        )
        assert (actual["_TEMP_0"] == expected).all()


def test_generate_id_within_group_single_join(snowpark_session):
    dataframe = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2, "__INDEX": 1}, {"A": 1, "B": 2, "__INDEX": 2}]
    )
    expected = pd.Series([0, 1])
    actual = (
        _generate_id_within_group(dataframe, ["A"], "_TEMP_0")
        .orderBy("__INDEX")
        .select("_TEMP_0")
    ).toPandas()
    assert (actual["_TEMP_0"] == expected).all()


@mock.patch("datacompy.snowflake.render")
@mock.patch("datacompy.snowflake.save_html_report")
def test_save_html(mock_save_html, mock_render, snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 2}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 2}])
    compare = SnowflakeCompare(snowpark_session, df1, df2, join_columns=["A"])

    # Test without HTML file
    compare.report()
    mock_render.assert_called_once()
    mock_save_html.assert_not_called()

    mock_render.reset_mock()
    mock_save_html.reset_mock()

    # Test with HTML file
    compare.report(html_file="test.html")
    mock_render.assert_called_once()
    mock_save_html.assert_called_once()
    args, _ = mock_save_html.call_args
    assert len(args) == 2
    assert args[1] == "test.html"  # The filename


def test_full_join_counts_no_matches(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 3}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 4}, {"A": 1, "B": 5}])
    compare = SnowflakeCompare(
        snowpark_session, df1, df2, ["A", "B"], ignore_spaces=False
    )
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert not compare.intersect_rows_match()
    assert compare.count_matching_rows() == 0
    assert_frame_equal(
        compare.sample_mismatch(column="A")
        .toPandas()
        .sort_values("A")
        .reset_index(drop=True),
        pd.DataFrame([{"A": 1}, {"A": 1}, {"A": 1}, {"A": 1}]),
    )
    assert_frame_equal(
        compare.sample_mismatch(column="B")
        .toPandas()
        .sort_values("B")
        .reset_index(drop=True),
        pd.DataFrame([{"B": 2}, {"B": 3}, {"B": 4}, {"B": 5}]),
    )
    assert_frame_equal(
        compare.all_mismatch()
        .toPandas()
        .sort_values(["A", "B"])
        .reset_index(drop=True),
        pd.DataFrame(
            [{"A": 1, "B": 2}, {"A": 1, "B": 3}, {"A": 1, "B": 4}, {"A": 1, "B": 5}]
        ),
    )


def test_full_join_counts_some_matches(snowpark_session):
    df1 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 3}])
    df2 = snowpark_session.createDataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 5}])
    compare = SnowflakeCompare(
        snowpark_session, df1, df2, ["A", "B"], ignore_spaces=False
    )
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 1
    assert_frame_equal(
        compare.sample_mismatch(column="A")
        .toPandas()
        .sort_values("A")
        .reset_index(drop=True),
        pd.DataFrame([{"A": 1}, {"A": 1}]),
    )
    assert_frame_equal(
        compare.sample_mismatch(column="B")
        .toPandas()
        .sort_values("B")
        .reset_index(drop=True),
        pd.DataFrame([{"B": 3}, {"B": 5}]),
    )
    assert_frame_equal(
        compare.all_mismatch()
        .toPandas()
        .sort_values(["A", "B"])
        .reset_index(drop=True),
        pd.DataFrame(
            [
                {"A": 1, "B": 3},
                {"A": 1, "B": 5},
            ]
        ),
    )


def test_non_full_join_counts_no_matches(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2, "C": 4}, {"A": 1, "B": 3, "C": 4}]
    )
    df2 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 4, "D": 5}, {"A": 1, "B": 5, "D": 5}]
    )
    compare = SnowflakeCompare(
        snowpark_session, df1, df2, ["A", "B"], ignore_spaces=False
    )
    assert not compare.matches()
    assert not compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert not compare.intersect_rows_match()
    assert compare.count_matching_rows() == 0
    assert_frame_equal(
        compare.sample_mismatch(column="A")
        .toPandas()
        .sort_values("A")
        .reset_index(drop=True),
        pd.DataFrame(
            [{"A": 1}, {"A": 1}, {"A": 1}, {"A": 1}],
        ),
    )
    assert_frame_equal(
        compare.sample_mismatch(column="B")
        .toPandas()
        .sort_values("B")
        .reset_index(drop=True),
        pd.DataFrame([{"B": 2}, {"B": 3}, {"B": 4}, {"B": 5}]),
    )
    assert_frame_equal(
        compare.all_mismatch()
        .toPandas()
        .sort_values(["A", "B"])
        .reset_index(drop=True),
        pd.DataFrame(
            [
                {"A": 1, "B": 2},
                {"A": 1, "B": 3},
                {"A": 1, "B": 4},
                {"A": 1, "B": 5},
            ]
        ),
    )


def test_non_full_join_counts_some_matches(snowpark_session):
    df1 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2, "C": 4}, {"A": 1, "B": 3, "C": 4}]
    )
    df2 = snowpark_session.createDataFrame(
        [{"A": 1, "B": 2, "D": 5}, {"A": 1, "B": 5, "D": 5}]
    )
    compare = SnowflakeCompare(
        snowpark_session, df1, df2, ["A", "B"], ignore_spaces=False
    )
    assert not compare.matches()
    assert not compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 1
    assert_frame_equal(
        compare.sample_mismatch(column="A")
        .toPandas()
        .sort_values("A")
        .reset_index(drop=True),
        pd.DataFrame([{"A": 1}, {"A": 1}]),
    )
    assert_frame_equal(
        compare.sample_mismatch(column="B")
        .toPandas()
        .sort_values("B")
        .reset_index(drop=True),
        pd.DataFrame([{"B": 3}, {"B": 5}]),
    )
    assert_frame_equal(
        compare.all_mismatch()
        .toPandas()
        .sort_values(["A", "B"])
        .reset_index(drop=True),
        pd.DataFrame(
            [
                {"A": 1, "B": 3},
                {"A": 1, "B": 5},
            ]
        ),
    )


def test_custom_template_usage(snowpark_session):
    """Test using a custom template with template_path parameter."""
    df1 = snowpark_session.createDataFrame([("a", 1), ("b", 2)], ["id", "value"])
    df2 = snowpark_session.createDataFrame([("a", 1), ("b", 3)], ["id", "value"])
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["id"])

    # Create a simple test template
    with tempfile.NamedTemporaryFile(suffix=".j2", delete=False, mode="w") as tmp:
        tmp.write("Custom Template\n")
        tmp.write(
            "Columns: {{ mismatch_stats.stats|map(attribute='column')|join(', ') if mismatch_stats.has_mismatches else '' }}\n"
        )
        tmp.write(
            "Matches: "
            "{% if mismatch_stats.has_mismatches %}"
            "{% for col in mismatch_stats.stats %}"
            "{% if col.unequal_cnt > 0 %}False{% else %}True{% endif %}"
            "{% endfor %}"
            "{% else %}All match{% endif %}"
        )
        template_path = tmp.name

    try:
        # Test with custom template
        result = compare.report(template_path=template_path)
        assert "Custom Template" in result
        # Should list the column with mismatches (value)
        assert "VALUE" in result
        # Should show False for column value (has mismatches)
        assert "False" in result
    finally:
        # Clean up the temporary file
        if os.path.exists(template_path):
            os.unlink(template_path)


def test_template_without_extension(snowpark_session):
    """Test that template_path works without .j2 extension."""
    df1 = snowpark_session.createDataFrame([("a", 1), ("b", 2)], ["id", "value"])
    df2 = snowpark_session.createDataFrame([("a", 1), ("b", 3)], ["id", "value"])
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["id"])

    # Create a test template without .j2 extension
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
        tmp.write("Test Template")
        template_path = tmp.name

    try:
        result = compare.report(template_path=template_path)
        assert "Test Template" in result
    finally:
        if os.path.exists(template_path):
            os.unlink(template_path)


def test_nonexistent_template(snowpark_session):
    """Test that a clear error is raised when template file doesn't exist."""
    df1 = snowpark_session.createDataFrame([("a", 1), ("b", 2)], ["id", "value"])
    df2 = snowpark_session.createDataFrame([("a", 1), ("b", 3)], ["id", "value"])
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["id"])

    with pytest.raises(Exception) as exc_info:
        compare.report(template_path="nonexistent_template.j2")
    # Check that the error message is helpful
    assert "Template not found" in str(
        exc_info.value
    ) or "nonexistent_template.j2" in str(exc_info.value)


def test_template_context_variables(snowpark_session):
    """Test that all expected context variables are available in the template."""
    df1 = snowpark_session.createDataFrame([("a", 1), ("b", 2)], ["id", "value"])
    df2 = snowpark_session.createDataFrame([("a", 1), ("b", 3)], ["id", "value"])
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["id"])

    # Create a test template that checks for expected variables
    with tempfile.NamedTemporaryFile(suffix=".j2", delete=False, mode="w") as tmp:
        tmp.write(
            "{% if mismatch_stats is defined and df1_name is defined and df2_name is defined %}"
        )
        tmp.write("All required variables present\n")
        tmp.write("{% else %}")
        tmp.write("Missing required variables\n")
        tmp.write("{% endif %}")
        tmp.write(
            "Columns: {{ mismatch_stats.stats|map(attribute='column')|join(', ') if mismatch_stats.has_mismatches else '' }}"
        )
        template_path = tmp.name

    try:
        result = compare.report(template_path=template_path)
        assert "All required variables present" in result
        # Should list the column with mismatches (value)
        assert "VALUE" in result
    finally:
        if os.path.exists(template_path):
            os.unlink(template_path)


@mock.patch("datacompy.snowflake.save_html_report")
@mock.patch("datacompy.snowflake.render")
def test_html_report_generation(mock_render, mock_save_html, snowpark_session):
    """Test that HTML reports can be generated and saved to a file."""
    df1 = snowpark_session.createDataFrame([("a", 1), ("b", 2)], ["id", "value"])
    df2 = snowpark_session.createDataFrame([("a", 1), ("b", 3)], ["id", "value"])
    compare = SnowflakeCompare(snowpark_session, df1, df2, ["id"])

    # Mock the render function to return a test string
    mock_render.return_value = "<html><body>Test Report</body></html>"

    # Create a temporary file for the HTML output
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        html_file = tmp.name

    try:
        # Call report with html_file parameter
        result = compare.report(html_file=html_file)

        # Check that save_html_report was called with the correct arguments
        mock_save_html.assert_called_once_with(
            "<html><body>Test Report</body></html>", html_file
        )
        # Check that the result is the rendered template
        assert result == "<html><body>Test Report</body></html>"
    finally:
        # Clean up the temporary file
        if os.path.exists(html_file):
            os.unlink(html_file)
