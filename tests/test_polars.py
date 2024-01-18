#
# Copyright 2024 Capital One Services, LLC
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
Testing out the datacompy functionality
"""
import io
import logging
import sys
from datetime import datetime
from decimal import Decimal
from unittest import mock

import numpy as np
import pytest
from pytest import raises

pytest.importorskip("polars")

import polars as pl
from polars.exceptions import ComputeError, DuplicateError
from polars.testing import assert_series_equal

from datacompy import PolarsCompare
from datacompy.polars import (
    calculate_max_diff,
    columns_equal,
    generate_id_within_group,
    temp_column_name,
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def test_numeric_columns_equal_abs():
    data = """a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""
    df = pl.read_csv(io.StringIO(data), separator="|", null_values=["NULL"])
    actual_out = columns_equal(df["a"], df["b"], abs_tol=0.2)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_numeric_columns_equal_rel():
    data = """a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""
    df = pl.read_csv(io.StringIO(data), separator="|", null_values=["NULL"])
    actual_out = columns_equal(df["a"], df["b"], rel_tol=0.2)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_string_columns_equal():
    data = """a|b|expected
Hi|Hi|True
Yo|Yo|True
Hey|Hey |False
résumé|resume|False
résumé|résumé|True
💩|💩|True
💩|🤔|False
" "|" "|True
"  "|" "|False
datacompy|DataComPy|False
something|NULL|False
NULL|something|False
NULL|NULL|True"""
    df = pl.read_csv(
        io.StringIO(data),
        separator="|",
        null_values=["NULL"],
        missing_utf8_is_empty_string=True,
    )
    actual_out = columns_equal(df["a"], df["b"], rel_tol=0.2)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_string_columns_equal_with_ignore_spaces():
    data = """a|b|expected
Hi|Hi|True
Yo|Yo|True
Hey|Hey |True
résumé|resume|False
résumé|résumé|True
💩|💩|True
💩|🤔|False
" "|" "|True
"  "|"       "|True
datacompy|DataComPy|False
something||False
NULL|something|False
NULL|NULL|True"""
    df = pl.read_csv(
        io.StringIO(data),
        separator="|",
        null_values=["NULL"],
        missing_utf8_is_empty_string=True,
    )
    actual_out = columns_equal(df["a"], df["b"], rel_tol=0.2, ignore_spaces=True)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_string_columns_equal_with_ignore_spaces_and_case():
    data = """a|b|expected
Hi|Hi|True
Yo|Yo|True
Hey|Hey |True
résumé|resume|False
résumé|résumé|True
💩|💩|True
💩|🤔|False
" "|" "|True
"  "|"       "|True
datacompy|DataComPy|True
something||False
NULL|something|False
NULL|NULL|True"""
    df = pl.read_csv(
        io.StringIO(data),
        separator="|",
        null_values=["NULL"],
        missing_utf8_is_empty_string=True,
    )
    actual_out = columns_equal(
        df["a"], df["b"], rel_tol=0.2, ignore_spaces=True, ignore_case=True
    )
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_date_columns_equal():
    data = """a|b|expected
2017-01-01|2017-01-01|True
2017-01-02|2017-01-02|True
2017-10-01|2017-10-10|False
2017-01-01||False
|2017-01-01|False
||True"""
    df = pl.read_csv(
        io.StringIO(data),
        separator="|",
        null_values=["NULL"],
        missing_utf8_is_empty_string=True,
    )
    # First compare just the strings
    actual_out = columns_equal(df["a"], df["b"], rel_tol=0.2)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    col_a = df["a"].str.to_datetime(strict=False)
    col_b = df["b"].str.to_datetime(strict=False)
    actual_out = columns_equal(col_a, col_b, rel_tol=0.2)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_date_columns_equal_with_ignore_spaces():
    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    df = pl.read_csv(
        io.StringIO(data),
        separator="|",
        null_values=["NULL"],
        missing_utf8_is_empty_string=True,
    )
    # First compare just the strings
    actual_out = columns_equal(df["a"], df["b"], rel_tol=0.2, ignore_spaces=True)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    col_a = df["a"].str.strip_chars().str.to_date(strict=False)
    col_b = df["b"].str.strip_chars().str.to_date(strict=False)

    actual_out = columns_equal(col_a, col_b, rel_tol=0.2, ignore_spaces=True)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_date_columns_equal_with_ignore_spaces_and_case():
    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    df = pl.read_csv(
        io.StringIO(data),
        separator="|",
        null_values=["NULL"],
        missing_utf8_is_empty_string=True,
    )
    # First compare just the strings
    actual_out = columns_equal(
        df["a"], df["b"], rel_tol=0.2, ignore_spaces=True, ignore_case=True
    )
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    col_a = df["a"].str.strip_chars().str.to_date(strict=False)
    col_b = df["b"].str.strip_chars().str.to_date(strict=False)

    actual_out = columns_equal(col_a, col_b, rel_tol=0.2, ignore_spaces=True)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_date_columns_unequal():
    """I want datetime fields to match with dates stored as strings"""
    df = pl.DataFrame([{"a": "2017-01-01", "b": "2017-01-02"}, {"a": "2017-01-01"}])
    df = df.with_columns(df["a"].str.to_date().alias("a_dt"))
    df = df.with_columns(df["b"].str.to_date().alias("b_dt"))
    assert columns_equal(df["a"], df["a_dt"]).all()
    assert columns_equal(df["b"], df["b_dt"]).all()
    assert columns_equal(df["a_dt"], df["a"]).all()
    assert columns_equal(df["b_dt"], df["b"]).all()
    assert not columns_equal(df["b_dt"], df["a"]).any()
    assert not columns_equal(df["a_dt"], df["b"]).any()
    assert not columns_equal(df["a"], df["b_dt"]).any()
    assert not columns_equal(df["b"], df["a_dt"]).any()


def test_bad_date_columns():
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    df = pl.DataFrame(
        [{"a": "2017-01-01", "b": "2017-01-01"}, {"a": "2017-01-01", "b": "2A17-01-01"}]
    )
    df = df.with_columns(df["a"].str.to_date(exact=True).alias("a_dt"))
    assert not columns_equal(df["a_dt"], df["b"]).any()


def test_rounded_date_columns():
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    df = pl.DataFrame(
        [
            {"a": "2017-01-01", "b": "2017-01-01 00:00:00.000000", "exp": True},
            {"a": "2017-01-01", "b": "2017-01-01 00:00:00.123456", "exp": False},
            {"a": "2017-01-01", "b": "2017-01-01 00:00:01.000000", "exp": False},
            {"a": "2017-01-01", "b": "2017-01-01 00:00:00", "exp": True},
        ]
    )

    df = df.with_columns(df["a"].str.to_date().alias("a_dt"))
    actual = columns_equal(df["a_dt"], df["b"])
    expected = df["exp"]
    assert_series_equal(actual, expected, check_names=False)


def test_decimal_float_columns_equal():
    df = pl.DataFrame(
        [
            {"a": Decimal("1"), "b": 1, "expected": True},
            {"a": Decimal("1.3"), "b": 1.3, "expected": True},
            {"a": Decimal("1.000003"), "b": 1.000003, "expected": True},
            {"a": Decimal("1.000000004"), "b": 1.000000003, "expected": False},
            {"a": Decimal("1.3"), "b": 1.2, "expected": False},
            {"a": np.nan, "b": np.nan, "expected": True},
            {"a": np.nan, "b": 1, "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    actual_out = columns_equal(df["a"], df["b"])
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_float_columns_equal_rel():
    df = pl.DataFrame(
        [
            {"a": Decimal("1"), "b": 1, "expected": True},
            {"a": Decimal("1.3"), "b": 1.3, "expected": True},
            {"a": Decimal("1.000003"), "b": 1.000003, "expected": True},
            {"a": Decimal("1.000000004"), "b": 1.000000003, "expected": True},
            {"a": Decimal("1.3"), "b": 1.2, "expected": False},
            {"a": np.nan, "b": np.nan, "expected": True},
            {"a": np.nan, "b": 1, "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    actual_out = columns_equal(df["a"], df["b"], abs_tol=0.001)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_columns_equal():
    df = pl.DataFrame(
        [
            {"a": Decimal("1"), "b": Decimal("1"), "expected": True},
            {"a": Decimal("1.3"), "b": Decimal("1.3"), "expected": True},
            {"a": Decimal("1.000003"), "b": Decimal("1.000003"), "expected": True},
            {
                "a": Decimal("1.000000004"),
                "b": Decimal("1.000000003"),
                "expected": False,
            },
            {"a": Decimal("1.3"), "b": Decimal("1.2"), "expected": False},
            {"a": np.nan, "b": np.nan, "expected": True},
            {"a": np.nan, "b": Decimal("1"), "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    actual_out = columns_equal(df["a"], df["b"])
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_columns_equal_rel():
    df = pl.DataFrame(
        [
            {"a": Decimal("1"), "b": Decimal("1"), "expected": True},
            {"a": Decimal("1.3"), "b": Decimal("1.3"), "expected": True},
            {"a": Decimal("1.000003"), "b": Decimal("1.000003"), "expected": True},
            {
                "a": Decimal("1.000000004"),
                "b": Decimal("1.000000003"),
                "expected": True,
            },
            {"a": Decimal("1.3"), "b": Decimal("1.2"), "expected": False},
            {"a": np.nan, "b": np.nan, "expected": True},
            {"a": np.nan, "b": Decimal("1"), "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    actual_out = columns_equal(df["a"], df["b"], abs_tol=0.001)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_infinity_and_beyond():
    df = pl.DataFrame(
        [
            {"a": np.inf, "b": np.inf, "expected": True},
            {"a": -np.inf, "b": -np.inf, "expected": True},
            {"a": -np.inf, "b": np.inf, "expected": False},
            {"a": np.inf, "b": -np.inf, "expected": False},
            {"a": 1, "b": 1, "expected": True},
            {"a": 1, "b": 0, "expected": False},
        ]
    )
    actual_out = columns_equal(df["a"], df["b"])
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_compare_df_setter_bad():
    df = pl.DataFrame([{"a": 1, "c": 2}, {"a": 2, "c": 2}])
    df_same_col_names = pl.DataFrame([{"a": 1, "A": 2}, {"a": 2, "A": 2}])
    df_dupe = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 3}])
    with raises(TypeError, match="df1 must be a Polars DataFrame"):
        compare = PolarsCompare("a", "a", ["a"])
    with raises(ValueError, match="df1 must have all columns from join_columns"):
        compare = PolarsCompare(df, df.clone(), ["b"])
    with raises(DuplicateError, match="duplicate column names found"):
        compare = PolarsCompare(df_same_col_names, df_same_col_names.clone(), ["a"])
    assert (
        PolarsCompare(df_dupe, df_dupe.clone(), ["a", "b"])
        .df1.drop("_merge_left")
        .equals(df_dupe)
    )


def test_compare_df_setter_good():
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = pl.DataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 3}])
    compare = PolarsCompare(df1, df2, ["a"])
    assert compare.df1.drop("_merge_left").equals(df1)
    assert compare.df2.equals(df2)
    assert compare.join_columns == ["a"]
    compare = PolarsCompare(df1, df2, ["A", "b"])
    assert compare.df1.equals(df1)
    assert compare.df2.equals(df2)
    assert compare.join_columns == ["a", "b"]


def test_compare_df_setter_different_cases():
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = pl.DataFrame([{"A": 1, "b": 2}, {"A": 2, "b": 3}])
    compare = PolarsCompare(df1, df2, ["a"])
    assert compare.df1.equals(df1)
    assert compare.df2.equals(df2)


def test_compare_df_setter_bad_index():
    df = pl.DataFrame([{"a": 1, "A": 2}, {"a": 2, "A": 2}])
    with raises(TypeError, match="df1 must be a Polars DataFrame"):
        compare = PolarsCompare("a", "a", join_columns="a")
    with raises(DuplicateError, match="duplicate column names found"):
        compare = PolarsCompare(df, df.clone(), join_columns="a")


def test_compare_df_setter_good_index():
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    compare = PolarsCompare(df1, df2, join_columns="a")
    assert compare.df1.equals(df1)
    assert compare.df2.equals(df2)


def test_columns_overlap():
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    compare = PolarsCompare(df1, df2, ["a"])
    assert compare.df1_unq_columns() == set()
    assert compare.df2_unq_columns() == set()
    assert compare.intersect_columns() == {"a", "b"}


def test_columns_no_overlap():
    df1 = pl.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    df2 = pl.DataFrame([{"a": 1, "b": 2, "d": "oh"}, {"a": 2, "b": 3, "d": "ya"}])
    compare = PolarsCompare(df1, df2, ["a"])
    assert compare.df1_unq_columns() == {"c"}
    assert compare.df2_unq_columns() == {"d"}
    assert compare.intersect_columns() == {"a", "b"}


def test_columns_maintain_order_through_set_operations():
    df1 = pl.DataFrame(
        [
            (("A"), (0), (1), (2), (3), (4), (-2)),
            (("B"), (0), (2), (2), (3), (4), (-3)),
        ],
        schema=["join", "f", "g", "b", "h", "a", "c"],
    )
    df2 = pl.DataFrame(
        [
            (("A"), (0), (1), (2), (-1), (4), (-3)),
            (("B"), (1), (2), (3), (-1), (4), (-2)),
        ],
        schema=["join", "e", "h", "b", "a", "g", "d"],
    )
    compare = PolarsCompare(df1, df2, ["join"])
    assert list(compare.df1_unq_columns()) == ["f", "c"]
    assert list(compare.df2_unq_columns()) == ["e", "d"]
    assert list(compare.intersect_columns()) == ["join", "g", "b", "h", "a"]


def test_10k_rows():
    df1 = pl.DataFrame(np.random.randint(0, 100, size=(10000, 2)), schema=["b", "c"])
    df1 = df1.with_row_index()
    df1.columns = ["a", "b", "c"]
    df2 = df1.clone()
    df2 = df2.with_columns(pl.col("b") + 0.1)
    compare_tol = PolarsCompare(df1, df2, ["a"], abs_tol=0.2)
    assert compare_tol.matches()
    assert len(compare_tol.df1_unq_rows) == 0
    assert len(compare_tol.df2_unq_rows) == 0
    assert compare_tol.intersect_columns() == {"a", "b", "c"}
    assert compare_tol.all_columns_match()
    assert compare_tol.all_rows_overlap()
    assert compare_tol.intersect_rows_match()

    compare_no_tol = PolarsCompare(df1, df2, ["a"])
    assert not compare_no_tol.matches()
    assert len(compare_no_tol.df1_unq_rows) == 0
    assert len(compare_no_tol.df2_unq_rows) == 0
    assert compare_no_tol.intersect_columns() == {"a", "b", "c"}
    assert compare_no_tol.all_columns_match()
    assert compare_no_tol.all_rows_overlap()
    assert not compare_no_tol.intersect_rows_match()


def test_subset(caplog):
    caplog.set_level(logging.DEBUG)
    df1 = pl.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    df2 = pl.DataFrame([{"a": 1, "c": "hi"}])
    comp = PolarsCompare(df1, df2, ["a"])
    assert comp.subset()
    assert "Checking equality" in caplog.text


def test_not_subset(caplog):
    caplog.set_level(logging.INFO)
    df1 = pl.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    df2 = pl.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "great"}])
    comp = PolarsCompare(df1, df2, ["a"])
    assert not comp.subset()
    assert "c: 1 / 2 (50.00%) match" in caplog.text


def test_large_subset():
    df1 = pl.DataFrame(np.random.randint(0, 100, size=(10000, 2)), schema=["b", "c"])
    df1 = df1.with_row_index()
    df1.columns = ["a", "b", "c"]
    df2 = df1[["a", "b"]].sample(50).clone()
    comp = PolarsCompare(df1, df2, ["a"])
    assert not comp.matches()
    assert comp.subset()


def test_string_joiner():
    df1 = pl.DataFrame([{"ab": 1, "bc": 2}, {"ab": 2, "bc": 2}])
    df2 = pl.DataFrame([{"ab": 1, "bc": 2}, {"ab": 2, "bc": 2}])
    compare = PolarsCompare(df1, df2, "ab")
    assert compare.matches()


def test_float_and_string_with_joins():
    df1 = pl.DataFrame([{"a": float("1"), "b": 2}, {"a": float("2"), "b": 2}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    with raises(ComputeError):
        compare = PolarsCompare(df1, df2, "a")


def test_decimal_with_nulls():
    df1 = pl.DataFrame([{"a": 1, "b": Decimal("2")}, {"a": 2, "b": Decimal("2")}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}, {"a": 3, "b": 2}])
    compare = PolarsCompare(df1, df2, "a")
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_joins():
    df1 = pl.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df2 = pl.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    compare = PolarsCompare(df1, df2, "a")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_temp_column_name():
    df1 = pl.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df2 = pl.DataFrame(
        [{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_0"


def test_temp_column_name_one_has():
    df1 = pl.DataFrame([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    df2 = pl.DataFrame(
        [{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_1"


def test_temp_column_name_both_have():
    df1 = pl.DataFrame([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    df2 = pl.DataFrame(
        [
            {"_temp_0": "hi", "b": 2},
            {"_temp_0": "bye", "b": 2},
            {"a": "back fo mo", "b": 3},
        ]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_1"


def test_temp_column_name_both_have():
    df1 = pl.DataFrame([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    df2 = pl.DataFrame(
        [
            {"_temp_0": "hi", "b": 2},
            {"_temp_1": "bye", "b": 2},
            {"a": "back fo mo", "b": 3},
        ]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_2"


def test_temp_column_name_one_already():
    df1 = pl.DataFrame([{"_temp_1": "hi", "b": 2}, {"_temp_1": "bye", "b": 2}])
    df2 = pl.DataFrame(
        [
            {"_temp_1": "hi", "b": 2},
            {"_temp_1": "bye", "b": 2},
            {"a": "back fo mo", "b": 3},
        ]
    )
    actual = temp_column_name(df1, df2)
    assert actual == "_temp_0"


### Duplicate testing!
def test_simple_dupes_one_field():
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    compare = PolarsCompare(df1, df2, join_columns=["a"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_two_fields():
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 2}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 2}])
    compare = PolarsCompare(df1, df2, join_columns=["a", "b"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_one_field_two_vals():
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    compare = PolarsCompare(df1, df2, join_columns=["a"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_one_field_two_vals():
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 0}])
    compare = PolarsCompare(df1, df2, join_columns=["a"])
    assert not compare.matches()
    assert len(compare.df1_unq_rows) == 1
    assert len(compare.df2_unq_rows) == 1
    assert len(compare.intersect_rows) == 1
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_one_field_three_to_two_vals():
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}, {"a": 1, "b": 0}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    compare = PolarsCompare(df1, df2, join_columns=["a"])
    assert not compare.matches()
    assert len(compare.df1_unq_rows) == 1
    assert len(compare.df2_unq_rows) == 0
    assert len(compare.intersect_rows) == 2
    # Just render the report to make sure it renders.
    t = compare.report()

    assert "(First 1 Columns)" in compare.report(column_count=1)
    assert "(First 2 Columns)" in compare.report(column_count=2)


def test_dupes_from_real_data():
    data = """acct_id,acct_sfx_num,trxn_post_dt,trxn_post_seq_num,trxn_amt,trxn_dt,debit_cr_cd,cash_adv_trxn_comn_cntry_cd,mrch_catg_cd,mrch_pstl_cd,visa_mail_phn_cd,visa_rqstd_pmt_svc_cd,mc_pmt_facilitator_idn_num
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
    df1 = pl.read_csv(io.StringIO(data), separator=",")
    df2 = df1.clone()
    compare_acct = PolarsCompare(df1, df2, join_columns=["acct_id"])
    assert compare_acct.matches()
    compare_unq = PolarsCompare(
        df1,
        df2,
        join_columns=["acct_id", "acct_sfx_num", "trxn_post_dt", "trxn_post_seq_num"],
    )
    assert compare_unq.matches()
    # Just render the report to make sure it renders.
    t = compare_acct.report()
    r = compare_unq.report()


def test_strings_with_joins_with_ignore_spaces():
    df1 = pl.DataFrame([{"a": "hi", "b": " A"}, {"a": "bye", "b": "A"}])
    df2 = pl.DataFrame([{"a": "hi", "b": "A"}, {"a": "bye", "b": "A "}])
    compare = PolarsCompare(df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = PolarsCompare(df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_joins_with_ignore_case():
    df1 = pl.DataFrame([{"a": "hi", "b": "a"}, {"a": "bye", "b": "A"}])
    df2 = pl.DataFrame([{"a": "hi", "b": "A"}, {"a": "bye", "b": "a"}])
    compare = PolarsCompare(df1, df2, "a", ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = PolarsCompare(df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_joins_with_ignore_spaces():
    df1 = pl.DataFrame([{"a": 1, "b": " A"}, {"a": 2, "b": "A"}])
    df2 = pl.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A "}])
    compare = PolarsCompare(df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = PolarsCompare(df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_joins_with_ignore_case():
    df1 = pl.DataFrame([{"a": 1, "b": "a"}, {"a": 2, "b": "A"}])
    df2 = pl.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "a"}])
    compare = PolarsCompare(df1, df2, "a", ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = PolarsCompare(df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_joins_with_ignore_spaces():
    df1 = pl.DataFrame([{"a": 1, "b": " A"}, {"a": 2, "b": "A"}])
    df2 = pl.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A "}])
    compare = PolarsCompare(df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_joins_with_ignore_case():
    df1 = pl.DataFrame([{"a": 1, "b": "a"}, {"a": 2, "b": "A"}])
    df2 = pl.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "a"}])
    compare = PolarsCompare(df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_ignore_spaces_and_join_columns():
    df1 = pl.DataFrame([{"a": "hi", "b": "A"}, {"a": "bye", "b": "A"}])
    df2 = pl.DataFrame([{"a": " hi ", "b": "A"}, {"a": " bye ", "b": "A"}])
    compare = PolarsCompare(df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.count_matching_rows() == 0

    compare = PolarsCompare(df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2


def test_integers_with_ignore_spaces_and_join_columns():
    df1 = pl.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A"}])
    df2 = pl.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A"}])
    compare = PolarsCompare(df1, df2, "a", ignore_spaces=False)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2

    compare = PolarsCompare(df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2


def test_sample_mismatch():
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.4,George Michael Bluth,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Robert Loblaw,345.12,
    10000001238,1.05,Loose Seal Bluth,111,
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """
    df1 = pl.read_csv(io.StringIO(data1), separator=",")
    df2 = pl.read_csv(io.StringIO(data2), separator=",")
    compare = PolarsCompare(df1, df2, "acct_id")

    output = compare.sample_mismatch(column="name", sample_count=1)
    assert output.shape[0] == 1
    assert (output["name_df1"] != output["name_df2"]).all()

    output = compare.sample_mismatch(column="name", sample_count=2)
    assert output.shape[0] == 2
    assert (output["name_df1"] != output["name_df2"]).all()

    output = compare.sample_mismatch(column="name", sample_count=3)
    assert output.shape[0] == 2
    assert (["name_df1"] != output["name_df2"]).all()


def test_all_mismatch_not_ignore_matching_cols_no_cols_matching():
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.4,George Michael Bluth,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Robert Loblaw,345.12,
    10000001238,1.05,Loose Seal Bluth,111,
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """
    df1 = pl.read_csv(io.StringIO(data1), separator=",")
    df2 = pl.read_csv(io.StringIO(data2), separator=",")
    compare = PolarsCompare(df1, df2, "acct_id")

    output = compare.all_mismatch()
    assert output.shape[0] == 4
    assert output.shape[1] == 9

    assert (output["name_df1"] != output["name_df2"]).sum() == 2
    assert (~(output["name_df1"] != output["name_df2"])).sum() == 2

    assert (output["dollar_amt_df1"] != output["dollar_amt_df2"]).sum() == 1
    assert (~(output["dollar_amt_df1"] != output["dollar_amt_df2"])).sum() == 3

    # need to use eq_missing
    assert (
        output.filter(pl.col.float_fld_df1.eq_missing(pl.col.float_fld_df2).not_())[
            "acct_id"
        ].count()
        == 3
    )
    assert (
        output.filter(pl.col.float_fld_df1.eq_missing(pl.col.float_fld_df2))[
            "acct_id"
        ].count()
        == 1
    )

    assert (
        output.filter(pl.col.date_fld_df1.eq_missing(pl.col.date_fld_df2).not_())[
            "acct_id"
        ].count()
        == 4
    )
    assert (
        output.filter(pl.col.date_fld_df1.eq_missing(pl.col.date_fld_df2))[
            "acct_id"
        ].count()
        == 0
    )


def test_all_mismatch_not_ignore_matching_cols_some_cols_matching():
    # Columns dollar_amt and name are matching
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
        10000001234,123.45,George Maharis,14530.1555,2017-01-01
        10000001235,0.45,Michael Bluth,1,2017-01-01
        10000001236,1345,George Bluth,,2017-01-01
        10000001237,123456,Bob Loblaw,345.12,2017-01-01
        10000001239,1.05,Lucille Bluth,,2017-01-01
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
        10000001234,123.45,George Maharis,14530.155,
        10000001235,0.45,Michael Bluth,,
        10000001236,1345,George Bluth,1,
        10000001237,123456,Bob Loblaw,345.12,
        10000001238,1.05,Lucille Bluth,111,
        10000001240,123.45,George Maharis,14530.1555,2017-01-02
        """
    df1 = pl.read_csv(io.StringIO(data1), separator=",")
    df2 = pl.read_csv(io.StringIO(data2), separator=",")
    compare = PolarsCompare(df1, df2, "acct_id")

    output = compare.all_mismatch()
    assert output.shape[0] == 4
    assert output.shape[1] == 9

    assert (output["name_df1"] != output["name_df2"]).sum() == 0
    assert (~(output["name_df1"] != output["name_df2"])).sum() == 4

    assert (output["dollar_amt_df1"] != output["dollar_amt_df2"]).sum() == 0
    assert (~(output["dollar_amt_df1"] != output["dollar_amt_df2"])).sum() == 4

    # need to use eq_missing
    assert (
        output.filter(pl.col.float_fld_df1.eq_missing(pl.col.float_fld_df2).not_())[
            "acct_id"
        ].count()
        == 3
    )
    assert (
        output.filter(pl.col.float_fld_df1.eq_missing(pl.col.float_fld_df2))[
            "acct_id"
        ].count()
        == 1
    )

    assert (
        output.filter(pl.col.date_fld_df1.eq_missing(pl.col.date_fld_df2).not_())[
            "acct_id"
        ].count()
        == 4
    )
    assert (
        output.filter(pl.col.date_fld_df1.eq_missing(pl.col.date_fld_df2))[
            "acct_id"
        ].count()
        == 0
    )


def test_all_mismatch_ignore_matching_cols_some_cols_matching_diff_rows():
    # Case where there are rows on either dataset which don't match up.
    # Columns dollar_amt and name are matching
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    10000001241,1111.05,Lucille Bluth,
    """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Bob Loblaw,345.12,
    10000001238,1.05,Lucille Bluth,111,
    """
    df1 = pl.read_csv(io.StringIO(data1), separator=",")
    df2 = pl.read_csv(io.StringIO(data2), separator=",")
    compare = PolarsCompare(df1, df2, "acct_id")

    output = compare.all_mismatch(ignore_matching_cols=True)

    assert output.shape[0] == 4
    assert output.shape[1] == 5

    # need to use eq_missing
    assert (
        output.filter(pl.col.float_fld_df1.eq_missing(pl.col.float_fld_df2).not_())[
            "acct_id"
        ].count()
        == 3
    )
    assert (
        output.filter(pl.col.float_fld_df1.eq_missing(pl.col.float_fld_df2))[
            "acct_id"
        ].count()
        == 1
    )

    assert (
        output.filter(pl.col.date_fld_df1.eq_missing(pl.col.date_fld_df2).not_())[
            "acct_id"
        ].count()
        == 4
    )
    assert (
        output.filter(pl.col.date_fld_df1.eq_missing(pl.col.date_fld_df2))[
            "acct_id"
        ].count()
        == 0
    )

    assert not ("name_df1" in output and "name_df2" in output)
    assert not ("dollar_amt_df1" in output and "dollar_amt_df1" in output)


def test_all_mismatch_ignore_matching_cols_some_cols_matching():
    # Columns dollar_amt and name are matching
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Bob Loblaw,345.12,
    10000001238,1.05,Lucille Bluth,111,
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """
    df1 = pl.read_csv(io.StringIO(data1), separator=",")
    df2 = pl.read_csv(io.StringIO(data2), separator=",")
    compare = PolarsCompare(df1, df2, "acct_id")

    output = compare.all_mismatch(ignore_matching_cols=True)

    assert output.shape[0] == 4
    assert output.shape[1] == 5

    # need to use eq_missing
    assert (
        output.filter(pl.col.float_fld_df1.eq_missing(pl.col.float_fld_df2).not_())[
            "acct_id"
        ].count()
        == 3
    )
    assert (
        output.filter(pl.col.float_fld_df1.eq_missing(pl.col.float_fld_df2))[
            "acct_id"
        ].count()
        == 1
    )

    assert (
        output.filter(pl.col.date_fld_df1.eq_missing(pl.col.date_fld_df2).not_())[
            "acct_id"
        ].count()
        == 4
    )
    assert (
        output.filter(pl.col.date_fld_df1.eq_missing(pl.col.date_fld_df2))[
            "acct_id"
        ].count()
        == 0
    )

    assert not ("name_df1" in output and "name_df2" in output)
    assert not ("dollar_amt_df1" in output and "dollar_amt_df1" in output)


def test_all_mismatch_ignore_matching_cols_no_cols_matching():
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.4,George Michael Bluth,14530.155,
    10000001235,0.45,Michael Bluth,,
    10000001236,1345,George Bluth,1,
    10000001237,123456,Robert Loblaw,345.12,
    10000001238,1.05,Loose Seal Bluth,111,
    10000001240,123.45,George Maharis,14530.1555,2017-01-02
    """
    df1 = pl.read_csv(io.StringIO(data1), separator=",")
    df2 = pl.read_csv(io.StringIO(data2), separator=",")
    compare = PolarsCompare(df1, df2, "acct_id")

    output = compare.all_mismatch()
    assert output.shape[0] == 4
    assert output.shape[1] == 9

    assert (output["name_df1"] != output["name_df2"]).sum() == 2
    assert (~(output["name_df1"] != output["name_df2"])).sum() == 2

    assert (output["dollar_amt_df1"] != output["dollar_amt_df2"]).sum() == 1
    assert (~(output["dollar_amt_df1"] != output["dollar_amt_df2"])).sum() == 3

    # need to use eq_missing
    assert (
        output.filter(pl.col.float_fld_df1.eq_missing(pl.col.float_fld_df2).not_())[
            "acct_id"
        ].count()
        == 3
    )
    assert (
        output.filter(pl.col.float_fld_df1.eq_missing(pl.col.float_fld_df2))[
            "acct_id"
        ].count()
        == 1
    )

    assert (
        output.filter(pl.col.date_fld_df1.eq_missing(pl.col.date_fld_df2).not_())[
            "acct_id"
        ].count()
        == 4
    )
    assert (
        output.filter(pl.col.date_fld_df1.eq_missing(pl.col.date_fld_df2))[
            "acct_id"
        ].count()
        == 0
    )


MAX_DIFF_DF = pl.DataFrame(
    {
        "base": [1, 1, 1, 1, 1],
        "floats": [1.1, 1.1, 1.1, 1.2, 0.9],
        "decimals": [
            Decimal("1.1"),
            Decimal("1.1"),
            Decimal("1.1"),
            Decimal("1.1"),
            Decimal("1.1"),
        ],
        "null_floats": [np.nan, 1.1, 1, 1, 1],
        "strings": ["1", "1", "1", "1.1", "1"],
        "mixed_strings": ["1", "1", "1", "2", "some string"],
        "infinity": [1, 1, 1, 1, np.inf],
    }
)


@pytest.mark.parametrize(
    "column,expected",
    [
        ("base", 0),
        ("floats", 0.2),
        ("decimals", 0.1),
        ("null_floats", 0.1),
        ("strings", 0.1),
        ("mixed_strings", 0),
        ("infinity", np.inf),
    ],
)
def test_calculate_max_diff(column, expected):
    assert np.isclose(
        calculate_max_diff(MAX_DIFF_DF["base"], MAX_DIFF_DF[column]), expected
    )


def test_dupes_with_nulls():
    df1 = pl.DataFrame(
        {
            "fld_1": [1, 2, 2, 3, 3, 4, 5, 5],
            "fld_2": ["A", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    df2 = pl.DataFrame(
        {"fld_1": [1, 2, 3, 4, 5], "fld_2": ["A", np.nan, np.nan, np.nan, np.nan]}
    )
    comp = PolarsCompare(df1, df2, join_columns=["fld_1", "fld_2"])
    assert comp.subset()


@pytest.mark.parametrize(
    "dataframe,expected",
    [
        (pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), pl.Series([1, 1, 1])),
        (
            pl.DataFrame({"a": ["a", "a", "DATACOMPY_NULL"], "b": [1, 1, 2]}),
            pl.Series([1, 2, 1]),
        ),
        (pl.DataFrame({"a": [-999, 2, 3], "b": [1, 2, 3]}), pl.Series([1, 1, 1])),
        (
            pl.DataFrame({"a": [1, np.nan, np.nan], "b": [1, 2, 2]}),
            pl.Series([1, 1, 2]),
        ),
        (
            pl.DataFrame({"a": ["1", np.nan, np.nan], "b": ["1", "2", "2"]}),
            pl.Series([1, 1, 2]),
        ),
        (
            pl.DataFrame(
                {"a": [datetime(2018, 1, 1), np.nan, np.nan], "b": ["1", "2", "2"]}
            ),
            pl.Series([1, 1, 2]),
        ),
    ],
)
def test_generate_id_within_group(dataframe, expected):
    assert (generate_id_within_group(dataframe, ["a", "b"]) == expected).all()


@pytest.mark.parametrize(
    "dataframe, message",
    [
        (
            pl.DataFrame({"a": [1, np.nan, "DATACOMPY_NULL"], "b": [1, 2, 3]}),
            "DATACOMPY_NULL was found in your join columns",
        )
    ],
)
def test_generate_id_within_group_valueerror(dataframe, message):
    with raises(ValueError, match=message):
        generate_id_within_group(dataframe, ["a", "b"])


def test_lower():
    """This function tests the toggle to use lower case for column names or not"""
    # should match
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    df2 = pl.DataFrame({"a": [1, 2, 3], "B": [0, 1, 2]})
    compare = PolarsCompare(df1, df2, join_columns=["a"])
    assert compare.matches()
    # should not match
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    df2 = pl.DataFrame({"a": [1, 2, 3], "B": [0, 1, 2]})
    compare = PolarsCompare(df1, df2, join_columns=["a"], cast_column_names_lower=False)
    assert not compare.matches()

    # test join column
    # should match
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    df2 = pl.DataFrame({"A": [1, 2, 3], "B": [0, 1, 2]})
    compare = PolarsCompare(df1, df2, join_columns=["a"])
    assert compare.matches()
    # should fail because "a" is not found in df2
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    df2 = pl.DataFrame({"A": [1, 2, 3], "B": [0, 1, 2]})
    expected_message = "df2 must have all columns from join_columns"
    with raises(ValueError, match=expected_message):
        compare = PolarsCompare(
            df1, df2, join_columns=["a"], cast_column_names_lower=False
        )


@mock.patch("datacompy.polars.render")
def test_save_html(mock_render):
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    compare = PolarsCompare(df1, df2, join_columns=["a"])

    m = mock.mock_open()
    with mock.patch("datacompy.polars.open", m, create=True):
        # assert without HTML call
        compare.report()
        assert mock_render.call_count == 4
        m.assert_not_called()

    mock_render.reset_mock()
    m = mock.mock_open()
    with mock.patch("datacompy.polars.open", m, create=True):
        # assert with HTML call
        compare.report(html_file="test.html")
        assert mock_render.call_count == 4
        m.assert_called_with("test.html", "w")
