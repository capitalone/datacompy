# -*- coding: utf-8 -*-
#
# Copyright 2017 Capital One Services, LLC
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
Testing out datacompy utils
"""

import logging
import sys
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import six
from pandas.util.testing import assert_series_equal
from pytest import raises

from datacompy import utils


def test_numeric_columns_equal_abs():
    data = """a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""
    df = pd.read_csv(six.StringIO(data), sep="|")
    actual_out = utils.columns_equal(df.a, df.b, abs_tol=0.2)
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
    df = pd.read_csv(six.StringIO(data), sep="|")
    actual_out = utils.columns_equal(df.a, df.b, rel_tol=0.2)
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
 | |True
  | |False
datacompy|DataComPy|False
something||False
|something|False
||True"""
    df = pd.read_csv(six.StringIO(data), sep="|")
    actual_out = utils.columns_equal(df.a, df.b, rel_tol=0.2)
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
 | |True
  |       |True
datacompy|DataComPy|False
something||False
|something|False
||True"""
    df = pd.read_csv(six.StringIO(data), sep="|")
    actual_out = utils.columns_equal(df.a, df.b, rel_tol=0.2, ignore_spaces=True)
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
 | |True
  |       |True
datacompy|DataComPy|True
something||False
|something|False
||True"""
    df = pd.read_csv(six.StringIO(data), sep="|")
    actual_out = utils.columns_equal(df.a, df.b, rel_tol=0.2, ignore_spaces=True, ignore_case=True)
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
    df = pd.read_csv(six.StringIO(data), sep="|")
    # First compare just the strings
    actual_out = utils.columns_equal(df.a, df.b, rel_tol=0.2)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    df["a"] = pd.to_datetime(df["a"])
    df["b"] = pd.to_datetime(df["b"])
    actual_out = utils.columns_equal(df.a, df.b, rel_tol=0.2)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)
    # and reverse
    actual_out_rev = utils.columns_equal(df.b, df.a, rel_tol=0.2)
    assert_series_equal(expect_out, actual_out_rev, check_names=False)


def test_date_columns_equal_with_ignore_spaces():
    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    df = pd.read_csv(six.StringIO(data), sep="|")
    # First compare just the strings
    actual_out = utils.columns_equal(df.a, df.b, rel_tol=0.2, ignore_spaces=True)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    df["a"] = pd.to_datetime(df["a"])
    df["b"] = pd.to_datetime(df["b"])
    actual_out = utils.columns_equal(df.a, df.b, rel_tol=0.2, ignore_spaces=True)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)
    # and reverse
    actual_out_rev = utils.columns_equal(df.b, df.a, rel_tol=0.2, ignore_spaces=True)
    assert_series_equal(expect_out, actual_out_rev, check_names=False)


def test_date_columns_equal_with_ignore_spaces_and_case():
    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    df = pd.read_csv(six.StringIO(data), sep="|")
    # First compare just the strings
    actual_out = utils.columns_equal(df.a, df.b, rel_tol=0.2, ignore_spaces=True, ignore_case=True)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    df["a"] = pd.to_datetime(df["a"])
    df["b"] = pd.to_datetime(df["b"])
    actual_out = utils.columns_equal(df.a, df.b, rel_tol=0.2, ignore_spaces=True)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)
    # and reverse
    actual_out_rev = utils.columns_equal(df.b, df.a, rel_tol=0.2, ignore_spaces=True)
    assert_series_equal(expect_out, actual_out_rev, check_names=False)


def test_date_columns_unequal():
    """I want datetime fields to match with dates stored as strings
    """
    df = pd.DataFrame([{"a": "2017-01-01", "b": "2017-01-02"}, {"a": "2017-01-01"}])
    df["a_dt"] = pd.to_datetime(df["a"])
    df["b_dt"] = pd.to_datetime(df["b"])
    assert utils.columns_equal(df.a, df.a_dt).all()
    assert utils.columns_equal(df.b, df.b_dt).all()
    assert utils.columns_equal(df.a_dt, df.a).all()
    assert utils.columns_equal(df.b_dt, df.b).all()
    assert not utils.columns_equal(df.b_dt, df.a).any()
    assert not utils.columns_equal(df.a_dt, df.b).any()
    assert not utils.columns_equal(df.a, df.b_dt).any()
    assert not utils.columns_equal(df.b, df.a_dt).any()


def test_bad_date_columns():
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    df = pd.DataFrame(
        [{"a": "2017-01-01", "b": "2017-01-01"}, {"a": "2017-01-01", "b": "217-01-01"}]
    )
    df["a_dt"] = pd.to_datetime(df["a"])
    assert not utils.columns_equal(df.a_dt, df.b).any()


def test_rounded_date_columns():
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    df = pd.DataFrame(
        [
            {"a": "2017-01-01", "b": "2017-01-01 00:00:00.000000", "exp": True},
            {"a": "2017-01-01", "b": "2017-01-01 00:00:00.123456", "exp": False},
            {"a": "2017-01-01", "b": "2017-01-01 00:00:01.000000", "exp": False},
            {"a": "2017-01-01", "b": "2017-01-01 00:00:00", "exp": True},
        ]
    )
    df["a_dt"] = pd.to_datetime(df["a"])
    actual = utils.columns_equal(df.a_dt, df.b)
    expected = df["exp"]
    assert_series_equal(actual, expected, check_names=False)


def test_decimal_float_columns_equal():
    df = pd.DataFrame(
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
    actual_out = utils.columns_equal(df.a, df.b)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_float_columns_equal_rel():
    df = pd.DataFrame(
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
    actual_out = utils.columns_equal(df.a, df.b, abs_tol=0.001)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_columns_equal():
    df = pd.DataFrame(
        [
            {"a": Decimal("1"), "b": Decimal("1"), "expected": True},
            {"a": Decimal("1.3"), "b": Decimal("1.3"), "expected": True},
            {"a": Decimal("1.000003"), "b": Decimal("1.000003"), "expected": True},
            {"a": Decimal("1.000000004"), "b": Decimal("1.000000003"), "expected": False},
            {"a": Decimal("1.3"), "b": Decimal("1.2"), "expected": False},
            {"a": np.nan, "b": np.nan, "expected": True},
            {"a": np.nan, "b": Decimal("1"), "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    actual_out = utils.columns_equal(df.a, df.b)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_columns_equal_rel():
    df = pd.DataFrame(
        [
            {"a": Decimal("1"), "b": Decimal("1"), "expected": True},
            {"a": Decimal("1.3"), "b": Decimal("1.3"), "expected": True},
            {"a": Decimal("1.000003"), "b": Decimal("1.000003"), "expected": True},
            {"a": Decimal("1.000000004"), "b": Decimal("1.000000003"), "expected": True},
            {"a": Decimal("1.3"), "b": Decimal("1.2"), "expected": False},
            {"a": np.nan, "b": np.nan, "expected": True},
            {"a": np.nan, "b": Decimal("1"), "expected": False},
            {"a": Decimal("1"), "b": np.nan, "expected": False},
        ]
    )
    actual_out = utils.columns_equal(df.a, df.b, abs_tol=0.001)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_infinity_and_beyond():
    df = pd.DataFrame(
        [
            {"a": np.inf, "b": np.inf, "expected": True},
            {"a": -np.inf, "b": -np.inf, "expected": True},
            {"a": -np.inf, "b": np.inf, "expected": False},
            {"a": np.inf, "b": -np.inf, "expected": False},
            {"a": 1, "b": 1, "expected": True},
            {"a": 1, "b": 0, "expected": False},
        ]
    )
    actual_out = utils.columns_equal(df.a, df.b)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_mixed_column():
    df = pd.DataFrame(
        [
            {"a": "hi", "b": "hi", "expected": True},
            {"a": 1, "b": 1, "expected": True},
            {"a": np.inf, "b": np.inf, "expected": True},
            {"a": Decimal("1"), "b": Decimal("1"), "expected": True},
            {"a": 1, "b": "1", "expected": False},
            {"a": 1, "b": "yo", "expected": False},
        ]
    )
    actual_out = utils.columns_equal(df.a, df.b)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_mixed_column_with_ignore_spaces():
    df = pd.DataFrame(
        [
            {"a": "hi", "b": "hi ", "expected": True},
            {"a": 1, "b": 1, "expected": True},
            {"a": np.inf, "b": np.inf, "expected": True},
            {"a": Decimal("1"), "b": Decimal("1"), "expected": True},
            {"a": 1, "b": "1 ", "expected": False},
            {"a": 1, "b": "yo ", "expected": False},
        ]
    )
    actual_out = utils.columns_equal(df.a, df.b, ignore_spaces=True)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_mixed_column_with_ignore_spaces_and_case():
    df = pd.DataFrame(
        [
            {"a": "hi", "b": "hi ", "expected": True},
            {"a": 1, "b": 1, "expected": True},
            {"a": np.inf, "b": np.inf, "expected": True},
            {"a": Decimal("1"), "b": Decimal("1"), "expected": True},
            {"a": 1, "b": "1 ", "expected": False},
            {"a": 1, "b": "yo ", "expected": False},
            {"a": "Hi", "b": "hI ", "expected": True},
            {"a": "HI", "b": "HI ", "expected": True},
            {"a": "hi", "b": "hi ", "expected": True},
        ]
    )
    actual_out = utils.columns_equal(df.a, df.b, ignore_spaces=True, ignore_case=True)
    expect_out = df["expected"]
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_temp_column_name():
    df1 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df2 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}])
    actual = utils.temp_column_name(df1, df2)
    assert actual == "_temp_0"


def test_temp_column_name_one_has():
    df1 = pd.DataFrame([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    df2 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}])
    actual = utils.temp_column_name(df1, df2)
    assert actual == "_temp_1"


def test_temp_column_name_both_have():
    df1 = pd.DataFrame([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    df2 = pd.DataFrame(
        [{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    actual = utils.temp_column_name(df1, df2)
    assert actual == "_temp_1"


def test_temp_column_name_both_have():
    df1 = pd.DataFrame([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    df2 = pd.DataFrame(
        [{"_temp_0": "hi", "b": 2}, {"_temp_1": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    actual = utils.temp_column_name(df1, df2)
    assert actual == "_temp_2"


def test_temp_column_name_one_already():
    df1 = pd.DataFrame([{"_temp_1": "hi", "b": 2}, {"_temp_1": "bye", "b": 2}])
    df2 = pd.DataFrame(
        [{"_temp_1": "hi", "b": 2}, {"_temp_1": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    actual = utils.temp_column_name(df1, df2)
    assert actual == "_temp_0"


MAX_DIFF_DF = pd.DataFrame(
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
    assert np.isclose(utils.calculate_max_diff(MAX_DIFF_DF["base"], MAX_DIFF_DF[column]), expected)


@pytest.mark.parametrize(
    "dataframe,expected",
    [
        (pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), pd.Series([0, 0, 0])),
        (pd.DataFrame({"a": ["a", "a", "DATACOMPY_NULL"], "b": [1, 1, 2]}), pd.Series([0, 1, 0])),
        (pd.DataFrame({"a": [-999, 2, 3], "b": [1, 2, 3]}), pd.Series([0, 0, 0])),
        (pd.DataFrame({"a": [1, np.nan, np.nan], "b": [1, 2, 2]}), pd.Series([0, 0, 1])),
        (pd.DataFrame({"a": ["1", np.nan, np.nan], "b": ["1", "2", "2"]}), pd.Series([0, 0, 1])),
        (
            pd.DataFrame({"a": [datetime(2018, 1, 1), np.nan, np.nan], "b": ["1", "2", "2"]}),
            pd.Series([0, 0, 1]),
        ),
    ],
)
def test_generate_id_within_group(dataframe, expected):
    assert (utils.generate_id_within_group(dataframe, ["a", "b"]) == expected).all()


@pytest.mark.parametrize(
    "dataframe, message",
    [
        (
            pd.DataFrame({"a": [1, np.nan, "DATACOMPY_NULL"], "b": [1, 2, 3]}),
            "DATACOMPY_NULL was found in your join columns",
        )
    ],
)
def test_generate_id_within_group_valueerror(dataframe, message):
    with raises(ValueError, message=message):
        utils.generate_id_within_group(dataframe, ["a", "b"])
