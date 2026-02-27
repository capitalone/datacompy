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

"""
Testing out the datacompy functionality
"""

import io
import logging
import os
import sys
import tempfile
from datetime import datetime
from decimal import Decimal
from unittest import mock

import numpy as np
import pyarrow as pa
import polars as pl
import pytest
from datacompy.comparator.base import BaseComparator
from datacompy.comparator.string import pyarrow_normalize_string_column
from datacompy.pyarrow import (
    PyArrowCompare,
    calculate_max_diff,
    columns_equal,
    generate_id_within_group,
    temp_column_name,
)

from pyarrow import (
    ArrowNotImplementedError, 
    ArrowKeyError, 
    ArrowInvalid,
)
import pyarrow.compute as pc
from polars.testing import assert_frame_equal, assert_series_equal
from pytest import raises

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def test_numeric_columns_equal_abs():
    data = """a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""

    table = pa._csv.read_csv(
        io.BytesIO(data.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter="|"), 
        convert_options=pa._csv.ConvertOptions(null_values=["NULL"])
    )
    
    actual_out = pl.Series(columns_equal(table["a"], table["b"], abs_tol=0.2))
    expect_out = pl.Series(table["expected"])

    assert_series_equal(expect_out, actual_out, check_names=False)


def test_numeric_columns_equal_rel():
    data = """a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True"""
    table = pa._csv.read_csv(
        io.BytesIO(data.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter="|"), 
        convert_options=pa._csv.ConvertOptions(null_values=["NULL"])
    )
    actual_out = pl.Series(columns_equal(table["a"], table["b"], rel_tol=0.2))
    expect_out = pl.Series(table["expected"])
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
    table = pa._csv.read_csv(
        io.BytesIO(data.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter="|"), 
        convert_options=pa._csv.ConvertOptions(
            null_values=["NULL"],
            strings_can_be_null=True,
        )
    )
    actual_out = pl.Series(columns_equal(table["a"], table["b"], rel_tol=0.2))
    expect_out = pl.Series(table["expected"])
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
    table = pa._csv.read_csv(
        io.BytesIO(data.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter="|"), 
        convert_options=pa._csv.ConvertOptions(
            null_values=["NULL"],
            strings_can_be_null=True,
        )
    )
    actual_out = columns_equal(table["a"], table["b"], rel_tol=0.2, ignore_spaces=True)
    expect_out = table["expected"]
    assert expect_out.equals(actual_out)


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
    table = pa._csv.read_csv(
        io.BytesIO(data.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter="|"), 
        convert_options=pa._csv.ConvertOptions(
            null_values=["NULL"],
            strings_can_be_null=True,
        )
    )
    actual_out = pl.Series(
        columns_equal(
            table["a"], table["b"], rel_tol=0.2, ignore_spaces=True, ignore_case=True
        )
    )
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_date_columns_equal():
    data = """a|b|expected
2017-01-01|2017-01-01|True
2017-01-02|2017-01-02|True
2017-10-01|2017-10-10|False
2017-01-01||False
|2017-01-01|False
||True"""
    table = pa._csv.read_csv(
        io.BytesIO(data.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter="|"), 
        convert_options=pa._csv.ConvertOptions(
            null_values=["NULL"],
            strings_can_be_null=True,
        )
    )
    # First compare just the strings
    actual_out = pl.Series(columns_equal(table["a"], table["b"], rel_tol=0.2))
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    col_a = pc.strptime(table["a"], format="%Y-%m-%d", unit="ns", error_is_null=True)
    col_b = pc.strptime(table["b"], format="%Y-%m-%d", unit="ns", error_is_null=True)
    actual_out = pl.Series(columns_equal(col_a, col_b, rel_tol=0.2))
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_date_columns_equal_with_ignore_spaces():
    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    table = pa._csv.read_csv(
        io.BytesIO(data.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter="|"), 
        convert_options=pa._csv.ConvertOptions(
            null_values=["NULL"],
            strings_can_be_null=True,
        )
    )
    # First compare just the strings
    actual_out = pl.Series(columns_equal(table["a"], table["b"], rel_tol=0.2, ignore_spaces=True))
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    col_a = pc.strptime(
        pc.utf8_trim(table["a"], characters=" "), 
        format="%Y-%m-%d", unit="ns", 
        error_is_null=True)
    col_b = pc.strptime(
        pc.utf8_trim(table["b"], characters=" "), 
        format="%Y-%m-%d", 
        unit="ns", 
        error_is_null=True
    )

    actual_out = pl.Series(columns_equal(col_a, col_b, rel_tol=0.2, ignore_spaces=True))
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_date_columns_equal_with_ignore_spaces_and_case():
    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    table = pa._csv.read_csv(
        io.BytesIO(data.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter="|"), 
        convert_options=pa._csv.ConvertOptions(
            null_values=["NULL"],
            strings_can_be_null=True,
        )
    )
    # First compare just the strings
    actual_out = pl.Series(
        columns_equal(table["a"], table["b"], rel_tol=0.2, ignore_spaces=True, ignore_case=True)
    )
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)

    # Then compare converted to datetime objects
    col_a = pc.strptime(
        pc.utf8_trim(table["a"], characters=" "), 
        format="%Y-%m-%d", unit="ns", 
        error_is_null=True)
    col_b = pc.strptime(
        pc.utf8_trim(table["b"], characters=" "), 
        format="%Y-%m-%d", 
        unit="ns", 
        error_is_null=True
    )

    actual_out = pl.Series(columns_equal(col_a, col_b, rel_tol=0.2, ignore_spaces=True))
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_date_columns_unequal():
    """I want datetime fields to match with dates stored as strings"""
    dt_cols = {
        "a": pa.chunked_array([["2017-01-01", "2017-01-03"]]),
        "b": pa.chunked_array([["2017-01-02", None]], type=pa.string()),
        "a_dt": pc.strptime(
            pa.chunked_array([["2017-01-01", "2017-01-03"]]), 
            format="%Y-%m-%d", unit="ns", 
            error_is_null=True
        ),
        "b_dt": pc.strptime(
            pa.chunked_array([["2017-01-02", None]], type=pa.string()), 
            format="%Y-%m-%d", unit="ns", 
            error_is_null=True
        ),
    }
    table = pa.Table.from_pydict(dt_cols)
    assert pc.all(columns_equal(table["a"], table["a_dt"])).as_py()
    assert pc.all(columns_equal(table["b"], table["b_dt"])).as_py()
    assert pc.all(columns_equal(table["a_dt"], table["a"])).as_py()
    assert pc.all(columns_equal(table["b_dt"], table["b"])).as_py()
    assert not pc.any(columns_equal(table["b_dt"], table["a"])).as_py()
    assert not pc.any(columns_equal(table["a_dt"], table["b"])).as_py()
    assert not pc.any(columns_equal(table["a"], table["b_dt"])).as_py()
    assert not pc.any(columns_equal(table["b"], table["a_dt"])).as_py()


def test_bad_date_columns():
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    table = pa.Table.from_pydict(
        {"a": ["2017-01-01", "2017-01-01"], "b": ["2017-01-01", "2A17-01-01"]}
    )
    col_a = pc.strptime(table["a"], format="%Y-%m-%d", unit="ns", error_is_null=True)
    col_b = table["b"]
    assert columns_equal(col_a, col_b).to_pylist() == [True, False]

    col_a = table["a"]
    col_b = pc.strptime(table["b"], format="%Y-%m-%d", unit="ns", error_is_null=True)
    assert columns_equal(col_a, col_b).to_pylist() == [True, False]


def test_rounded_date_columns():
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    cola = pa.chunked_array([["2017-01-01", "2017-01-01", "2017-01-01", "2017-01-01"]])
    colb = pa.chunked_array([["2017-01-01 00:00:00.000000", "2017-01-01 00:00:00.123456",
                                "2017-01-01 00:00:01.000000", "2017-01-01 00:00:00"]])
    dt_cols = {
        "a": cola,
        "b": colb,
        "exp": pa.chunked_array([[True, False, False, True]]),
        "a_dt": pc.strptime(
            cola,
            format="%Y-%m-%d", unit="ns", 
            error_is_null=True
        ),
    }
    table = pa.Table.from_pydict(dt_cols)

    actual = pl.Series(columns_equal(table["a_dt"], table["b"]))
    expected = pl.Series(table["exp"])
    assert_series_equal(actual, expected, check_names=False)


def test_decimal_float_columns_equal():
    table = pa.Table.from_pydict({
        "a": [Decimal("1"), Decimal("1.3"), Decimal("1.000003"), Decimal("1.000000004"), 
              Decimal("1.3"), None, None, Decimal("1")],
        "b": [1, 1.3, 1.000003, 1.000000003, 1.2, None, 1, None],
        "expected": [True, True, True, False, False, True, False, False]
    })   
    actual_out = pl.Series(columns_equal(table["a"], table["b"]))
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_float_columns_equal_rel():
    table = pa.Table.from_pydict({
        "a": [Decimal("1"), Decimal("1.3"), Decimal("1.000003"), Decimal("1.000000004"),
                Decimal("1.3"), None, None, Decimal("1")],
        "b": [1, 1.3, 1.000003, 1.000000003, 1.2, None, 1, None],
        "expected": [True, True, True, True, False, True, False, False]
    })
    actual_out = pl.Series(columns_equal(table["a"], table["b"], abs_tol=0.001))
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_columns_equal():
    table = pa.Table.from_pydict({
        "a": [Decimal("1"), Decimal("1.3"), Decimal("1.000003"), Decimal("1.000000004"),
                Decimal("1.3"), None, None, Decimal("1")],
        "b": [Decimal("1"), Decimal("1.3"), Decimal("1.000003"), Decimal("1.000000003"),
                Decimal("1.2"), None, Decimal("1"), None],
        "expected": [True, True, True, False, False, True, False, False]
    })
    actual_out = pl.Series(columns_equal(table["a"], table["b"]))
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_columns_equal_rel():
    table = pa.Table.from_pylist(
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
            {"a": None, "b": None, "expected": True},
            {"a": None, "b": Decimal("1"), "expected": False},
            {"a": Decimal("1"), "b": None, "expected": False},
        ]
    )
    actual_out = pl.Series(columns_equal(table["a"], table["b"], abs_tol=0.001))
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_infinity_and_beyond():
    table = pa.Table.from_pylist(
        [
            {"a": np.inf, "b": np.inf, "expected": True},
            {"a": -np.inf, "b": -np.inf, "expected": True},
            {"a": -np.inf, "b": np.inf, "expected": False},
            {"a": np.inf, "b": -np.inf, "expected": False},
            {"a": 1, "b": 1, "expected": True},
            {"a": 1, "b": 0, "expected": False},
        ]
    )
    actual_out = pl.Series(columns_equal(table["a"], table["b"]))
    expect_out = pl.Series(table["expected"])
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_compare_table_setter_bad():
    table = pa.Table.from_pylist([{"a": 1, "c": 2}, {"a": 2, "c": 2}])
    table_same_col_names = pa.Table.from_pylist([{"a": 1, "A": 2}, {"a": 2, "A": 2}])
    table_dupe = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 3}])
    with raises(TypeError, match="Dataframe is not a pyarrow streamable object"):
        PyArrowCompare("a", "a", ["a"])
    with raises(ValueError, match="df1 must have all columns from join_columns"):
        table_clone = table
        PyArrowCompare(table, table_clone, ["b"])
    with raises(
        ValueError, match="df1 must have unique column names"
    ):
        table_same_col_names_clone = table_same_col_names
        PyArrowCompare(table_same_col_names, table_same_col_names_clone, ["a"])
    
    table_dupe_clone = table_dupe
    assert PyArrowCompare(table_dupe, table_dupe_clone, ["a", "b"]).df1.equals(table_dupe)


def test_compare_table_setter_good():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    table2 = pa.Table.from_pylist([{"A": 1, "B": 2}, {"A": 2, "B": 3}])
    compare = PyArrowCompare(table1, table2, ["a"])
    assert compare.df1.equals(table1)
    assert compare.df2.equals(table2.rename_columns(["a", "b"])) # Since pyarrow tables remain unchanged
    assert compare.join_columns == ["a"]
    compare = PyArrowCompare(table1, table2, ["A", "b"])
    assert compare.df1.equals(table1)
    assert compare.df2.equals(table2.rename_columns(["a", "b"]))
    assert compare.join_columns == ["a", "b"]


def test_compare_table_setter_different_cases():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    table2 = pa.Table.from_pylist([{"A": 1, "b": 2}, {"A": 2, "b": 3}])
    compare = PyArrowCompare(table1, table2, ["a"])
    assert compare.df1.equals(table1)
    assert compare.df2.equals(table2.rename_columns(["a", "b"]))


def test_compare_table_setter_bad_index():
    table = pa.Table.from_pylist([{"a": 1, "A": 2}, {"a": 2, "A": 2}])
    with raises(TypeError, match="Dataframe is not a pyarrow streamable object"):
        PyArrowCompare("a", "a", join_columns="a")
    with raises(
        ValueError, match="df1 must have unique column names"
    ):
        table_clone = table
        PyArrowCompare(table, table_clone, join_columns="a")


def test_compare_table_setter_good_index():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    compare = PyArrowCompare(table1, table2, join_columns="a")
    assert compare.df1.equals(table1)
    assert compare.df2.equals(table2)


def test_columns_overlap():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    compare = PyArrowCompare(table1, table2, ["a"])
    assert compare.df1_unq_columns() == set()
    assert compare.df2_unq_columns() == set()
    assert compare.intersect_columns() == {"a", "b"}


def test_columns_no_overlap():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2, "d": "oh"}, {"a": 2, "b": 3, "d": "ya"}])
    compare = PyArrowCompare(table1, table2, ["a"])
    assert compare.df1_unq_columns() == {"c"}
    assert compare.df2_unq_columns() == {"d"}
    assert compare.intersect_columns() == {"a", "b"}


def test_columns_maintain_order_through_set_operations():
    data = [
        ("A", 0, 1, 2, 3, 4, -2),
        ("B", 0, 2, 2, 3, 4, -3),
    ]
    table1 = pa.Table.from_arrays(
        list(zip(*data)), # Transpose the data to match the expected column-wise format
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64(), 1)),
            pa.field("f", pa.int64()),
            pa.field("g", pa.int64()),
            pa.field("b", pa.int64()),
            pa.field("h", pa.int64()),
            pa.field("a", pa.int64()),
            pa.field("c", pa.int64()),
        ])
    )
    data2 = [
        ("A", 0, 1, 2, -1, 4, -3),
        ("B", 1, 2, 3, -1, 4, -2),
    ]
    table2 = pa.Table.from_arrays(
        list(zip(*data2)),
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64(), 1)),
            pa.field("e", pa.int64()),
            pa.field("h", pa.int64()),
            pa.field("b", pa.int64()),
            pa.field("a", pa.int64()),
            pa.field("g", pa.int64()),
            pa.field("d", pa.int64()),
        ])
    )
    compare = PyArrowCompare(table1, table2, ["join"])
    assert list(compare.df1_unq_columns()) == ["f", "c"]
    assert list(compare.df2_unq_columns()) == ["e", "d"]
    assert list(compare.intersect_columns()) == ["join", "g", "b", "h", "a"]


def test_10k_rows():
    rng = np.random.default_rng()
    data = rng.integers(0, 100, size=(2, 10000)), 
    table1 = pa.Table.from_arrays(
        [
            np.arange(10000),
            data[0][0],
            data[0][1],
        ],
        schema=pa.schema([
            pa.field("a", pa.int64()),
            pa.field("b", pa.int64()),
            pa.field("c", pa.int64()),
        ])
    )
    idx = table1.schema.get_field_index("b")
    table2 = table1.set_column(idx, "b", pc.add(table1["b"], 0.1))
    compare_tol = PyArrowCompare(table1, table2, ["a"], abs_tol=0.2)
    assert compare_tol.matches()
    assert len(compare_tol.df1_unq_rows) == 0
    assert len(compare_tol.df2_unq_rows) == 0
    assert compare_tol.intersect_columns() == {"a", "b", "c"}
    assert compare_tol.all_columns_match()
    assert compare_tol.all_rows_overlap()
    assert compare_tol.intersect_rows_match()

    compare_no_tol = PyArrowCompare(table1, table2, ["a"])
    assert not compare_no_tol.matches()
    assert len(compare_no_tol.df1_unq_rows) == 0
    assert len(compare_no_tol.df2_unq_rows) == 0
    assert compare_no_tol.intersect_columns() == {"a", "b", "c"}
    assert compare_no_tol.all_columns_match()
    assert compare_no_tol.all_rows_overlap()
    assert not compare_no_tol.intersect_rows_match()


def test_subset(caplog):
    caplog.set_level(logging.DEBUG)
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    table2 = pa.Table.from_pylist([{"a": 1, "c": "hi"}])
    comp = PyArrowCompare(table1, table2, ["a"])
    assert comp.subset()
    assert "Checking equality" in caplog.text


def test_not_subset(caplog):
    caplog.set_level(logging.INFO)
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "great"}])
    comp = PyArrowCompare(table1, table2, ["a"])
    assert not comp.subset()
    assert "c: 1 / 2 (50.00%) match" in caplog.text


def test_large_subset():
    rng = np.random.default_rng()
    data = rng.integers(0, 100, size=(2, 10000)),
    table1 = pa.Table.from_arrays(
        [
            np.arange(10000),
            data[0][0],
            data[0][1],
        ],
        schema=pa.schema([
            pa.field("a", pa.int64()),
            pa.field("b", pa.int64()),
            pa.field("c", pa.int64()),
        ])
    )
    indices = np.random.choice(len(table1), size=50, replace=False)
    table2 = table1.select(["a", "b"]).take(indices)
    comp = PyArrowCompare(table1, table2, ["a"])
    assert not comp.matches()
    assert comp.subset()


def test_string_joiner():
    table1 = pa.Table.from_pylist([{"ab": 1, "bc": 2}, {"ab": 2, "bc": 2}])
    table2 = pa.Table.from_pylist([{"ab": 1, "bc": 2}, {"ab": 2, "bc": 2}])
    compare = PyArrowCompare(table1, table2, "ab")
    assert compare.matches()


def test_float_and_string_with_joins():
    table1 = pa.Table.from_pylist([{"a": float("1"), "b": 2}, {"a": float("2"), "b": 2}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    with raises(ArrowInvalid):
        PyArrowCompare(table1, table2, "a")


def test_decimal_with_nulls():
    table1 = pa.Table.from_pylist([{"a": 1, "b": Decimal("2")}, {"a": 2, "b": Decimal("2")}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 2}, {"a": 3, "b": 2}])
    compare = PyArrowCompare(table1, table2, "a")
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_joins():
    table1 = pa.Table.from_pylist([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    table2 = pa.Table.from_pylist([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    compare = PyArrowCompare(table1, table2, "a")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_temp_column_name():
    table1 = pa.Table.from_pylist([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    table2 = pa.Table.from_pylist(
        [{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    actual = temp_column_name(table1, table2)
    assert actual == "_temp_0"


def test_temp_column_name_one_has():
    table1 = pa.Table.from_pylist([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    table2 = pa.Table.from_pylist(
        [{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}]
    )
    actual = temp_column_name(table1, table2)
    assert actual == "_temp_1"


def test_temp_column_name_both_have_temp_1():
    table1 = pa.Table.from_pylist([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    table2 = pa.Table.from_pylist(
        [
            {"_temp_0": "hi", "b": 2},
            {"_temp_0": "bye", "b": 2},
            {"a": "back fo mo", "b": 3},
        ]
    )
    actual = temp_column_name(table1, table2)
    assert actual == "_temp_1"


def test_temp_column_name_both_have_temp_2():
    table1 = pa.Table.from_pylist([{"_temp_0": "hi", "b": 2}, {"_temp_0": "bye", "b": 2}])
    table2 = pa.Table.from_pylist(
        [
            {"_temp_0": "hi", "b": 2, "_temp_1": None, "a": None},
            {"_temp_0": None, "b": 2, "_temp_1": "bye", "a": None},
            {"_temp_0": None, "b": 3, "_temp_1": None, "a": "back fo mo"},
        ]
    )
    actual = temp_column_name(table1, table2)
    assert actual == "_temp_2"


def test_temp_column_name_one_already():
    table1 = pa.Table.from_pylist([{"_temp_1": "hi", "b": 2}, {"_temp_1": "bye", "b": 2}])
    table2 = pa.Table.from_pylist(
        [
            {"_temp_1": "hi", "b": 2},
            {"_temp_1": "bye", "b": 2},
            {"a": "back fo mo", "b": 3},
        ]
    )
    actual = temp_column_name(table1, table2)
    assert actual == "_temp_0"


# Duplicate testing!
def test_simple_dupes_one_field():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    compare = PyArrowCompare(table1, table2, join_columns=["a"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    compare.report()


def test_simple_dupes_two_fields():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 2}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 2}])
    compare = PyArrowCompare(table1, table2, join_columns=["a", "b"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    compare.report()


def test_simple_dupes_one_field_two_vals_1():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    compare = PyArrowCompare(table1, table2, join_columns=["a"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    compare.report()


def test_simple_dupes_one_field_two_vals_2():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 0}])
    compare = PyArrowCompare(table1, table2, join_columns=["a"])
    assert not compare.matches()
    assert len(compare.df1_unq_rows) == 1
    assert len(compare.df2_unq_rows) == 1
    assert len(compare.intersect_rows) == 1
    # Just render the report to make sure it renders.
    compare.report()


def test_simple_dupes_one_field_three_to_two_vals():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 0}, {"a": 1, "b": 0}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    compare = PyArrowCompare(table1, table2, join_columns=["a"])
    assert not compare.matches()
    assert len(compare.df1_unq_rows) == 1
    assert len(compare.df2_unq_rows) == 0
    assert len(compare.intersect_rows) == 2
    # Just render the report to make sure it renders.
    compare.report()

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
    table1 = pa._csv.read_csv(
                io.BytesIO(data.encode()), 
                parse_options=pa._csv.ParseOptions(delimiter=","), 
            )
    table2 = table1
    compare_acct = PyArrowCompare(table1, table2, join_columns=["acct_id"])
    assert compare_acct.matches()
    compare_unq = PyArrowCompare(
        table1,
        table2,
        join_columns=["acct_id", "acct_sfx_num", "trxn_post_dt", "trxn_post_seq_num"],
    )
    assert compare_unq.matches()
    # Just render the report to make sure it renders.
    compare_acct.report()
    compare_unq.report()


def test_strings_with_joins_with_ignore_spaces():
    table1 = pa.Table.from_pylist([{"a": "hi", "b": " A"}, {"a": "bye", "b": "A"}])
    table2 = pa.Table.from_pylist([{"a": "hi", "b": "A"}, {"a": "bye", "b": "A "}])
    compare = PyArrowCompare(table1, table2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = PyArrowCompare(table1, table2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_joins_with_ignore_case():
    table1 = pa.Table.from_pylist([{"a": "hi", "b": "a"}, {"a": "bye", "b": "A"}])
    table2 = pa.Table.from_pylist([{"a": "hi", "b": "A"}, {"a": "bye", "b": "a"}])
    compare = PyArrowCompare(table1, table2, "a", ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = PyArrowCompare(table1, table2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_joins_with_ignore_spaces():
    table1 = pa.Table.from_pylist([{"a": 1, "b": " A"}, {"a": 2, "b": "A"}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": "A"}, {"a": 2, "b": "A "}])
    compare = PyArrowCompare(table1, table2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = PyArrowCompare(table1, table2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_joins_with_ignore_case():
    table1 = pa.Table.from_pylist([{"a": 1, "b": "a"}, {"a": 2, "b": "A"}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": "A"}, {"a": 2, "b": "a"}])
    compare = PyArrowCompare(table1, table2, "a", ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = PyArrowCompare(table1, table2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_joins_with_ignore_spaces():
    table1 = pa.Table.from_pylist([{"a": 1, "b": " A"}, {"a": 2, "b": "A"}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": "A"}, {"a": 2, "b": "A "}])
    compare = PyArrowCompare(table1, table2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_joins_with_ignore_case():
    table1 = pa.Table.from_pylist([{"a": 1, "b": "a"}, {"a": 2, "b": "A"}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": "A"}, {"a": 2, "b": "a"}])
    compare = PyArrowCompare(table1, table2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_full_join_counts_all_matches():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    compare = PyArrowCompare(table1, table2, ["a", "b"], ignore_spaces=False)
    assert compare.count_matching_rows() == 2


def test_strings_with_ignore_spaces_and_join_columns():
    table1 = pa.Table.from_pylist([{"a": "hi", "b": "A"}, {"a": "bye", "b": "A"}])
    table2 = pa.Table.from_pylist([{"a": " hi ", "b": "A"}, {"a": " bye ", "b": "A"}])
    compare = PyArrowCompare(table1, table2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.count_matching_rows() == 0

    compare = PyArrowCompare(table1, table2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2


def test_integers_with_ignore_spaces_and_join_columns():
    table1 = pa.Table.from_pylist([{"a": 1, "b": "A"}, {"a": 2, "b": "A"}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": "A"}, {"a": 2, "b": "A"}])
    compare = PyArrowCompare(table1, table2, "a", ignore_spaces=False)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 2

    compare = PyArrowCompare(table1, table2, "a", ignore_spaces=True)
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
    table = pa._csv.read_csv(
        io.BytesIO(data1.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","), 
    )
    table2 = pa._csv.read_csv(
        io.BytesIO(data2.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","), 
    )
    compare = PyArrowCompare(table, table2, "acct_id")

    output = compare.sample_mismatch(column="name", sample_count=1)
    assert output.shape[0] == 1
    assert output["name_df1"] != output["name_df2"]

    output = compare.sample_mismatch(column="name", sample_count=2)
    assert output.shape[0] == 2
    assert output["name_df1"] != output["name_df2"]

    output = compare.sample_mismatch(column="name", sample_count=3)
    assert output.shape[0] == 2
    assert output["name_df2"] != ["name_df1"]


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
    table1 = pa._csv.read_csv(
        io.BytesIO(data1.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","), 
    )
    table2 = pa._csv.read_csv(
        io.BytesIO(data2.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","), 
    )
    compare = PyArrowCompare(table1, table2, "acct_id")

    output = compare.all_mismatch()
    assert output.shape[0] == 4
    assert output.shape[1] == 9

    diff = pc.not_equal(output["name_df1"], output["name_df2"])
    assert pc.sum(diff.cast(pa.int32())).as_py() == 2
    assert pc.sum(pc.invert(diff).cast(pa.int32())).as_py() == 2

    diff_amt = pc.not_equal(output["dollar_amt_df1"], output["dollar_amt_df2"])
    assert pc.sum(diff_amt.cast(pa.int32())).as_py() == 1
    assert pc.sum(pc.invert(diff_amt).cast(pa.int32())).as_py() == 3

    # mask is equivalent to eq_missing(...) used in polars test
    mask_float = pc.or_(
            pc.equal(output["float_fld_df1"], output["float_fld_df2"]).fill_null(False),
            pc.and_(pc.is_null(output["float_fld_df1"]), pc.is_null(output["float_fld_df2"]))
        )
    assert output.filter(pc.invert(mask_float)).num_rows == 3
    assert output.filter(mask_float).num_rows == 1

    mask_date = pc.or_(
            pc.equal(output["date_fld_df1"], output["date_fld_df2"]).fill_null(False),
            pc.and_(pc.is_null(output["date_fld_df1"]), pc.is_null(output["date_fld_df2"]))
        )
    assert output.filter(pc.invert(mask_date)).num_rows == 4
    assert output.filter(mask_date).num_rows == 0


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
    table1 = pa._csv.read_csv(
        io.BytesIO(data1.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","), 
    )
    table2 = pa._csv.read_csv(
        io.BytesIO(data2.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","), 
    )
    compare = PyArrowCompare(table1, table2, "acct_id")

    output = compare.all_mismatch()
    assert output.shape[0] == 4
    assert output.shape[1] == 9

    diff = pc.not_equal(output["name_df1"], output["name_df2"])
    assert pc.sum(diff.cast(pa.int32())).as_py() == 0
    assert pc.sum(pc.invert(diff).cast(pa.int32())).as_py() == 4

    diff_amt = pc.not_equal(output["dollar_amt_df1"], output["dollar_amt_df2"])
    assert pc.sum(diff_amt.cast(pa.int32())).as_py() == 0
    assert pc.sum(pc.invert(diff_amt).cast(pa.int32())).as_py() == 4

    # mask is equivalent to eq_missing(...) used in polars test
    mask_float = pc.or_(
            pc.equal(output["float_fld_df1"], output["float_fld_df2"]).fill_null(False),
            pc.and_(pc.is_null(output["float_fld_df1"]), pc.is_null(output["float_fld_df2"]))
        )
    assert output.filter(pc.invert(mask_float)).num_rows == 3
    assert output.filter(mask_float).num_rows == 1

    mask_date = pc.or_(
            pc.equal(output["date_fld_df1"], output["date_fld_df2"]).fill_null(False),
            pc.and_(pc.is_null(output["date_fld_df1"]), pc.is_null(output["date_fld_df2"]))
        )
    assert output.filter(pc.invert(mask_date)).num_rows == 4
    assert output.filter(mask_date).num_rows == 0


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
10000001241,1111.05,Lucille Bluth,,
"""

    data2 = """acct_id,dollar_amt,name,float_fld,date_fld
10000001234,123.45,George Maharis,14530.155,
10000001235,0.45,Michael Bluth,,
10000001236,1345,George Bluth,1,
10000001237,123456,Bob Loblaw,345.12,
10000001238,1.05,Lucille Bluth,111,
"""
    # Defining schema to ensure date_fld is read properly
    convert_options = pa._csv.ConvertOptions(
        column_types={
            "acct_id": pa.int64(),
            "dollar_amt": pa.float64(),
            "name": pa.string(),
            "float_fld": pa.float64(),
            "date_fld": pa.date32(),
        }
    )
    table1 = pa._csv.read_csv(
        io.BytesIO(data1.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","),
        convert_options=convert_options 
    )
    table2 = pa._csv.read_csv(
        io.BytesIO(data2.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","),
        convert_options=convert_options
    )
    compare = PyArrowCompare(table1, table2, "acct_id")

    output = compare.all_mismatch(ignore_matching_cols=True)

    assert output.shape[0] == 4
    assert output.shape[1] == 5

    # mask is equivalent to eq_missing(...) used in polars test
    mask_float = pc.or_(
            pc.equal(output["float_fld_df1"], output["float_fld_df2"]).fill_null(False),
            pc.and_(pc.is_null(output["float_fld_df1"]), pc.is_null(output["float_fld_df2"]))
        )
    assert output.filter(pc.invert(mask_float)).num_rows == 3
    assert output.filter(mask_float).num_rows == 1

    mask_date = pc.or_(
            pc.equal(output["date_fld_df1"], output["date_fld_df2"]).fill_null(False),
            pc.and_(pc.is_null(output["date_fld_df1"]), pc.is_null(output["date_fld_df2"]))
        )
    assert output.filter(pc.invert(mask_date)).num_rows == 4
    assert output.filter(mask_date).num_rows == 0

    assert not ("name_df1" in output and "name_df2" in output)
    assert not ("dollar_amt_df1" in output and "dollar_amt_df2" in output)


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
    # Defining schema to ensure date_fld is read properly
    convert_options = pa._csv.ConvertOptions(
        column_types={
            "acct_id": pa.int64(),
            "dollar_amt": pa.float64(),
            "name": pa.string(),
            "float_fld": pa.float64(),
            "date_fld": pa.date32(),
        }
    )
    table1 = pa._csv.read_csv(
        io.BytesIO(data1.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","),
        convert_options=convert_options 
    )
    table2 = pa._csv.read_csv(
        io.BytesIO(data2.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","),
        convert_options=convert_options
    )
    compare = PyArrowCompare(table1, table2, "acct_id")

    output = compare.all_mismatch(ignore_matching_cols=True)

    assert output.shape[0] == 4
    assert output.shape[1] == 5

    # mask is equivalent to eq_missing(...) used in polars test
    mask_float = pc.or_(
            pc.equal(output["float_fld_df1"], output["float_fld_df2"]).fill_null(False),
            pc.and_(pc.is_null(output["float_fld_df1"]), pc.is_null(output["float_fld_df2"]))
        )
    assert output.filter(pc.invert(mask_float)).num_rows == 3
    assert output.filter(mask_float).num_rows == 1

    mask_date = pc.or_(
            pc.equal(output["date_fld_df1"], output["date_fld_df2"]).fill_null(False),
            pc.and_(pc.is_null(output["date_fld_df1"]), pc.is_null(output["date_fld_df2"]))
        )
    assert output.filter(pc.invert(mask_date)).num_rows == 4
    assert output.filter(mask_date).num_rows == 0

    assert not ("name_df1" in output and "name_df2" in output)
    assert not ("dollar_amt_df1" in output and "dollar_amt_df2" in output)


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
    # Defining schema to ensure date_fld is read properly
    convert_options = pa._csv.ConvertOptions(
        column_types={
            "acct_id": pa.int64(),
            "dollar_amt": pa.float64(),
            "name": pa.string(),
            "float_fld": pa.float64(),
            "date_fld": pa.date32(),
        }
    )
    table1 = pa._csv.read_csv(
        io.BytesIO(data1.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","),
        convert_options=convert_options 
    )
    table2 = pa._csv.read_csv(
        io.BytesIO(data2.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter=","),
        convert_options=convert_options
    )
    compare = PyArrowCompare(table1, table2, "acct_id")

    output = compare.all_mismatch()
    assert output.shape[0] == 4
    assert output.shape[1] == 9

    diff = pc.not_equal(output["name_df1"], output["name_df2"])
    assert pc.sum(diff.cast(pa.int32())).as_py() == 2
    assert pc.sum(pc.invert(diff).cast(pa.int32())).as_py() == 2

    diff_amt = pc.not_equal(output["dollar_amt_df1"], output["dollar_amt_df2"])
    assert pc.sum(diff_amt.cast(pa.int32())).as_py() == 1
    assert pc.sum(pc.invert(diff_amt).cast(pa.int32())).as_py() == 3

    # mask is equivalent to eq_missing(...) used in polars test
    mask_float = pc.or_(
            pc.equal(output["float_fld_df1"], output["float_fld_df2"]).fill_null(False),
            pc.and_(pc.is_null(output["float_fld_df1"]), pc.is_null(output["float_fld_df2"]))
        )
    assert output.filter(pc.invert(mask_float)).num_rows == 3
    assert output.filter(mask_float).num_rows == 1

    mask_date = pc.or_(
            pc.equal(output["date_fld_df1"], output["date_fld_df2"]).fill_null(False),
            pc.and_(pc.is_null(output["date_fld_df1"]), pc.is_null(output["date_fld_df2"]))
        )
    assert output.filter(pc.invert(mask_date)).num_rows == 4
    assert output.filter(mask_date).num_rows == 0


MAX_DIFF_table = pa.Table.from_pydict(
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
        "null_floats": [None, 1.1, 1, 1, 1],
        "strings": ["1", "1", "1", "1.1", "1"],
        "mixed_strings": ["1", "1", "1", "2", "some string"],
        "infinity": [1, 1, 1, 1, np.inf],
        "nulls": [None, None, None, None, None],
        "some_nulls": [10, 10, 10, None, None],
    },
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
        ("some_nulls", 9),
    ],
)
def test_calculate_max_diff(column, expected):
    assert np.isclose(
        calculate_max_diff(MAX_DIFF_table["base"], MAX_DIFF_table[column]), expected
    )

def test_calculate_max_diff_with_nulls():
    with raises(ValueError, match="Cannot calculate max difference for null type columns."):
        calculate_max_diff(MAX_DIFF_table["base"], MAX_DIFF_table["nulls"])


def test_dupes_with_nulls():
    table1 = pa.Table.from_pydict(
        {
            "fld_1": [1, 2, 2, 3, 3, 4, 5, 5],
            "fld_2": ["A", None, None, None, None, None, None, None],
        },
    )
    table2 = pa.Table.from_pydict(
        {"fld_1": [1, 2, 3, 4, 5], "fld_2": ["A", None, None, None, None]},
    )
    comp = PyArrowCompare(table1, table2, join_columns=["fld_1", "fld_2"])
    assert comp.subset()


@pytest.mark.parametrize(
    "dataframe,expected",
    [
        (
            pa.Table.from_pydict({"a": [1, 2, 3], "b": [1, 2, 3]}),
            pa.array([1, 1, 1], safe=False),
        ),
        (
            pa.Table.from_pydict(
                {"a": ["a", "a", "DATACOMPY_NULL"], "b": [1, 1, 2]}, 
            ),
            pa.array([1, 2, 1], safe=False),
        ),
        (
            pa.Table.from_pydict({"a": [-999, 2, 3], "b": [1, 2, 3]} ),
            pa.array([1, 1, 1], safe=False),
        ),
        (
            pa.Table.from_pydict({"a": [1, np.nan, np.nan], "b": [1, 2, 2]}),
            pa.array([1, 1, 2], safe=False),
        ),
        (
            pa.Table.from_pydict(
                {"a": ["1", None, None], "b": ["1", "2", "2"]},
            ),
            pa.array([1, 1, 2], safe=False),
        ),
        (
            pa.Table.from_pydict(
                {"a": [datetime(2018, 1, 1), None, None], "b": ["1", "2", "2"]},
            ),
            pa.array([1, 1, 2], safe=False),
        ),
    ],
)
def test_generate_id_within_group(dataframe, expected):
    assert pc.all(generate_id_within_group(dataframe, ["a", "b"]) == expected)


def test_lower():
    """This function tests the toggle to use lower case for column names or not"""
    # should match
    table1 = pa.Table.from_pydict({"a": [1, 2, 3], "b": [0, 1, 2]})
    table2 = pa.Table.from_pydict({"a": [1, 2, 3], "B": [0, 1, 2]})
    compare = PyArrowCompare(table1, table2, join_columns=["a"])
    assert compare.matches()
    # should not match
    table1 = pa.Table.from_pydict({"a": [1, 2, 3], "b": [0, 1, 2]})
    table2 = pa.Table.from_pydict({"a": [1, 2, 3], "B": [0, 1, 2]})
    compare = PyArrowCompare(table1, table2, join_columns=["a"], cast_column_names_lower=False)
    assert not compare.matches()

    # test join column
    # should match
    table1 = pa.Table.from_pydict({"a": [1, 2, 3], "b": [0, 1, 2]})
    table2 = pa.Table.from_pydict({"A": [1, 2, 3], "B": [0, 1, 2]})
    compare = PyArrowCompare(table1, table2, join_columns=["a"])
    assert compare.matches()
    # should fail because "a" is not found in table2
    table1 = pa.Table.from_pydict({"a": [1, 2, 3], "b": [0, 1, 2]})
    table2 = pa.Table.from_pydict({"A": [1, 2, 3], "B": [0, 1, 2]})
    expected_message = "df2 must have all columns from join_columns: {'a'}"
    with raises(ValueError, match=expected_message):
        compare = PyArrowCompare(
            table1, table2, join_columns=["a"], cast_column_names_lower=False
        )


@mock.patch("datacompy.pyarrow.render")
@mock.patch("datacompy.pyarrow.save_html_report")
def test_save_html(mock_save_html, mock_render):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    compare = PyArrowCompare(table1, table2, join_columns=["a"])

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


def test_full_join_counts_no_matches():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 3}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 4}, {"a": 1, "b": 5}])
    compare = PyArrowCompare(table1, table2, ["a", "b"], ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert not compare.intersect_rows_match()
    assert compare.count_matching_rows() == 0
    assert compare.sample_mismatch(column="a").equals(
        pa.Table.from_pydict({"a": [1, 1, 1, 1]}),
    )
    assert compare.sample_mismatch(column="a").sort_by([("a", "ascending")]).equals(
        pa.Table.from_pydict({"a": [1, 1, 1, 1]}),
    )
    assert compare.sample_mismatch(column="b").sort_by([("b", "ascending")]).equals(
        pa.Table.from_pydict({"b": [2, 3, 4, 5]}),
    )
    assert compare.all_mismatch().sort_by([("a", "ascending"), ("b", "ascending")]).equals(
        pa.Table.from_pylist(
            [{"a": 1, "b": 2}, {"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 1, "b": 5}]
        ),
    )


def test_full_join_counts_some_matches():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 3}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 1, "b": 5}])
    compare = PyArrowCompare(table1, table2, ["a", "b"], ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 1
    assert compare.sample_mismatch(column="a").sort_by([("a", "ascending")]).equals(
        pa.Table.from_pydict({"a": [1, 1]}),
    )
    assert compare.sample_mismatch(column="b").sort_by([("b", "ascending")]).equals(
        pa.Table.from_pydict({"b": [3, 5]}),
    )
    assert compare.all_mismatch().sort_by([("a", "ascending"), ("b", "ascending")]).equals(
        pa.Table.from_pylist(
            [
                {"a": 1, "b": 3},
                {"a": 1, "b": 5},
            ]
        ),
    )


def test_non_full_join_counts_no_matches():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2, "c": 4}, {"a": 1, "b": 3, "c": 4}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 4, "d": 5}, {"a": 1, "b": 5, "d": 5}])
    compare = PyArrowCompare(table1, table2, ["a", "b"], ignore_spaces=False)
    assert not compare.matches()
    assert not compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert not compare.intersect_rows_match()
    assert compare.count_matching_rows() == 0
    assert compare.sample_mismatch(column="a").sort_by([("a", "ascending")]).equals(
        pa.Table.from_pydict({"a": [1, 1, 1, 1]}),
    )
    assert compare.sample_mismatch(column="b").sort_by([("b", "ascending")]).equals(
        pa.Table.from_pydict({"b": [2, 3, 4, 5]}),
    )
    assert compare.all_mismatch().sort_by([("a", "ascending"), ("b", "ascending")]).equals(
        pa.Table.from_pylist(
            [{"a": 1, "b": 2}, {"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 1, "b": 5}]
        ),
    )


def test_non_full_join_counts_some_matches():
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2, "c": 4}, {"a": 1, "b": 3, "c": 4}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2, "d": 5}, {"a": 1, "b": 5, "d": 5}])
    compare = PyArrowCompare(table1, table2, ["a", "b"], ignore_spaces=False)
    assert not compare.matches()
    assert not compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.intersect_rows_match()
    assert compare.count_matching_rows() == 1
    assert compare.sample_mismatch(column="a").sort_by([("a", "ascending")]).equals(
        pa.Table.from_pydict({"a": [1, 1]}),
    )
    assert compare.sample_mismatch(column="b").sort_by([("b", "ascending")]).equals(
        pa.Table.from_pydict({"b": [3, 5]}),
    )
    assert compare.all_mismatch().sort_by([("a", "ascending"), ("b", "ascending")]).equals(
        pa.Table.from_pylist(
            [
                {"a": 1, "b": 3},
                {"a": 1, "b": 5},
            ]
        ),
    )


def test_categorical_column():
    table = pa.Table.from_pydict(
        {
            "idx": [1, 2, 3],
            "foo": ["A", "B", None],
            "bar": ["A", "B", None],
            "foo_bad": ["    A   ", "B", None],
        },
    )

    actual_out = columns_equal(
        table["foo"], table["bar"], ignore_spaces=True, ignore_case=True
    )
    assert pc.all(actual_out)

    actual_out = columns_equal(
        table["foo"], table["foo_bad"], ignore_spaces=False, ignore_case=True
    )
    assert list(actual_out) == [pa.scalar(False), pa.scalar(True), pa.scalar(True)]

    compare = PyArrowCompare(table, table, join_columns=["idx"])
    assert pc.all(compare.intersect_rows["foo_match"])
    assert pc.all(compare.intersect_rows["bar_match"])


def test_string_as_numeric():
    table1 = pa.Table.from_pydict({"ID": [1], "REFER_NR": ["9998700990704001708177961516923014"]})
    table2 = pa.Table.from_pydict({"ID": [1], "REFER_NR": ["9998700990704001708177961516923015"]})
    actual_out = columns_equal(table1["REFER_NR"], table2["REFER_NR"])
    assert not pc.all(actual_out)


def test_single_date_columns_equal_to_string():
    data = """a|b|expected
2017-01-01|2017-01-01   |True
2017-01-02  |2017-01-02|True
2017-10-01  |2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    table = pa._csv.read_csv(
        io.BytesIO(data.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter="|"),
        convert_options=pa._csv.ConvertOptions(null_values=["NULL"])
    )
    col_a = pc.utf8_trim_whitespace(table["a"])
    col_a = pc.strptime(col_a, format="%Y-%m-%d", unit="ns", error_is_null=True)
    col_b = table["b"]

    actual_out = columns_equal(
        col_a, col_b, rel_tol=0.2, ignore_spaces=True
    )
    expect_out = table["expected"]
    assert expect_out.equals(actual_out)


def test_temporal_equal():
    data = """a|b|expected
2017-01-01|2017-01-01|True
2017-01-02|2017-01-02|True
2017-10-01|2017-10-10   |False
2017-01-01||False
|2017-01-01|False
||True"""
    table = pa._csv.read_csv(
        io.BytesIO(data.encode()), 
        parse_options=pa._csv.ParseOptions(delimiter="|"),
        convert_options=pa._csv.ConvertOptions(null_values=["NULL"])
    )
    expect_out = table["expected"]

    col_a = pc.strptime(table["a"], format="%Y-%m-%d", unit="ns", error_is_null=True)
    col_b = pc.strptime(table["b"], format="%Y-%m-%d", unit="ns", error_is_null=True)
    actual_out = columns_equal(col_a, col_b)
    assert expect_out.equals(actual_out)


def test_columns_equal_arrays():
    # all equal
    table1 = pa.Table.from_pydict(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64(), 1)),
        ])
    )
    table2 = pa.Table.from_pydict(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64(), 1)),
        ])
    )
    actual = columns_equal(table1["array_col"], table2["array_col"])
    assert pc.all(actual)

    # all mismatch
    table1 = pa.Table.from_pydict(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64(), 1)),
        ])
    )
    table2 = pa.Table.from_pydict(
        {"array_col": [[2], [3], [4], [5], [6]]},
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64(), 1)),
        ])
    )
    actual = columns_equal(table1["array_col"], table2["array_col"])
    assert not pc.all(actual)

    # some equal
    table1 = pa.Table.from_pydict(
        {"array_col": [[1], [2], [3], [4], [5]]},
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64(), 1)),
        ])
    )
    table2 = pa.Table.from_pydict(
        {"array_col": [[1], [1], [3], [4], [5]]},
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64(), 1)),
        ])
    )
    actual = columns_equal(table1["array_col"], table2["array_col"])
    expected = pa.array([True, False, True, True, True])
    assert pc.all(pc.equal(actual, expected))

    # empty
    table1 = pa.Table.from_pydict(
        {"array_col": [[], [], [], [], []]},
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64(), 0)),
        ])
    )
    table2 = pa.Table.from_pydict(
        {"array_col": [[], [], [], [], []]},
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64(), 0)),
        ])
    )
    actual = columns_equal(table1["array_col"], table2["array_col"])
    assert pc.all(actual)
    
    # different shapes, equivalent to pl.List
    table1 = pa.Table.from_pydict(
        {
            "array_col": [
                [],
                [None],
                [1, 2],
                [1, 3],
                [2, 3],
                [1, 2, 3],
            ]
        },
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64())),
        ])
    )
    table2 = pa.Table.from_pydict(
        {
            "array_col": [
                [],
                [None],
                [1, 2, 3],
                [1, 3],
                [2, 3],
                [1, 2],
            ]
        },
        schema=pa.schema([
            ("array_col", pa.list_(pa.int64())),
        ])
    )
    actual = columns_equal(table1["array_col"], table2["array_col"])
    expected = pa.array([True, True, False, True, True, False])
    assert pc.all(pc.equal(actual, expected))


@pytest.mark.parametrize(
    "input_data, ignore_spaces, ignore_case, expected",
    [  # string datatype should just passthough
        (
            pa.array(["  cat  ", "dog", "  mouse  ", None], type=pa.string()),
            True,
            True,
            pa.array(["CAT", "DOG", "MOUSE", None], type=pa.string()),
        ),
        # test case for integers
        (pa.array([1, 2, 3, 4]), True, True, pa.array([1, 2, 3, 4])),
        (pa.array([1, 2, 3, 4]), True, False, pa.array([1, 2, 3, 4])),
        (pa.array([1, 2, 3, 4]), False, True, pa.array([1, 2, 3, 4])),
        (pa.array([1, 2, 3, 4]), False, False, pa.array([1, 2, 3, 4])),
        # test case for floats
        (pa.array([1.1, 2.2, 3.3, 4.4]), True, True, pa.array([1.1, 2.2, 3.3, 4.4])),
        (pa.array([1.1, 2.2, 3.3, 4.4]), True, False, pa.array([1.1, 2.2, 3.3, 4.4])),
        (pa.array([1.1, 2.2, 3.3, 4.4]), False, True, pa.array([1.1, 2.2, 3.3, 4.4])),
        (
            pa.array([1.1, 2.2, 3.3, 4.4]),
            False,
            False,
            pa.array([1.1, 2.2, 3.3, 4.4]),
        ),
        # list of strings should just passthrough
        (
            pa.array([["  hello  ", "WORLD", "  Foo  ", None]]),
            True,
            True,
            pa.array([["  hello  ", "WORLD", "  Foo  ", None]]),
        ),
        # array of strings should just passthrough
        (
            pa.array([["  hello  ", "WORLD", "  Foo  ", None]]),
            True,
            True,
            pa.array([["  hello  ", "WORLD", "  Foo  ", None]]),
        ),
        # strings
        (
            pa.array(["  hello  ", "WORLD", "  Foo  ", None]),
            True,
            True,
            pa.array(["HELLO", "WORLD", "FOO", None]),
        ),
        (
            pa.array(["  hello  ", "WORLD", "  Foo  ", None]),
            True,
            False,
            pa.array(["hello", "WORLD", "Foo", None]),
        ),
        (
            pa.array(["  hello  ", "WORLD", "  Foo  ", None]),
            False,
            True,
            pa.array(["  HELLO  ", "WORLD", "  FOO  ", None]),
        ),
        (
            pa.array(["  hello  ", "WORLD", "  Foo  ", None]),
            False,
            False,
            pa.array(["  hello  ", "WORLD", "  Foo  ", None]),
        ),
        # emoji
        (
            pa.array(["👋", "🌍", "🍕", None]),
            True,
            True,
            pa.array(["👋", "🌍", "🍕", None]),
        ),
        (
            pa.array(["  👋  ", "🌍", "  🍕  ", None]),
            False,
            True,
            pa.array(["  👋  ", "🌍", "  🍕  ", None]),
        ),
    ],
)
def test_normalize_string_column(input_data, ignore_spaces, ignore_case, expected):
    result = pyarrow_normalize_string_column(
        input_data, ignore_spaces=ignore_spaces, ignore_case=ignore_case
    )
    assert result.equals(expected)


def test_custom_template_usage():
    """Test using a custom template with template_path parameter."""
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 4}])
    compare = PyArrowCompare(table1, table2, ["a"])

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
        # Should list the column with mismatches (b)
        assert "b" in result
        # Should show False for column b (has mismatches)
        assert "False" in result
    finally:
        # Clean up the temporary file
        if os.path.exists(template_path):
            os.unlink(template_path)


def test_template_without_extension():
    """Test that template files without .j2 extension still work."""
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    compare = PyArrowCompare(table1, table2, ["a"])

    # Create a test template without extension
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
        tmp.write("Template without extension\n")
        tmp.write(
            "Match status: {% if column_stats|selectattr('all_match', 'equalto', False)|list|length == 0 %}Match{% else %}No match{% endif %}"
        )
        template_path = tmp.name

    try:
        # Test with template that doesn't have .j2 extension
        result = compare.report(template_path=template_path)
        assert "Template without extension" in result
        assert "Match status: Match" in result
    finally:
        if os.path.exists(template_path):
            os.unlink(template_path)


def test_nonexistent_template():
    """Test that a clear error is raised when template file doesn't exist."""
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    compare = PyArrowCompare(table1, table2, ["a"])

    with pytest.raises(FileNotFoundError):
        compare.report(template_path="/nonexistent/path/template.j2")


def test_template_context_variables():
    """Test that all expected context variables are available in the template."""
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 4}])
    compare = PyArrowCompare(table1, table2, ["a"])

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
        # Should list the column with mismatches (b)
        assert "b" in result
    finally:
        if os.path.exists(template_path):
            os.unlink(template_path)


def test_html_report_generation():
    """Test that HTML report is properly generated and saved."""
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    table2 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 2, "b": 4}])
    compare = PyArrowCompare(table1, table2, ["a"])

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        html_file = os.path.join(temp_dir, "test_report.html")

        # Generate the report
        result = compare.report(html_file=html_file)

        # Check that the file was created
        assert os.path.exists(html_file)

        # Check that the file has content
        with open(html_file) as f:
            content = f.read()
            assert len(content) > 0
            # Should contain some HTML tags
            assert "<html" in content.lower()
            assert "</html>" in content.lower()

        # The result should be the same as the rendered HTML content
        assert isinstance(result, str)
        assert len(result) > 0


def test_per_column_tolerances() -> None:
    """Test comparison with per-column tolerances."""
    table1 = pa.Table.from_pydict(
        {"id": [1, 2, 3], "col1": [1.0, 2.0, 3.0], "col2": [1.0, 2.0, 3.0]}
    )
    table2 = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "col1": [1.1, 2.2, 3.3],  # Larger differences
            "col2": [1.01, 2.01, 3.01],  # Smaller differences
        }
    )

    compare = PyArrowCompare(
        table1,
        table2,
        join_columns=["id"],
        abs_tol={"col1": 0.5, "col2": 0.00001},  # col1 should match, col2 should not
    )

    col1_stats = next(stat for stat in compare.column_stats if stat["column"] == "col1")
    col2_stats = next(stat for stat in compare.column_stats if stat["column"] == "col2")
    assert col1_stats["unequal_cnt"] == 0
    assert col2_stats["unequal_cnt"] > 0
    assert compare._rel_tol_dict == {"default": 0.0}


def test_default_tolerance() -> None:
    """Test default tolerance behavior."""
    table1 = pa.Table.from_pydict({"id": [1, 2], "col1": [1.0, 2.0], "col2": [1.0, 2.0]})
    table2 = pa.Table.from_pydict({"id": [1, 2], "col1": [1.1, 2.1], "col2": [1.1, 2.1]})

    compare = PyArrowCompare(
        table1, table2, join_columns=["id"], abs_tol={"col1": 0.05, "default": 0.2}
    )

    col1_stats = next(stat for stat in compare.column_stats if stat["column"] == "col1")
    col2_stats = next(stat for stat in compare.column_stats if stat["column"] == "col2")
    assert col1_stats["unequal_cnt"] > 0  # col1 should not match (tolerance 0.05)
    assert col2_stats["unequal_cnt"] == 0  # col2 should match (default tolerance 0.2)
    assert compare._rel_tol_dict == {"default": 0.0}


def test_mixed_tolerances() -> None:
    """Test mixing absolute and relative tolerances."""
    table1 = pa.Table.from_pydict(
        {
            "id": [1, 2],
            "small_vals": [
                1.0,
                2.0,
            ],  # Small values where absolute tolerance matters more
            "large_vals": [
                1000.0,
                2000.0,
            ],  # Large values where relative tolerance matters more
        }
    )
    table2 = pa.Table.from_pydict(
        {"id": [1, 2], "small_vals": [1.1, 2.1], "large_vals": [1001.0, 2002.0]}
    )

    compare = PyArrowCompare(
        table1,
        table2,
        join_columns=["id"],
        abs_tol={"small_vals": 0.2, "default": 0.0},
        rel_tol={"large_vals": 0.001, "default": 0.0},
    )

    small_vals_stats = next(
        stat for stat in compare.column_stats if stat["column"] == "small_vals"
    )
    large_vals_stats = next(
        stat for stat in compare.column_stats if stat["column"] == "large_vals"
    )
    assert small_vals_stats["unequal_cnt"] == 0  # small_vals should match (abs_tol 0.2)
    assert (
        large_vals_stats["unequal_cnt"] == 0
    )  # large_vals should match (rel_tol 0.001 = 0.1%)
    assert compare._rel_tol_dict == {"large_vals": 0.001, "default": 0.0}
    assert compare._abs_tol_dict == {"small_vals": 0.2, "default": 0.0}


def test_custom_comparator_pyarrow():
    """Test that a custom comparator can be passed and used with PyArrow."""

    class StringLengthComparator(BaseComparator):
        """A custom comparator that matches strings based on length."""

        def compare(self, s1, s2):
            if s1.type == pa.string() and s2.type == pa.string():
                return pc.equal(pc.utf8_length(s1), pc.utf8_length(s2))
            return None

    table1 = pa.Table.from_pylist([{"id": 1, "value": "apple"}])
    table2 = pa.Table.from_pylist([{"id": 1, "value": "grape"}])

    # With custom comparator, it should match because 'apple' and 'grape' have the same length
    compare_custom = PyArrowCompare(
        table1, table2, join_columns=["id"], custom_comparators=[StringLengthComparator()]
    )
    assert compare_custom.matches()

    # Without custom comparator, it should not match
    compare_default = PyArrowCompare(table1, table2, join_columns=["id"])
    assert not compare_default.matches()

    # Test case where custom comparator does not apply (returns None)
    # and default comparison should be used.
    table3 = pa.Table.from_pylist([{"id": 1, "value": 10}])
    table4 = pa.Table.from_pylist([{"id": 1, "value": 20}])

    # With custom comparator, but it won't apply to integer 'value' column
    # so default comparison for integers should kick in, resulting in a mismatch.
    compare_custom_fallback = PyArrowCompare(
        table3, table4, join_columns=["id"], custom_comparators=[StringLengthComparator()]
    )
    assert not compare_custom_fallback.matches()

    # Test case where custom comparator does not apply (returns None)
    # and default comparison should be used.
    table5 = pa.Table.from_pylist([{"id": 1, "value": 10}])
    table6 = pa.Table.from_pylist([{"id": 1, "value": 10}])

    # With custom comparator, but it won't apply to integer 'value' column
    # so default comparison for integers should kick in, resulting in a match.
    compare_custom_fallback = PyArrowCompare(
        table5, table6, join_columns=["id"], custom_comparators=[StringLengthComparator()]
    )
    assert compare_custom_fallback.matches()

    # Ensure the StringLengthComparator is actually used for string columns
    table7 = pa.Table.from_pylist([{"id": 1, "value": "test"}])
    table8 = pa.Table.from_pylist([{"id": 1, "value": "abcd"}])

    compare_string_custom = PyArrowCompare(
        table7, table8, join_columns=["id"], custom_comparators=[StringLengthComparator()]
    )
    assert compare_string_custom.matches()

    compare_string_default = PyArrowCompare(table7, table8, join_columns=["id"])
    assert not compare_string_default.matches()

    # StringLengthComparator mismatch case
    table9 = pa.Table.from_pylist([{"id": 1, "value": "test"}])
    table10 = pa.Table.from_pylist([{"id": 1, "value": "abcde"}])

    compare_string_custom_mismatch = PyArrowCompare(
        table9, table10, join_columns=["id"], custom_comparators=[StringLengthComparator()]
    )
    assert not compare_string_custom_mismatch.matches()


def test_array_comparator_pyarrow():
    """Test that the PyArrowCompare can handle array columns."""
    # all equal
    table1 = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], [3, 4], [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64()))}),
    )
    table2 = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], [3, 4], [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64()))}),
    )
    compare = PyArrowCompare(table1, table2, join_columns=["id"])
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()

    # some mismatch (different order)
    table2_order = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], [4, 3], [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64()))}),
    )
    compare_order = PyArrowCompare(table1, table2_order, join_columns=["id"])
    assert not compare_order.matches()
    list_col_stats = next(
        stat for stat in compare_order.column_stats if stat["column"] == "list_col"
    )
    assert list_col_stats["unequal_cnt"] == 1
    assert list_col_stats["match_cnt"] == 2

    # with nulls matching
    table1_null = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], None, [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64(), 2))}),
    )
    table2_null = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], None, [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64(), 2))}),
    )
    compare_null = PyArrowCompare(table1_null, table2_null, join_columns=["id"])
    assert compare_null.matches()

    # with nulls mismatching
    table2_null_mismatch = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], [3, 4], [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64(), 2))}),
    )
    compare_null_mismatch = PyArrowCompare(
        table1_null, table2_null_mismatch, join_columns=["id"]
    )
    assert not compare_null_mismatch.matches()
    list_col_stats = next(
        stat
        for stat in compare_null_mismatch.column_stats
        if stat["column"] == "list_col"
    )
    assert list_col_stats["unequal_cnt"] == 1
    assert list_col_stats["match_cnt"] == 2


def test_list_comparator_pyarrow():
    """Test that the PyArrowCompare can handle list columns."""
    # all equal
    table1 = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], [3, 4], [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64()))}),
    )
    table2 = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], [3, 4], [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64()))}),
    )
    compare = PyArrowCompare(table1, table2, join_columns=["id"])
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()

    # some mismatch (different order)
    table2_order = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], [4, 3], [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64()))}),
    )
    compare_order = PyArrowCompare(table1, table2_order, join_columns=["id"])
    assert not compare_order.matches()
    list_col_stats = next(
        stat for stat in compare_order.column_stats if stat["column"] == "list_col"
    )
    assert list_col_stats["unequal_cnt"] == 1
    assert list_col_stats["match_cnt"] == 2

    # some mismatch (different shapes)
    table2_shape = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], [3, 4, 5], [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64()))}),
    )
    compare_shape = PyArrowCompare(table1, table2_shape, join_columns=["id"])
    assert not compare_shape.matches()
    list_col_stats = next(
        stat for stat in compare_shape.column_stats if stat["column"] == "list_col"
    )
    assert list_col_stats["unequal_cnt"] == 1
    assert list_col_stats["match_cnt"] == 2

    # with nulls matching
    table1_null = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], None, [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64()))}),
    )
    table2_null = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], None, [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64()))}),
    )
    compare_null = PyArrowCompare(table1_null, table2_null, join_columns=["id"])
    assert compare_null.matches()

    # with nulls mismatching
    table2_null_mismatch = pa.Table.from_pydict(
        {"id": [1, 2, 3], "list_col": [[1, 2], [3, 4], [5, 6]]},
        schema=pa.schema({
            pa.field("id", pa.int64()), 
            pa.field("list_col", pa.list_(pa.int64()))}),
    )
    compare_null_mismatch = PyArrowCompare(
        table1_null, table2_null_mismatch, join_columns=["id"]
    )
    assert not compare_null_mismatch.matches()
    list_col_stats = next(
        stat
        for stat in compare_null_mismatch.column_stats
        if stat["column"] == "list_col"
    )
    assert list_col_stats["unequal_cnt"] == 1
    assert list_col_stats["match_cnt"] == 2
