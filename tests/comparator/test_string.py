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

import pandas as pd
import polars as pl
from datacompy.comparator.string import (
    PandasStringComparator,
    PolarsStringComparator,
    pandas_compare_string_and_date_columns,
)


# tests for PolarsStringComparator
def test_polars_string_comparator_exact_match():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c"])
    col2 = pl.Series(["a", "b", "c"])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_string_comparator_case_space_insensitivity():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=True, ignore_space=True)
    assert result.to_list() == [True, True, True]

    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=True, ignore_space=False)
    assert result.to_list() == [True, False, False]

    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=False, ignore_space=True)
    assert result.to_list() == [False, True, False]

    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=False, ignore_space=False)
    assert result.to_list() == [False, False, False]


def test_polars_string_comparator_none_handling():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", None])
    col2 = pl.Series(["a", "b", None])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_string_comparator_mismatch():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c"])
    col2 = pl.Series(["a", "b", "d"])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, False]


def test_polars_string_comparator_error_handling():
    comparator = PolarsStringComparator()
    col1 = pl.Series([1, 2, 3])  # Invalid type for string comparison
    col2 = pl.Series([4, 5, 6])  # Invalid type for string comparison
    result = comparator.compare(col1, col2)
    assert result is None

    # different lengths
    col1 = pl.Series(["x", "y", "z"])
    col2 = pl.Series(["x", "y", "z", "c"])
    result = comparator.compare(col1, col2)
    assert result is None


# tests for PandasStringComparator
def test_pandas_string_comparator_exact_match():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c"])
    col2 = pd.Series(["a", "b", "c"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_string_comparator_case_space_insensitivity():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=True, ignore_space=True)
    assert result.tolist() == [True, True, True]

    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=True, ignore_space=False)
    assert result.tolist() == [True, False, False]

    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=False, ignore_space=True)
    assert result.tolist() == [False, True, False]

    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=False, ignore_space=False)
    assert result.tolist() == [False, False, False]


def test_pandas_string_comparator_nan_handling():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", float("nan")])
    col2 = pd.Series(["a", "b", float("nan")])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_string_comparator_mismatch():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c"])
    col2 = pd.Series(["a", "b", "d"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, False]


def test_pandas_string_comparator_error_handling():
    comparator = PandasStringComparator()
    col1 = pd.Series([1, 2, 3])  # Invalid type for string comparison
    col2 = pd.Series([4, 5, 6])  # Invalid type for string comparison
    result = comparator.compare(col1, col2)
    assert result is None

    # different lengths
    col1 = pd.Series(["x", "y", "z"])
    col2 = pd.Series(["x", "y", "z", "c"])
    result = comparator.compare(col1, col2)
    assert result is None


def test_pandas_compare_string_and_date_columns():
    # Test matching string and date columns
    str_col = pd.Series(["2023-01-01", "2023-02-01", "2023-03-01"])
    date_col = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    result = pandas_compare_string_and_date_columns(str_col, date_col)
    assert result.tolist() == [True, True, True]

    # Test mismatched values
    str_col = pd.Series(["2023-01-01", "2023-02-02", "2023-03-01"])
    date_col = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    result = pandas_compare_string_and_date_columns(str_col, date_col)
    assert result.tolist() == [True, False, True]

    # Test null handling
    str_col = pd.Series(["2023-01-01", None, "2023-03-01"])
    date_col = pd.to_datetime(["2023-01-01", pd.NaT, "2023-03-01"])
    result = pandas_compare_string_and_date_columns(str_col, date_col)
    assert result.tolist() == [True, True, True]

    # Test mixed format dates
    str_col = pd.Series(["Jan 1 2023", "02-01-2023", "March 1, 2023"])
    date_col = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    result = pandas_compare_string_and_date_columns(str_col, date_col)
    assert result.tolist() == [True, True, True]

    # Test invalid date strings
    str_col = pd.Series(["not a date", "also not a date", "2023-03-01"])
    date_col = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    result = pandas_compare_string_and_date_columns(str_col, date_col)
    assert result.tolist() == [False, False, True]
