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

"""Regression tests for Pandas Boolean comparisons."""

import pandas as pd
from datacompy.comparator import PandasBooleanComparator
from datacompy.pandas import PandasCompare, columns_equal
from pandas.testing import assert_series_equal


def test_native_boolean_columns_equal():
    left = pd.Series([True, False, True, False], dtype=bool)
    right = pd.Series([True, False, False, True], dtype=bool)

    expected = pd.Series([True, True, False, False], dtype=bool)

    assert_series_equal(columns_equal(left, right), expected)


def test_nullable_boolean_columns_equal_with_nulls():
    left = pd.Series([True, False, pd.NA, pd.NA], dtype="boolean")
    right = pd.Series([True, True, pd.NA, False], dtype="boolean")

    expected = pd.Series([True, False, True, False], dtype=bool)

    assert_series_equal(columns_equal(left, right), expected)


def test_all_null_nullable_boolean_columns_equal():
    left = pd.Series([pd.NA, pd.NA], dtype="boolean")
    right = pd.Series([pd.NA, pd.NA], dtype="boolean")

    expected = pd.Series([True, True], dtype=bool)

    assert_series_equal(columns_equal(left, right), expected)


def test_native_and_nullable_boolean_columns_equal():
    left = pd.Series([True, False], dtype=bool)
    right = pd.Series([True, False], dtype="boolean")

    expected = pd.Series([True, True], dtype=bool)

    assert_series_equal(columns_equal(left, right), expected)


def test_boolean_comparator_claims_cross_type_comparisons():
    boolean = pd.Series([True, False, True, False], dtype=bool)
    integer = pd.Series([1, 0, 0, 1], dtype=int)

    expected = pd.Series([True, True, False, False], dtype=bool)

    direct_result = PandasBooleanComparator().compare(boolean, integer)

    assert direct_result is not None
    assert_series_equal(direct_result, expected)
    assert_series_equal(columns_equal(boolean, integer), expected)
    assert_series_equal(columns_equal(integer, boolean), expected)


def test_boolean_comparator_ignores_non_boolean_columns():
    left = pd.Series([1, 2], dtype=int)
    right = pd.Series([1, 2], dtype=int)

    assert PandasBooleanComparator().compare(left, right) is None


def test_pandas_compare_matches_identical_boolean_values():
    left = pd.DataFrame({"id": [1, 2], "flag": [True, False]})
    right = pd.DataFrame({"id": [1, 2], "flag": [True, False]})

    comparison = PandasCompare(left, right, join_columns="id")

    assert comparison.matches()


def test_pandas_compare_detects_boolean_mismatch():
    left = pd.DataFrame({"id": [1, 2], "flag": [True, False]})
    right = pd.DataFrame({"id": [1, 2], "flag": [False, False]})

    comparison = PandasCompare(left, right, join_columns="id")

    assert not comparison.matches()
    assert comparison.intersect_rows["flag_match"].tolist() == [False, True]
