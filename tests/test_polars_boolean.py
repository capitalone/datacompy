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

"""Regression tests for Polars Boolean comparisons."""

import polars as pl
from datacompy.comparator import PolarsBooleanComparator
from datacompy.polars import PolarsCompare, columns_equal
from polars.testing import assert_series_equal


def test_polars_boolean_columns_equal_with_nulls():
    left = pl.Series("left", [True, False, None, None], dtype=pl.Boolean)
    right = pl.Series("right", [True, True, None, False], dtype=pl.Boolean)
    expected = pl.Series([True, False, True, False])

    assert_series_equal(columns_equal(left, right), expected, check_names=False)


def test_polars_boolean_numeric_cross_type_in_both_directions():
    boolean = pl.Series("boolean", [True, False, True, False], dtype=pl.Boolean)
    integer = pl.Series("integer", [1, 0, 0, 1], dtype=pl.Int64)
    expected = pl.Series([True, True, False, False])

    assert_series_equal(columns_equal(boolean, integer), expected, check_names=False)
    assert_series_equal(columns_equal(integer, boolean), expected, check_names=False)


def test_polars_boolean_comparator_ignores_non_boolean_columns():
    left = pl.Series([1, 2], dtype=pl.Int64)
    right = pl.Series([1, 2], dtype=pl.Int64)

    assert PolarsBooleanComparator().compare(left, right) is None


def test_polars_compare_matches_identical_boolean_values():
    left = pl.DataFrame({"id": [1, 2], "flag": [True, False]})
    right = pl.DataFrame({"id": [1, 2], "flag": [True, False]})

    assert PolarsCompare(left, right, join_columns="id").matches()
