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

import datetime

import polars as pl
from datacompy.comparator import PolarsBooleanComparator
from datacompy.polars import PolarsCompare, columns_equal


def test_boolean_columns_equal():
    left = pl.Series([True, False, True, False])
    right = pl.Series([True, False, False, True])

    assert columns_equal(left, right).to_list() == [True, True, False, False]


def test_boolean_columns_equal_with_nulls():
    left = pl.Series([True, False, None, None], dtype=pl.Boolean)
    right = pl.Series([True, True, None, False], dtype=pl.Boolean)

    assert columns_equal(left, right).to_list() == [True, False, True, False]


def test_all_null_boolean_columns_equal():
    left = pl.Series([None, None], dtype=pl.Boolean)
    right = pl.Series([None, None], dtype=pl.Boolean)

    assert columns_equal(left, right).to_list() == [True, True]


def test_boolean_comparator_claims_cross_type_comparisons():
    boolean = pl.Series([True, False, True, False])
    integer = pl.Series([1, 0, 0, 1])

    expected = [True, True, False, False]

    direct_result = PolarsBooleanComparator().compare(boolean, integer)

    assert direct_result is not None
    assert direct_result.to_list() == expected
    assert columns_equal(boolean, integer).to_list() == expected
    assert columns_equal(integer, boolean).to_list() == expected


def test_boolean_comparator_ignores_non_boolean_columns():
    left = pl.Series([1, 2])
    right = pl.Series([1, 2])

    assert PolarsBooleanComparator().compare(left, right) is None


def test_boolean_comparator_ignores_mismatched_shapes():
    left = pl.Series([True, False])
    right = pl.Series([True])

    assert PolarsBooleanComparator().compare(left, right) is None


def test_boolean_comparator_falls_through_on_incomparable_dtypes():
    """Polars raises when there is no comparison supertype; do not claim the pair."""
    boolean = pl.Series([True, False])
    dates = pl.Series([datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)])

    assert PolarsBooleanComparator().compare(boolean, dates) is None

    # The pipeline still resolves to "not equal" rather than raising.
    assert columns_equal(boolean, dates).to_list() == [False, False]


def test_polars_compare_matches_identical_boolean_values():
    left = pl.DataFrame({"id": [1, 2], "flag": [True, False]})
    right = pl.DataFrame({"id": [1, 2], "flag": [True, False]})

    comparison = PolarsCompare(left, right, join_columns="id")

    assert comparison.matches()


def test_polars_compare_detects_boolean_mismatch():
    left = pl.DataFrame({"id": [1, 2], "flag": [True, False]})
    right = pl.DataFrame({"id": [1, 2], "flag": [False, False]})

    comparison = PolarsCompare(left, right, join_columns="id")

    assert not comparison.matches()
    assert comparison.intersect_rows["flag_match"].to_list() == [False, True]


def test_polars_compare_boolean_with_non_overlapping_join_keys():
    left = pl.DataFrame({"id": [1, 2, 3], "flag": [True, False, True]})
    right = pl.DataFrame({"id": [1, 2, 4], "flag": [True, False, True]})

    comparison = PolarsCompare(left, right, join_columns="id")

    assert comparison.intersect_rows["flag_match"].to_list() == [True, True]
    assert comparison.count_matching_rows() == 2


def test_polars_compare_boolean_with_nulls():
    left = pl.DataFrame({"id": [1, 2, 3], "flag": [True, None, False]})
    right = pl.DataFrame({"id": [1, 2, 3], "flag": [True, None, False]})

    comparison = PolarsCompare(left, right, join_columns="id")

    assert comparison.matches()
    assert comparison.intersect_rows["flag_match"].to_list() == [True, True, True]


def test_polars_compare_boolean_null_against_value_is_mismatch():
    left = pl.DataFrame({"id": [1, 2], "flag": [True, None]})
    right = pl.DataFrame({"id": [1, 2], "flag": [True, False]})

    comparison = PolarsCompare(left, right, join_columns="id")

    assert not comparison.matches()
    assert comparison.intersect_rows["flag_match"].to_list() == [True, False]
