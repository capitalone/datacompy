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

"""Regression tests for Pandas and Polars Boolean comparisons."""

import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from datacompy.comparator import PandasBooleanComparator, PolarsBooleanComparator
from datacompy.pandas import PandasCompare
from datacompy.pandas import columns_equal as pandas_columns_equal
from datacompy.polars import PolarsCompare
from datacompy.polars import columns_equal as polars_columns_equal
from pandas.testing import assert_series_equal

# --------------------------------------------------------------------------
# Pandas
# --------------------------------------------------------------------------


def test_pandas_native_boolean_columns_equal():
    left = pd.Series([True, False, True, False], dtype=bool)
    right = pd.Series([True, False, False, True], dtype=bool)

    expected = pd.Series([True, True, False, False], dtype=bool)

    assert_series_equal(pandas_columns_equal(left, right), expected)


def test_pandas_nullable_boolean_columns_equal_with_nulls():
    left = pd.Series([True, False, pd.NA, pd.NA], dtype="boolean")
    right = pd.Series([True, True, pd.NA, False], dtype="boolean")

    expected = pd.Series([True, False, True, False], dtype=bool)

    assert_series_equal(pandas_columns_equal(left, right), expected)


def test_pandas_all_null_nullable_boolean_columns_equal():
    left = pd.Series([pd.NA, pd.NA], dtype="boolean")
    right = pd.Series([pd.NA, pd.NA], dtype="boolean")

    expected = pd.Series([True, True], dtype=bool)

    assert_series_equal(pandas_columns_equal(left, right), expected)


def test_pandas_native_and_nullable_boolean_columns_equal():
    left = pd.Series([True, False], dtype=bool)
    right = pd.Series([True, False], dtype="boolean")

    expected = pd.Series([True, True], dtype=bool)

    assert_series_equal(pandas_columns_equal(left, right), expected)


def test_pandas_boolean_comparator_claims_cross_type_comparisons():
    boolean = pd.Series([True, False, True, False], dtype=bool)
    integer = pd.Series([1, 0, 0, 1], dtype=int)

    expected = pd.Series([True, True, False, False], dtype=bool)

    direct_result = PandasBooleanComparator().compare(boolean, integer)

    assert direct_result is not None
    assert_series_equal(direct_result, expected)
    assert_series_equal(pandas_columns_equal(boolean, integer), expected)
    assert_series_equal(pandas_columns_equal(integer, boolean), expected)


def test_pandas_boolean_comparator_ignores_non_boolean_columns():
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


def test_pandas_object_dtype_boolean_columns_equal():
    """A Boolean column holding a null is ``object`` dtype, not ``bool``."""
    left = pd.Series([True, False, None])
    right = pd.Series([True, False, None])

    assert left.dtype == object

    expected = pd.Series([True, True, True], dtype=bool)

    assert_series_equal(pandas_columns_equal(left, right), expected)


def test_pandas_object_dtype_boolean_columns_detect_mismatch():
    left = pd.Series([True, False, None, True], dtype=object)
    right = pd.Series([True, True, True, None], dtype=object)

    expected = pd.Series([True, False, False, False], dtype=bool)

    assert_series_equal(pandas_columns_equal(left, right), expected)


def test_pandas_object_dtype_boolean_columns_equal_with_nan():
    left = pd.Series([True, False, np.nan], dtype=object)
    right = pd.Series([True, False, np.nan], dtype=object)

    expected = pd.Series([True, True, True], dtype=bool)

    assert_series_equal(pandas_columns_equal(left, right), expected)


def test_pandas_compare_boolean_with_non_overlapping_join_keys():
    """An outer merge upcasts ``bool`` to ``object``, so detection must survive it."""
    left = pd.DataFrame({"id": [1, 2, 3], "flag": [True, False, True]})
    right = pd.DataFrame({"id": [1, 2, 4], "flag": [True, False, True]})

    comparison = PandasCompare(left, right, join_columns="id")

    # Guard the premise of this test: the merge really does upcast to object.
    assert comparison.intersect_rows["flag_df1"].dtype == object

    assert comparison.intersect_rows["flag_match"].tolist() == [True, True]
    assert comparison.count_matching_rows() == 2


def test_pandas_compare_boolean_mismatch_with_non_overlapping_join_keys():
    left = pd.DataFrame({"id": [1, 2, 3], "flag": [True, False, True]})
    right = pd.DataFrame({"id": [1, 2, 4], "flag": [True, True, True]})

    comparison = PandasCompare(left, right, join_columns="id")

    assert not comparison.matches()
    assert comparison.intersect_rows["flag_match"].tolist() == [True, False]


def test_pandas_compare_boolean_with_nulls_and_non_overlapping_join_keys():
    left = pd.DataFrame({"id": [1, 2, 3], "flag": [True, None, False]})
    right = pd.DataFrame({"id": [1, 2, 4], "flag": [True, None, False]})

    comparison = PandasCompare(left, right, join_columns="id")

    assert comparison.intersect_rows["flag_match"].tolist() == [True, True]


def test_pandas_boolean_comparator_ignores_object_dtype_non_boolean_columns():
    left = pd.Series(["a", "b"], dtype=object)
    right = pd.Series(["a", "b"], dtype=object)

    assert PandasBooleanComparator().compare(left, right) is None


def test_pandas_boolean_comparator_ignores_mismatched_shapes():
    left = pd.Series([True, False], dtype=bool)
    right = pd.Series([True], dtype=bool)

    assert PandasBooleanComparator().compare(left, right) is None


# --------------------------------------------------------------------------
# Polars
# --------------------------------------------------------------------------


def test_polars_boolean_columns_equal():
    left = pl.Series([True, False, True, False])
    right = pl.Series([True, False, False, True])

    assert polars_columns_equal(left, right).to_list() == [True, True, False, False]


def test_polars_boolean_columns_equal_with_nulls():
    left = pl.Series([True, False, None, None], dtype=pl.Boolean)
    right = pl.Series([True, True, None, False], dtype=pl.Boolean)

    assert polars_columns_equal(left, right).to_list() == [True, False, True, False]


def test_polars_all_null_boolean_columns_equal():
    left = pl.Series([None, None], dtype=pl.Boolean)
    right = pl.Series([None, None], dtype=pl.Boolean)

    assert polars_columns_equal(left, right).to_list() == [True, True]


def test_polars_boolean_comparator_claims_cross_type_comparisons():
    boolean = pl.Series([True, False, True, False])
    integer = pl.Series([1, 0, 0, 1])

    expected = [True, True, False, False]

    direct_result = PolarsBooleanComparator().compare(boolean, integer)

    assert direct_result is not None
    assert direct_result.to_list() == expected
    assert polars_columns_equal(boolean, integer).to_list() == expected
    assert polars_columns_equal(integer, boolean).to_list() == expected


def test_polars_boolean_comparator_ignores_non_boolean_columns():
    left = pl.Series([1, 2])
    right = pl.Series([1, 2])

    assert PolarsBooleanComparator().compare(left, right) is None


def test_polars_boolean_comparator_ignores_mismatched_shapes():
    left = pl.Series([True, False])
    right = pl.Series([True])

    assert PolarsBooleanComparator().compare(left, right) is None


def test_polars_boolean_comparator_falls_through_on_incomparable_dtypes():
    """Polars raises when there is no comparison supertype; do not claim the pair."""
    boolean = pl.Series([True, False])
    dates = pl.Series([datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)])

    assert PolarsBooleanComparator().compare(boolean, dates) is None

    # The pipeline still resolves to "not equal" rather than raising.
    assert polars_columns_equal(boolean, dates).to_list() == [False, False]


@pytest.mark.parametrize(
    "dtype", [pl.String, pl.Categorical, pl.Enum(["true", "false"])]
)
def test_polars_boolean_comparator_declines_boolean_against_text(dtype):
    """Polars parses the string, so 'true' would match but 'True' would not."""
    boolean = pl.Series([True, False])
    text = pl.Series(["true", "false"], dtype=dtype)

    assert PolarsBooleanComparator().compare(boolean, text) is None
    assert PolarsBooleanComparator().compare(text, boolean) is None


def test_polars_boolean_against_text_does_not_match():
    """Pandas and Spark both report no match here; Polars must agree."""
    boolean = pl.Series([True, False])

    for text in (pl.Series(["true", "false"]), pl.Series(["True", "False"])):
        assert polars_columns_equal(boolean, text).to_list() == [False, False]
        assert polars_columns_equal(text, boolean).to_list() == [False, False]


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
