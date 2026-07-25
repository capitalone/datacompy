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

"""Regression tests for index alignment in the Pandas ``columns_equal``.

``columns_equal`` compares positionally, it stamps ``col_1``'s index onto the
result, but most Pandas operations align on labels. Columns whose indexes
disagree therefore used to be compared row-by-label, which either raised
(``Series.eq`` reindexing to the union) or silently answered a different
question. These tests pin the positional contract for every comparator.
"""

import pandas as pd
import pytest
from datacompy.pandas import columns_equal

# Same length, no labels in common. Values are identical position-by-position,
# so the correct answer is True everywhere.
LEFT_LABELS = [0, 1]
RIGHT_LABELS = [2, 3]


@pytest.mark.parametrize(
    ("name", "left", "right"),
    [
        ("numeric", [1, 2], [1, 2]),
        ("boolean", [True, False], [True, False]),
        ("string", ["a", "b"], ["a", "b"]),
        ("array_like", [[1], [2]], [[1], [2]]),
    ],
)
def test_disjoint_labels_compare_positionally(name, left, right):
    col_1 = pd.Series(left, index=LEFT_LABELS)
    col_2 = pd.Series(right, index=RIGHT_LABELS)

    result = columns_equal(col_1, col_2)

    assert result.dtype == bool
    assert result.tolist() == [True, True]
    assert result.index.tolist() == LEFT_LABELS


@pytest.mark.parametrize(
    ("name", "left", "right", "expected"),
    [
        ("numeric", [1, 2], [1, 99], [True, False]),
        ("boolean", [True, False], [True, True], [True, False]),
        ("string", ["a", "b"], ["a", "z"], [True, False]),
    ],
)
def test_disjoint_labels_still_detect_mismatches(name, left, right, expected):
    """Alignment must not paper over real differences."""
    col_1 = pd.Series(left, index=LEFT_LABELS)
    col_2 = pd.Series(right, index=RIGHT_LABELS)

    assert columns_equal(col_1, col_2).tolist() == expected


def test_partially_overlapping_labels():
    col_1 = pd.Series([True, True], index=[0, 1])
    col_2 = pd.Series([True, True], index=[1, 2])

    result = columns_equal(col_1, col_2)

    assert result.tolist() == [True, True]
    assert result.index.tolist() == [0, 1]


def test_reordered_labels_compare_positionally():
    """Same label set in a different order used to silently change the answer."""
    col_1 = pd.Series([True, False, True], index=[0, 1, 2])
    col_2 = pd.Series([False, True, True], index=[1, 0, 2])

    assert columns_equal(col_1, col_2).tolist() == [False, False, True]


def test_reordered_labels_agree_across_dtypes():
    """Boolean and numeric must not disagree on structurally identical data."""
    col_1 = pd.Series([True, False, True], index=[0, 1, 2])
    col_2 = pd.Series([False, True, True], index=[1, 0, 2])

    as_boolean = columns_equal(col_1, col_2)
    as_numeric = columns_equal(col_1.astype(int), col_2.astype(int))

    assert as_boolean.tolist() == as_numeric.tolist()


def test_filtered_column_against_fresh_column():
    """Filtering keeps the original labels -- a realistic source of the gap."""
    source = pd.DataFrame(
        {"keep": [True, False, True, True], "flag": [True, True, True, True]}
    )
    filtered = source[source["keep"]]["flag"]  # labels 0, 2, 3
    fresh = pd.Series([True, True, True])  # labels 0, 1, 2

    result = columns_equal(filtered, fresh)

    assert result.tolist() == [True, True, True]
    assert result.index.tolist() == [0, 2, 3]


def test_nullable_boolean_dtype_survives_alignment():
    col_1 = pd.Series([True, None], index=LEFT_LABELS, dtype="boolean")
    col_2 = pd.Series([True, None], index=RIGHT_LABELS, dtype="boolean")

    assert columns_equal(col_1, col_2).tolist() == [True, True]


def test_datetime_dtype_survives_alignment():
    stamps = pd.to_datetime(["2020-01-01", "2020-01-02"])
    col_1 = pd.Series(stamps, index=LEFT_LABELS)
    col_2 = pd.Series(stamps, index=RIGHT_LABELS)

    assert columns_equal(col_1, col_2).tolist() == [True, True]


def test_matching_labels_are_untouched():
    col_1 = pd.Series([True, False, True])
    col_2 = pd.Series([True, True, True])

    assert columns_equal(col_1, col_2).tolist() == [True, False, True]


def test_duplicate_labels():
    col_1 = pd.Series([True, False], index=[0, 0])
    col_2 = pd.Series([True, False], index=[0, 0])

    assert columns_equal(col_1, col_2).tolist() == [True, True]


def test_different_lengths_fall_through_to_false():
    """Unequal lengths keep the pre-existing all-False behaviour."""
    col_1 = pd.Series([True, True])
    col_2 = pd.Series([True, True, True])

    result = columns_equal(col_1, col_2)

    assert result.tolist() == [False, False]
    assert result.index.tolist() == [0, 1]


def test_caller_series_is_not_mutated():
    """Alignment must not rewrite the index of the caller's column."""
    col_1 = pd.Series([True, True], index=LEFT_LABELS)
    col_2 = pd.Series([True, True], index=RIGHT_LABELS)

    columns_equal(col_1, col_2)

    assert col_1.index.tolist() == LEFT_LABELS
    assert col_2.index.tolist() == RIGHT_LABELS
