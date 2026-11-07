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

from decimal import Decimal

import pandas as pd
import polars as pl
import pytest
from datacompy.comparator.numeric import (
    PandasNumericComparator,
    PolarsNumericComparator,
    _polars_is_close,
)

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize(
    ("col1", "col2", "rtol", "atol", "expected"),
    [
        (
            [1.0, 1.0, 0.0, 100.0, 100.0],
            [1.0 + 1e-9, 1.1, 1e-9, 100.001, 101.0],
            1e-5,
            1e-8,
            [True, False, True, True, False],
        ),
        (
            [NAN, NAN, 1.0, INF, INF, -INF, INF, 1e308],
            [NAN, 1.0, NAN, INF, -INF, -INF, 1e308, INF],
            1e-5,
            1e-8,
            [True, False, False, True, False, True, False, False],
        ),
        ([INF, 1.0], [1.0, 1.0], 0.0, 0.0, [False, True]),
        ([None, 1.0, None], [1.0, None, None], 1e-5, 1e-8, [None, None, None]),
        ([1, 2, 3], [1.0, 2.5, 3.00001], 1e-5, 0.0, [True, False, True]),
        ([10, -10], [11, -8], 0.1, 0.0, [True, False]),
        (
            [Decimal("1.10"), Decimal("2.00")],
            [Decimal("1.10"), Decimal("2.01")],
            0.0,
            0.0,
            [True, False],
        ),
    ],
)
@pytest.mark.parametrize("native", [True, False])
def test_polars_is_close_matches_math_isclose(
    monkeypatch, native, col1, col2, rtol, atol, expected
):
    if native and not hasattr(pl.Series, "is_close"):
        pytest.skip("native is_close requires polars>=1.32")
    if not native:
        monkeypatch.delattr(pl.Series, "is_close", raising=False)
    result = _polars_is_close(pl.Series("x", col1), pl.Series("y", col2), rtol, atol)
    assert result.dtype == pl.Boolean
    assert result.to_list() == expected


def test_polars_numeric_comparator_fallback(monkeypatch):
    monkeypatch.delattr(pl.Series, "is_close", raising=False)
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, NAN, None, None, 2.0, 1])
    col2 = pl.Series([1.0 + 1e-9, NAN, None, 1.0, 2.5, 1])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True, False, False, True]


# tests for PolarsNumericComparator
def test_polars_numeric_comparator_exact_match():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.0, 2.0, 3.0])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_approximate_match():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.001, 2.002, 3.003])
    result = comparator.compare(col1, col2, rtol=1e-3, atol=1e-3)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_type_casting():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1, 2, 3])  # Integer type
    col2 = pl.Series([1.0, 2.0, 3.0])  # Float type
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_nan_handling():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, float("nan"), 3.0])
    col2 = pl.Series([1.0, float("nan"), 3.0])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_mismatch():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.0, 2.5, 3.0])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, False, True]


def test_polars_numeric_comparator_error_handling():
    comparator = PolarsNumericComparator()
    col1 = pl.Series(["a", "b", "c"])  # Invalid type for numeric comparison
    col2 = pl.Series(["x", "y", "z"])  # Invalid type for numeric comparison
    result = comparator.compare(col1, col2)
    assert result is None

    # different lengths
    col1 = pl.Series([1, 2, 3])
    col2 = pl.Series([1, 2, 3, 4])
    result = comparator.compare(col1, col2)
    assert result is None


def test_polars_numeric_comparator_large_integers_exact():
    # Integers above 2**53 are not distinct as floats, so without a tolerance they must be
    # compared exactly
    comparator = PolarsNumericComparator()
    col1 = pl.Series([2**60, 2**60 + 1, 5, None, None])
    col2 = pl.Series([2**60 + 1, 2**60 + 1, 5, None, 1])
    result = comparator.compare(col1, col2, rtol=0, atol=0)
    assert result.to_list() == [False, True, True, True, False]


def test_polars_numeric_comparator_large_integers_with_tolerance():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([2**60])
    col2 = pl.Series([2**60 + 1])
    result = comparator.compare(col1, col2, rtol=1e-5, atol=0)
    assert result.to_list() == [True]


def test_polars_numeric_comparator_uint64_and_mixed_widths():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([2**64 - 1, 2**31 - 1], dtype=pl.UInt64)
    col2 = pl.Series([2**64 - 2, 2**31 - 1], dtype=pl.UInt64)
    result = comparator.compare(col1, col2, rtol=0, atol=0)
    assert result.to_list() == [False, True]
    col1 = pl.Series([2**31 - 1], dtype=pl.Int32)
    col2 = pl.Series([2**31 - 1], dtype=pl.Int64)
    result = comparator.compare(col1, col2, rtol=0, atol=0)
    assert result.to_list() == [True]


# tests for PandasNumericComparator
def test_pandas_numeric_comparator_exact_match():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.0, 2.0, 3.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_approximate_match():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.001, 2.002, 3.003])
    result = comparator.compare(col1, col2, rtol=1e-3, atol=1e-3)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_type_casting():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1, 2, 3])  # Integer type
    col2 = pd.Series([1.0, 2.0, 3.0])  # Float type
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_nan_handling():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, float("nan"), 3.0])
    col2 = pd.Series([1.0, float("nan"), 3.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_mismatch():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.0, 2.5, 3.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, False, True]


def test_pandas_numeric_comparator_error_handling():
    comparator = PandasNumericComparator()
    col1 = pd.Series(["a", "b", "c"])  # Invalid type for numeric comparison
    col2 = pd.Series(["x", "y", "z"])  # Invalid type for numeric comparison
    result = comparator.compare(col1, col2)
    assert result is None

    # different lengths
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.0, 2.5, 3.0, 4.0])
    result = comparator.compare(col1, col2)
    assert result is None


def test_pandas_numeric_comparator_large_integers_exact():
    # Integers above 2**53 are not distinct as floats, so without a tolerance they must be
    # compared exactly
    comparator = PandasNumericComparator()
    col1 = pd.Series([2**60, 2**60 + 1, 5])
    col2 = pd.Series([2**60 + 1, 2**60 + 1, 5])
    result = comparator.compare(col1, col2, rtol=0, atol=0)
    assert result.tolist() == [False, True, True]

    col1 = pd.Series([2**60, None, None, 3], dtype="Int64")
    col2 = pd.Series([2**60 + 1, None, 1, 3], dtype="Int64")
    result = comparator.compare(col1, col2, rtol=0, atol=0)
    assert result.tolist() == [False, True, False, True]


def test_pandas_numeric_comparator_large_integers_with_tolerance():
    comparator = PandasNumericComparator()
    col1 = pd.Series([2**60])
    col2 = pd.Series([2**60 + 1])
    result = comparator.compare(col1, col2, rtol=1e-5, atol=0)
    assert result.tolist() == [True]


def test_pandas_numeric_comparator_large_integers_uint64():
    # UInt64 values near 2**64 are not distinct as floats either
    comparator = PandasNumericComparator()
    col1 = pd.Series([2**64 - 2, 5], dtype="UInt64")
    col2 = pd.Series([2**64 - 3, 5], dtype="UInt64")
    result = comparator.compare(col1, col2, rtol=0, atol=0)
    assert result.tolist() == [False, True]
