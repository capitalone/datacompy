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

import numpy as np
import pandas as pd
import polars as pl
from datacompy.comparator.numeric import (
    PandasNumericComparator,
    PolarsNumericComparator,
)


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


def test_pandas_numeric_comparator_symmetric_tolerance():
    # Asymmetric np.isclose(90, 100) is True while np.isclose(100, 90) is False
    # with rtol=0.1, atol=0. math.isclose is True in both directions.
    comparator = PandasNumericComparator()
    forward = comparator.compare(
        pd.Series([90.0]), pd.Series([100.0]), rtol=0.1, atol=0
    )
    reverse = comparator.compare(
        pd.Series([100.0]), pd.Series([90.0]), rtol=0.1, atol=0
    )
    assert forward.tolist() == [True]
    assert reverse.tolist() == [True]


def test_pandas_numeric_comparator_max_not_sum_tolerances():
    # NumPy allows atol + rtol*|b|; math.isclose uses max(...). With both set,
    # |10-11|=1 is within NumPy's 1.05 but outside math's 0.55.
    comparator = PandasNumericComparator()
    result = comparator.compare(
        pd.Series([10.0]), pd.Series([11.0]), rtol=0.05, atol=0.5
    )
    assert result.tolist() == [False]
    assert np.isclose(10.0, 11.0, rtol=0.05, atol=0.5)
