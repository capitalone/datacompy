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
    comparator = PolarsNumericComparator(rtol=1e-3, atol=1e-3)
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.001, 2.002, 3.003])
    result = comparator.compare(col1, col2)
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


# tests for PandasNumericComparator
def test_pandas_numeric_comparator_exact_match():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.0, 2.0, 3.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_approximate_match():
    comparator = PandasNumericComparator(rtol=1e-3, atol=1e-3)
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.001, 2.002, 3.003])
    result = comparator.compare(col1, col2)
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
