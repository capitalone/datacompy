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

import numpy as np
import pandas as pd
import polars as pl
from datacompy.comparator.array import (
    PandasArrayLikeComparator,
    PolarsArrayLikeComparator,
)


# Pandas
def test_pandas_compare_equal_arrays():
    # Setup
    col1 = pd.Series([np.array([1, 2]), np.array([3, 4])])
    col2 = pd.Series([np.array([1, 2]), np.array([3, 4])])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pd.Series)
    assert result.all()


def test_pandas_compare_equal_lists():
    # Setup
    col1 = pd.Series([[1, 2], [3, 4]])
    col2 = pd.Series([[1, 2], [3, 4]])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pd.Series)
    assert result.all()


def test_pandas_compare_unequal_arrays():
    # Setup
    col1 = pd.Series([np.array([1, 2]), np.array([3, 4])])
    col2 = pd.Series([np.array([1, 2]), np.array([3, 5])])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pd.Series)
    assert not result.all()
    assert result.tolist() == [True, False]


def test_pandas_compare_unequal_lists():
    # Setup
    col1 = pd.Series([[1, 2], [3, 4]])
    col2 = pd.Series([[1, 2], [3, 5]])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pd.Series)
    assert not result.all()
    assert result.tolist() == [True, False]


def test_pandas_compare_with_nans():
    # Setup
    col1 = pd.Series([np.array([1, np.nan]), np.array([3, 4])])
    col2 = pd.Series([np.array([1, np.nan]), np.array([3, 4])])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pd.Series)
    assert result.all()


def test_pandas_compare_different_shapes():
    # Setup
    col1 = pd.Series([np.array([1, 2]), np.array([3, 4]), np.array([3, 4])])
    col2 = pd.Series([np.array([1, 2, 3]), np.array([3, 4, 5])])
    comparator = PandasArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert result is None


def test_pandas_compare_non_array_like():
    # integers
    col1 = pd.Series([1, 2])
    col2 = pd.Series([1, 2])
    comparator = PandasArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None

    # floats
    col1 = pd.Series([1.0, 2.0])
    col2 = pd.Series([1.0, 2.0])
    comparator = PandasArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None

    # dict
    col1 = pd.Series([{"a": 1}, {"b": 2}])
    col2 = pd.Series([{"a": 1}, {"b": 2}])
    comparator = PandasArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None


# Polars
def test_polars_compare_equal_arrays():
    # Setup
    col1 = pl.Series([[1, 2], [3, 4]])
    col2 = pl.Series([[1, 2], [3, 6]])
    comparator = PolarsArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pl.Series)
    assert result.to_list() == [True, False]


def test_polars_compare_unequal_arrays():
    # Setup
    col1 = pl.Series([[1, 2], [3, 4]])
    col2 = pl.Series([[1, 2], [3, 5]])
    comparator = PolarsArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pl.Series)
    assert result.to_list() == [True, False]


def test_polars_compare_with_nulls():
    # Setup
    col1 = pl.Series([[1, None], [3, 4]])
    col2 = pl.Series([[1, None], [3, 4]])
    comparator = PolarsArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert isinstance(result, pl.Series)
    assert result.to_list() == [True, True]


def test_polars_compare_different_shapes():
    # Setup
    col1 = pl.Series([[1, 2], [3, 4], [5, 6]])
    col2 = pl.Series([[1, 2], [3, 4]])
    comparator = PolarsArrayLikeComparator()

    # Execute
    result = comparator.compare(col1, col2)

    # Assert
    assert result is None


def test_polars_compare_non_array():
    # integers
    col1 = pl.Series([1, 2, 3])
    col2 = pl.Series([1, 2, 3])
    comparator = PolarsArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None

    # floats
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.0, 2.0, 3.0])
    comparator = PolarsArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None

    # dicts
    col1 = pl.Series([{"a": 1}, {"b": 2}])
    col2 = pl.Series([{"a": 1}, {"b": 2}])
    comparator = PolarsArrayLikeComparator()
    result = comparator.compare(col1, col2)
    assert result is None
