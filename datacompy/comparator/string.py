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

"""String Comparator Class."""

import logging
from functools import singledispatch
from typing import Any

import pandas as pd
import polars as pl

from datacompy.comparator.base import BaseComparator
from datacompy.polars import STRING_TYPE as POLARS_STRING_TYPE

DEFAULT_VALUE = "DATACOMPY_NULL"
LOG = logging.getLogger(__name__)

try:
    import pyspark.sql as ps
    import pyspark.sql.functions as psf
except ImportError:
    LOG.warning(
        "Please note that you are missing the optional dependency: spark. "
        "If you need to use this functionality it must be installed."
    )

try:
    import snowflake.snowpark as sp
    import snowflake.snowpark.functions as spf
except ImportError:
    LOG.warning(
        "Please note that you are missing the optional dependency: snowflake. "
        "If you need to use this functionality it must be installed."
    )


# concrete implementation of the BaseComparator for string columns
class StringComparator(BaseComparator):
    """Comparator for string columns.

    Parameters
    ----------
    ignore_space : bool
        Whether to ignore leading and trailing whitespace when comparing strings.
    ignore_case : bool
        Whether to ignore case when comparing strings.
    """

    def __init__(self, ignore_space: bool = True, ignore_case: bool = True):
        self.ignore_space = ignore_space
        self.ignore_case = ignore_case

    # abstract method which uses singledispatching
    @singledispatch
    def compare(self, col1: Any, col2: Any) -> Any:
        """Compare two columns to determine if they are equal.

        Parameters
        ----------
        col1 : Any
            The first column to compare.
        col2 : Any
            The second column to compare.

        Returns
        -------
        Any

        Raises
        ------
        NotImplementedError
            If the column types are unsupported.
        """
        raise NotImplementedError("Unsupported column types")

    # pandas implementation of compare
    @compare.register
    def _(self, col1: pd.Series, col2: pd.Series) -> pd.Series:
        col1 = pandas_normalize_string_column(col1, self.ignore_space, self.ignore_case)
        col2 = pandas_normalize_string_column(col2, self.ignore_space, self.ignore_case)
        try:
            return pd.Series(
                (col1.fillna(DEFAULT_VALUE) == col2.fillna(DEFAULT_VALUE))
                | (col1.isnull() & col2.isnull())
            )
        except Exception:
            try:
                return pd.Series(col1.astype(str) == col2.astype(str))
            except Exception:
                return pd.Series(False * col1.shape[0])

    # polars implementation of compare
    @compare.register
    def _(self, col1: pl.Series, col2: pl.Series) -> pl.Series:
        col1 = polars_normalize_string_column(col1, self.ignore_space, self.ignore_case)
        col2 = polars_normalize_string_column(col2, self.ignore_space, self.ignore_case)
        try:
            return pl.Series(
                (col1.eq_missing(col2)) | (col1.is_null() & col2.is_null())
            )
        except Exception:
            try:
                return pl.Series(col1.cast(pl.String) == col2.cast(pl.String))
            except Exception:
                return pl.Series(False * col1.shape[0])

    # pyspark implementation of compare
    @compare.register
    def _(self, col1: ps.Column, col2: ps.Column) -> ps.Column:
        return (
            psf.when(psf.col(col1).eqNullSafe(psf.col(col2)), psf.lit(True)).otherwise(
                psf.lit(False)
            ),
        )

    # snowpark implementation of compare
    @compare.register
    def _(self, col1: sp.Column, col2: sp.Column) -> sp.Column:
        return (
            spf.when(spf.col(col1).eqNullSafe(spf.col(col2)), spf.lit(True)).otherwise(
                spf.lit(False)
            ),
        )


def pandas_normalize_string_column(
    column: pd.Series, ignore_spaces: bool, ignore_case: bool
) -> pd.Series:
    """Normalize a string column by converting to upper case and stripping whitespace.

    Parameters
    ----------
    column : pd.Series
        The column to normalize
    ignore_spaces : bool
        Whether to ignore spaces when normalizing
    ignore_case : bool
        Whether to ignore case when normalizing

    Returns
    -------
    pd.Series
        The normalized column

    Notes
    -----
    Will not operate on categorical columns.
    """
    if (column.dtype.kind == "O" and pd.api.types.infer_dtype(column) == "string") or (
        pd.api.types.is_string_dtype(column)
        and not isinstance(column.dtype, pd.CategoricalDtype)
    ):
        column = column.str.strip() if ignore_spaces else column
        column = column.str.upper() if ignore_case else column
    return column


def polars_normalize_string_column(
    column: pl.Series, ignore_spaces: bool, ignore_case: bool
) -> pl.Series:
    """Normalize a string column by converting to upper case and stripping whitespace.

    Parameters
    ----------
    column : pl.Series
        The column to normalize
    ignore_spaces : bool
        Whether to ignore spaces when normalizing
    ignore_case : bool
        Whether to ignore case when normalizing

    Returns
    -------
    pl.Series
        The normalized column

    Notes
    -----
    Will not operate on categorical columns.
    """
    if str(column.dtype.base_type()) in POLARS_STRING_TYPE:
        if ignore_spaces:
            column = column.str.strip_chars()
        if ignore_case:
            column = column.str.to_uppercase()
    return column


def spark_normalize_string_column(
    column: ps.Column, ignore_spaces: bool, ignore_case: bool
) -> ps.Column:
    """Normalize a string column by converting to upper case and stripping whitespace.

    Parameters
    ----------
    column : ps.Column
        The column to normalize
    ignore_spaces : bool
        Whether to ignore spaces when normalizing
    ignore_case : bool
        Whether to ignore case when normalizing

    Returns
    -------
    ps.Column
        The normalized column
    """
    if ignore_spaces:
        column = psf.trim(column)
    if ignore_case:
        column = psf.upper(column)
    return column


def snowpark_normalize_string_column(
    column: sp.Column, ignore_spaces: bool, ignore_case: bool
) -> sp.Column:
    """Normalize a string column by converting to upper case and stripping whitespace.

    Parameters
    ----------
    column : sp.Column
        The column to normalize
    ignore_spaces : bool
        Whether to ignore spaces when normalizing
    ignore_case : bool
        Whether to ignore case when normalizing

    Returns
    -------
    sp.Column
        The normalized column
    """
    if ignore_spaces:
        column = spf.trim(column)
    if ignore_case:
        column = spf.upper(column)
    return column
