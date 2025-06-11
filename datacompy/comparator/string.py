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

import pandas as pd
import polars as pl

from datacompy.comparator.base import BaseStringComparator
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


class PolarsStringComparator(BaseStringComparator):
    """Comparator for string columns in Polars.

    Parameters
    ----------
    ignore_space : bool
        Whether to ignore leading and trailing whitespace when comparing strings.
    ignore_case : bool
        Whether to ignore case when comparing strings.
    """

    def compare(self, col1: pl.Series, col2: pl.Series) -> pl.Series:
        """Compare two Polars Series column-wise, taking into account optional normalization for spaces and case sensitivity.

        Parameters
        ----------
        col1 : pl.Series
            The first Polars Series to compare.
        col2 : pl.Series
            The second Polars Series to compare.

        Returns
        -------
        pl.Series
            A Polars Series of boolean values where each element indicates
            whether the corresponding elements in `col1` and `col2` are equal.
            Handles missing values by treating nulls as equal.

        Raises
        ------
        Exception
            If the comparison fails due to incompatible types or other issues,
            attempts to cast both columns to strings for comparison. If this
            also fails, returns a Series of `False` values with the same length
            as `col1`.
        """
        if (col1.shape == col2.shape) and (
            str(col1.dtype.base_type()) in POLARS_STRING_TYPE
            and str(col2.dtype.base_type()) in POLARS_STRING_TYPE
        ):
            col1 = polars_normalize_string_column(
                col1, self.ignore_space, self.ignore_case
            )
            col2 = polars_normalize_string_column(
                col2, self.ignore_space, self.ignore_case
            )
            try:
                return pl.Series(
                    (col1.eq_missing(col2)) | (col1.is_null() & col2.is_null())
                )
            except Exception:
                try:
                    return pl.Series(col1.cast(pl.String) == col2.cast(pl.String))
                except Exception:
                    return pl.Series([False] * col1.shape[0])
        else:
            return pl.Series([False] * col1.shape[0])


class PandasStringComparator(BaseStringComparator):
    """Comparator for string columns in Pandas.

    Parameters
    ----------
    ignore_space : bool
        Whether to ignore leading and trailing whitespace when comparing strings.
    ignore_case : bool
        Whether to ignore case when comparing strings.
    """

    def compare(self, col1: pd.Series, col2: pd.Series) -> pd.Series:
        """Compare two Pandas Series column-wise, taking into account optional normalization for spaces and case sensitivity.

        Parameters
        ----------
        col1 : pd.Series
            The first Pandas Series to compare.
        col2 : pd.Series
            The second Pandas Series to compare.

        Returns
        -------
        pd.Series
            A Pandas Series of boolean values where each element indicates
            whether the corresponding elements in `col1` and `col2` are equal.
            Handles missing values by treating nulls as equal.

        Raises
        ------
        Exception
            If the comparison fails due to incompatible types or other issues,
            attempts to cast both columns to strings for comparison. If this
            also fails, returns a Series of `False` values with the same length
            as `col1`.
        """
        if (col1.shape == col2.shape) and (
            (pd.api.types.is_string_dtype(col1) and pd.api.types.is_string_dtype(col2))
            or (  # infer_dtype is also a string
                pd.api.types.infer_dtype(col1) == "string"
                and pd.api.types.infer_dtype(col2) == "string"
            )
        ):
            col1 = pandas_normalize_string_column(
                col1, self.ignore_space, self.ignore_case
            )
            col2 = pandas_normalize_string_column(
                col2, self.ignore_space, self.ignore_case
            )
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
        else:
            return pd.Series([False] * col1.shape[0])


class SparkStringComparator(BaseStringComparator):
    """Comparator for string columns in PySpark.

    Parameters
    ----------
    ignore_space : bool
        Whether to ignore leading and trailing whitespace when comparing strings.
    ignore_case : bool
        Whether to ignore case when comparing strings.
    """

    def compare(
        self, dataframe: ps.DataFrame, col1: str, col2: str, col_match: str
    ) -> ps.DataFrame:
        """Compare two columns in a PySpark DataFrame for string equality.

        Parameters
        ----------
        dataframe : pyspark.sql.DataFrame
            The PySpark DataFrame containing the columns to compare.
        col1 : str
            The name of the first column to compare.
        col2 : str
            The name of the second column to compare.
        col_match : str
            The name of the output column that will store the comparison results.

        Returns
        -------
        pyspark.sql.DataFrame
            The DataFrame with an additional column containing the comparison results.

        Raises
        ------
        Exception
            If the comparison fails due to incompatible types or other issues,
            returns a column of `False` values.
        """
        try:
            col1 = spark_normalize_string_column(
                psf.col(col1), self.ignore_space, self.ignore_case
            )
            col2 = spark_normalize_string_column(
                psf.col(col2), self.ignore_space, self.ignore_case
            )

            return dataframe.withColumn(
                col_match,
                psf.when(col1.eqNullSafe(col2), psf.lit(True)).otherwise(
                    psf.lit(False)
                ),
            )
        except Exception:
            return dataframe.withColumn(col_match, psf.lit(False))


class SnowflakeStringComparator(BaseStringComparator):
    """Comparator for string columns in Snowflake.

    Parameters
    ----------
    ignore_space : bool
        Whether to ignore leading and trailing whitespace when comparing strings.
    ignore_case : bool
        Whether to ignore case when comparing strings.
    """

    def compare(
        self, dataframe: sp.DataFrame, col1: str, col2: str, col_match: str
    ) -> sp.DataFrame:
        """Compare two columns in a Snowflake DataFrame for string equality.

        Parameters
        ----------
        dataframe : snowflake.snowpark.DataFrame
            The Snowflake DataFrame containing the columns to compare.
        col1 : str
            The name of the first column to compare.
        col2 : str
            The name of the second column to compare.
        col_match : str
            The name of the output column that will store the comparison results.

        Returns
        -------
        snowflake.snowpark.DataFrame
            The DataFrame with an additional column containing the comparison results.

        Raises
        ------
        Exception
            If the comparison fails due to incompatible types or other issues,
            returns a column of `False` values.
        """
        try:
            col1 = snowpark_normalize_string_column(
                spf.col(col1), self.ignore_space, self.ignore_case
            )
            col2 = snowpark_normalize_string_column(
                spf.col(col2), self.ignore_space, self.ignore_case
            )

            return dataframe.withColumn(
                col_match,
                spf.when(col1.eqNullSafe(col2), spf.lit(True)).otherwise(
                    spf.lit(False)
                ),
            )
        except Exception:
            return dataframe.withColumn(col_match, spf.lit(False))


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
