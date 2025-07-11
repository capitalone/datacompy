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

"""String / Dates / Mixed Comparator Class."""

import logging
from typing import Any

import pandas as pd
import polars as pl
import pyspark as ps
import pyspark.sql.functions as psf
import snowflake.snowpark as sp
import snowflake.snowpark.functions as spf

from datacompy.comparator.base import BaseStringComparator
from datacompy.comparator.utility import (
    get_snowflake_column_dtypes,
    get_spark_column_dtypes,
)

LOG = logging.getLogger(__name__)

DEFAULT_VALUE = "DATACOMPY_NULL"
POLARS_STRING_TYPE = {"String", "Utf8"}
POLARS_DATE_TYPE = {"Date", "Datetime"}
PYSPARK_STRING_TYPE = {"string"}
SNOWFLAKE_STRING_TYPE = {spf.StringType()}
PANDAS_STRING_TYPE = {"string", "categorical"}
PANDAS_DATE_TYPES = {
    "datetime64",
    "datetime",
    "date",
    "timedelta64",
    "timedelta",
    "time",
    "period",
}


class PolarsStringComparator(BaseStringComparator):
    """Comparator for string / temporal / date columns in Polars.

    Parameters
    ----------
    ignore_space : bool
        Whether to ignore leading and trailing whitespace when comparing strings.
    ignore_case : bool
        Whether to ignore case when comparing strings.
    """

    def compare(self, col1: pl.Series, col2: pl.Series) -> pl.Series | None:
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
        None
            if the columns are not comparable.

        Raises
        ------
        Exception
            If the comparison fails due to incompatible types or other issues,
            attempts to cast both columns to strings for comparison. If this
            also fails, returns a Series of `False` values with the same length
            as `col1`.
        """
        if col1.shape != col2.shape:
            return None

        col1_type = str(col1.dtype.base_type())
        col2_type = str(col2.dtype.base_type())

        # if one is a string and another is a date
        if (col1_type in POLARS_DATE_TYPE and col2_type in POLARS_STRING_TYPE) or (
            col2_type in POLARS_DATE_TYPE and col1_type in POLARS_STRING_TYPE
        ):
            return polars_compare_string_and_date_columns(col1, col2)
        # both are strings or both are temporal or both are categorical
        elif (
            (col1_type in POLARS_STRING_TYPE and col2_type in POLARS_STRING_TYPE)
            or (col1.dtype.is_temporal() and col2.dtype.is_temporal())
            or (col1_type == "Categorical" and col2_type == "Categorical")
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
            return None


class PandasStringComparator(BaseStringComparator):
    """Comparator for string / date / mixed columns in Pandas.

    Parameters
    ----------
    ignore_space : bool
        Whether to ignore leading and trailing whitespace when comparing strings.
    ignore_case : bool
        Whether to ignore case when comparing strings.
    """

    def compare(self, col1: pd.Series, col2: pd.Series) -> pd.Series | None:
        """Compare two Pandas Series column-wise, taking into account optional normalization for spaces and case sensitivity.

        Parameters
        ----------
        col1 : pd.Series
            The first Pandas Series to compare.
        col2 : pd.Series
            The second Pandas Series to compare.

        Returns
        -------
        pd.Series | None
            A Pandas Series of boolean values where each element indicates
            whether the corresponding elements in `col1` and `col2` are equal.
            Handles missing values by treating nulls as equal.
        None
            if the columns are not comparable.

        Note
        ----
        Pandas dataframes allow for mixed typing which is unique and is also handled here.

        Raises
        ------
        Exception
            If the comparison fails due to incompatible types or other issues,
            attempts to cast both columns to strings for comparison. If this
            also fails, returns a Series of `False` values with the same length
            as `col1`.
        """
        # check the shape first and short circuit
        if col1.shape != col2.shape:
            return None

        col1_type = pd.api.types.infer_dtype(col1, skipna=True)
        col2_type = pd.api.types.infer_dtype(col2, skipna=True)

        # if one is a string and another is a date
        if (col1_type in PANDAS_DATE_TYPES and col2_type in PANDAS_STRING_TYPE) or (
            col2_type in PANDAS_DATE_TYPES and col1_type in PANDAS_STRING_TYPE
        ):
            return pandas_compare_string_and_date_columns(col1, col2)
        # if both are strings
        elif col1_type in PANDAS_STRING_TYPE and col2_type in PANDAS_STRING_TYPE:
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
                    return pd.Series(False * col1.index)
        # if both are mixed or dates
        elif (
            pd.api.types.infer_dtype(col1).startswith("mixed")
            and pd.api.types.infer_dtype(col2).startswith("mixed")
        ) or (col1_type in PANDAS_DATE_TYPES and col2_type in PANDAS_DATE_TYPES):
            # Handle mixed type columns by casting to a string and comparing
            try:
                return pd.Series(col1.astype(str) == col2.astype(str))
            except Exception:
                return pd.Series(False * col1.index)
        else:  # if not one of the supported type usecases
            return None


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
        self, dataframe: ps.sql.DataFrame, col1: str, col2: str, col_match: str
    ) -> ps.sql.DataFrame | None:
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
        None
            if the columns are not comparable.

        Raises
        ------
        Exception
            If the comparison fails due to incompatible types or other issues,
            returns a column of `False` values.
        """
        # if col1 and col2 of dataframe are of type string
        base_dtype, compare_dtype = get_spark_column_dtypes(dataframe, col1, col2)
        if (base_dtype in PYSPARK_STRING_TYPE) and (
            compare_dtype in PYSPARK_STRING_TYPE
        ):
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
        else:
            return None


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
    ) -> sp.DataFrame | None:
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

        None
            if the columns are not comparable.

        Raises
        ------
        Exception
            If the comparison fails due to incompatible types or other issues,
            returns a column of `False` values.
        """
        # if col1 and col2 of dataframe are of type string
        base_dtype, compare_dtype = get_snowflake_column_dtypes(dataframe, col1, col2)
        if (
            base_dtype in SNOWFLAKE_STRING_TYPE
            and compare_dtype in SNOWFLAKE_STRING_TYPE
        ):
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
        else:
            return None


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
    column: ps.sql.Column, ignore_spaces: bool, ignore_case: bool
) -> ps.sql.Column:
    """Normalize a string column by converting to upper case and stripping whitespace.

    Parameters
    ----------
    column : pyspark.sql.Column
        The column to normalize
    ignore_spaces : bool
        Whether to ignore spaces when normalizing
    ignore_case : bool
        Whether to ignore case when normalizing

    Returns
    -------
    pyspark.sql.Column
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
    column : snowflake.snowpark.Column
        The column to normalize
    ignore_spaces : bool
        Whether to ignore spaces when normalizing
    ignore_case : bool
        Whether to ignore case when normalizing

    Returns
    -------
    snowflake.snowpark.Column
        The normalized column
    """
    if ignore_spaces:
        column = spf.trim(column)
    if ignore_case:
        column = spf.upper(column)
    return column


def pandas_compare_string_and_date_columns(
    col_1: "pd.Series[Any]", col_2: "pd.Series[Any]"
) -> "pd.Series[bool]":
    """Compare a string column and date column, value-wise.

    This tries to:
    - convert a string column to a date column and compare
    - try with format=mixed
    - finally cast as strings and then compare

    Parameters
    ----------
    col_1 : Pandas.Series
        The first column to look at
    col_2 : Pandas.Series
        The second column

    Returns
    -------
    pandas.Series
        A series of Boolean values.  True == the values match, False == the
        values don't match.
    """
    if col_1.dtype.kind == "O":
        obj_column = col_1
        date_column = col_2
    else:
        obj_column = col_2
        date_column = col_1

    try:
        return pd.Series(
            (pd.to_datetime(obj_column) == date_column)
            | (obj_column.isnull() & date_column.isnull())
        )
    except Exception:
        try:
            return pd.Series(
                (pd.to_datetime(obj_column, format="mixed") == date_column)
                | (obj_column.isnull() & date_column.isnull())
            )
        except Exception:
            try:
                return pd.Series(obj_column.astype(str) == date_column.astype(str))
            except Exception:
                return pd.Series(False, index=col_1.index)


def polars_compare_string_and_date_columns(
    col_1: pl.Series, col_2: pl.Series
) -> pl.Series:
    """Compare a string column and date column, value-wise.

    This tries to convert a string column to a date column and compare that way.

    Parameters
    ----------
    col_1 : Polars.Series
        The first column to look at
    col_2 : Polars.Series
        The second column

    Returns
    -------
    Polars.Series
        A series of Boolean values.  True == the values match, False == the
        values don't match.
    """
    if str(col_1.dtype) in POLARS_STRING_TYPE:
        str_column = col_1
        date_column = col_2
    else:
        str_column = col_2
        date_column = col_1

    try:  # datetime is inferred
        return pl.Series(
            (str_column.str.to_datetime(strict=False).eq_missing(date_column))
            | (str_column.is_null() & date_column.is_null())
        )
    except Exception:
        try:
            return pl.Series(str_column.cast(pl.String) == date_column.cast(pl.String))
        except Exception:
            return pl.Series([False] * col_1.shape[0])
