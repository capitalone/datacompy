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

"""Numeric Comparator Class."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl

from datacompy.comparator.base import BaseNumericComparator

LOG = logging.getLogger(__name__)

try:
    import pyspark.sql
    import pyspark.sql.functions as psf
except ImportError:
    LOG.warning(
        "Please note that you are missing the optional dependency: spark. "
        "If you need to use this functionality it must be installed."
    )

try:
    import snowflake.snowpark as sp
    import snowflake.snowpark.functions as spf
    import snowflake.snowpark.types as spt

    NUMERIC_SNOWPARK_TYPES = [
        spt.ByteType,
        spt.ShortType,
        spt.IntegerType,
        spt.LongType,
        spt.FloatType,
        spt.DoubleType,
        spt.DecimalType,
    ]
except ImportError:
    LOG.warning(
        "Please note that you are missing the optional dependency: snowflake. "
        "If you need to use this functionality it must be installed."
    )


def decimal_comparator():
    """Check equality with decimal(X, Y) types.

    Otherwise treated as the string "decimal".
    """

    class DecimalComparator(str):
        def __eq__(self, other):
            return len(other) >= 7 and other[0:7] == "decimal"

    return DecimalComparator("decimal")


NUMERIC_SPARK_TYPES = [
    "tinyint",
    "smallint",
    "int",
    "bigint",
    "float",
    "double",
    decimal_comparator(),
]


def _get_spark_column_dtypes(
    dataframe: "pyspark.sql.DataFrame", col_1: "str", col_2: "str"
) -> Tuple[str, str]:
    """Get the dtypes of two columns.

    Parameters
    ----------
    dataframe: pyspark.sql.DataFrame
        DataFrame to do comparison on
    col_1 : str
        The first column to look at
    col_2 : str
        The second column

    Returns
    -------
    Tuple(str, str)
        Tuple of base and compare datatype
    """
    base_dtype = next(d[1] for d in dataframe.dtypes if d[0] == col_1)
    compare_dtype = next(d[1] for d in dataframe.dtypes if d[0] == col_2)
    return base_dtype, compare_dtype


def _get_snowflake_column_dtypes(
    dataframe: "sp.DataFrame", col_1: "str", col_2: "str"
) -> tuple[str, str]:
    """Get the dtypes of two columns.

    Parameters
    ----------
    dataframe: sp.DataFrame
        DataFrame to do comparison on
    col_1 : str
        The first column to look at
    col_2 : str
        The second column

    Returns
    -------
    Tuple(str, str)
        Tuple of base and compare datatype
    """
    df_raw_dtypes = [
        (name, field.datatype)
        for name, field in zip(
            dataframe.schema.names, dataframe.schema.fields, strict=False
        )
    ]
    base_dtype = next(d[1] for d in df_raw_dtypes if d[0] == col_1)
    compare_dtype = next(d[1] for d in df_raw_dtypes if d[0] == col_2)
    return base_dtype, compare_dtype


class PolarsNumericComparator(BaseNumericComparator):
    """Comparator for numeric columns in Polars.

    Parameters
    ----------
    rtol : float
        The relative tolerance to use for comparison.
    atol : float
        The absolute tolerance to use for comparison.
    """

    def compare(self, col1: pl.Series, col2: pl.Series) -> pl.Series:
        """
        Compare two Polars Series for approximate equality.

        Parameters
        ----------
        col1 : pl.Series
            The first Polars Series to compare.
        col2 : pl.Series
            The second Polars Series to compare.

        Returns
        -------
        pl.Series
            A Polars Series of booleans indicating whether the values in `col1` and `col2`
            are approximately equal within the given tolerances.

        Notes
        -----
        - The comparison uses `np.isclose` to check for approximate equality.
        - If the series cannot be directly compared due to type mismatches, they are cast
          to `pl.Float64` for comparison.
        - If the Series shapes do not match, and neither type is numeric a series of `False`
          values is returned.
        """
        if (col1.shape == col2.shape) and (
            col1.dtype.is_numeric() and col2.dtype.is_numeric()
        ):
            try:
                return pl.Series(
                    np.isclose(
                        col1, col2, rtol=self.rtol, atol=self.atol, equal_nan=True
                    )
                )
            except Exception:
                return pl.Series(
                    np.isclose(
                        col1.cast(pl.Float64, strict=True),
                        col2.cast(pl.Float64, strict=True),
                        rtol=self.rtol,
                        atol=self.atol,
                        equal_nan=True,
                    )
                )
        else:
            return pl.Series([False] * col1.shape[0])


class PandasNumericComparator(BaseNumericComparator):
    """Comparator for numeric columns in Pandas.

    Parameters
    ----------
    rtol : float
        The relative tolerance to use for comparison.
    atol : float
        The absolute tolerance to use for comparison.
    """

    def compare(self, col1: pd.Series, col2: pd.Series) -> pd.Series:
        """
        Compare two Pandas Series for approximate equality.

        Parameters
        ----------
        col1 : pd.Series
            The first Pandas Series to compare.
        col2 : pd.Series
            The second Pandas Series to compare.

        Returns
        -------
        pd.Series
            A Pandas Series of booleans indicating whether the values in `col1` and `col2`
            are approximately equal within the given tolerances.

        Notes
        -----
        - The comparison uses `np.isclose` to check for approximate equality.
        - If the series cannot be directly compared due to type mismatches, they are cast
          to `float` for comparison.
        - If the Series shapes do not match, and neither type is numeric a series of `False`
          values is returned.
        """
        if (col1.shape == col2.shape) and (
            pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2)
        ):
            try:
                return pd.Series(
                    np.isclose(
                        col1, col2, rtol=self.rtol, atol=self.atol, equal_nan=True
                    )
                )
            except TypeError:
                return pd.Series(
                    np.isclose(
                        col1.astype(float),
                        col2.astype(float),
                        rtol=self.rtol,
                        atol=self.atol,
                        equal_nan=True,
                    )
                )
        else:
            return pd.Series(False, index=col1.index)


class SparkNumericComparator(BaseNumericComparator):
    """Comparator for numeric columns in PySpark.

    Parameters
    ----------
    rtol : float
        The relative tolerance to use for comparison.
    atol : float
        The absolute tolerance to use for comparison.

    """

    def compare(
        self, dataframe: pyspark.sql.DataFrame, col1: str, col2: str, col_match: str
    ) -> pyspark.sql.DataFrame:
        """
        Compare two columns in a PySpark DataFrame for approximate equality.

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
            A PySpark DataFrame with an additional column (`col_match`) containing
            boolean values indicating whether the values in `col_1` and `col_2` are
            approximately equal within the given tolerances.

        Notes
        -----
        - The comparison uses PySpark SQL functions to check for approximate equality.
        - Null-safe equality (`eqNullSafe`) is used to handle null values.
        - If either column contains NaN values, they are handled explicitly to avoid
          incorrect comparisons.
        """
        base_dtype, compare_dtype = _get_spark_column_dtypes(dataframe, col1, col2)
        if (base_dtype in NUMERIC_SPARK_TYPES) and (
            compare_dtype in NUMERIC_SPARK_TYPES
        ):
            output = dataframe.withColumn(
                col_match,
                psf.when(
                    (psf.col(col1).eqNullSafe(psf.col(col2)))
                    | (
                        psf.abs(psf.col(col1) - psf.col(col2))
                        <= psf.lit(self.atol)
                        + (psf.lit(self.rtol) * psf.abs(psf.col(col2)))
                    ),
                    # corner case of col1 != NaN and col2 == Nan returns True incorrectly
                    psf.when(
                        (psf.isnan(psf.col(col1)) == False)  # noqa: E712
                        & (psf.isnan(psf.col(col2)) == True),  # noqa: E712
                        psf.lit(False),
                    ).otherwise(psf.lit(True)),
                ).otherwise(psf.lit(False)),
            )
        else:
            output = dataframe.withColumn(col_match, psf.lit(False))
        return output


class SnowflakeNumericComparator(BaseNumericComparator):
    """Comparator for numeric columns in Snowflake.

    Parameters
    ----------
    rtol : float
        The relative tolerance to use for comparison.
    atol : float
        The absolute tolerance to use for comparison.

    """

    def compare(
        self,
        dataframe: sp.DataFrame,
        col1: str,
        col2: str,
        col_match: str,
    ) -> sp.DataFrame:
        """
        Compare two columns in a Snowpark DataFrame for approximate equality.

        Parameters
        ----------
        dataframe : snowflake.snowpark.DataFrame
            The Snowpark DataFrame containing the columns to compare.
        col1 : str
            The name of the first column to compare.
        col2 : str
            The name of the second column to compare.
        col_match : str
            The name of the output column that will store the comparison results.

        Returns
        -------
        snowflake.snowpark.DataFrame
            A Snowpark DataFrame with an additional column (`col_match`) containing
            boolean values indicating whether the values in `col1` and `col2` are
            approximately equal within the given tolerances.

        Notes
        -----
        - The comparison uses Snowpark SQL functions to check for approximate equality.
        - Null-safe equality (`eqNullSafe`) is used to handle null values.
        - If either column contains null values, they are handled explicitly to avoid
          incorrect comparisons.
        """
        base_dtype, compare_dtype = _get_snowflake_column_dtypes(dataframe, col1, col2)
        if (base_dtype in NUMERIC_SNOWPARK_TYPES) and (
            compare_dtype in NUMERIC_SNOWPARK_TYPES
        ):
            output = dataframe.withColumn(
                col_match,
                spf.when(
                    (spf.col(col1).eqNullSafe(spf.col(col2)))
                    | (
                        spf.abs(spf.col(col1) - spf.col(col2))
                        <= spf.lit(self.atol)
                        + (spf.lit(self.rtol) * spf.abs(spf.col(col2)))
                    ),
                    # corner case of col1 != null and col2 == null returns True incorrectly
                    spf.when(
                        (spf.is_null(spf.col(col1)) == False)  # noqa: E712
                        & (spf.is_null(spf.col(col2)) == True),  # noqa: E712
                        spf.lit(False),
                    ).otherwise(spf.lit(True)),
                ).otherwise(spf.lit(False)),
            )
        else:
            output = dataframe.withColumn(col_match, psf.lit(False))
        return output
