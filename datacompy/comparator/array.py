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

"""Array Like Comparator Class."""

import logging

import numpy as np
import pandas as pd
import polars as pl

from datacompy.comparator.base import BaseComparator

LOG = logging.getLogger(__name__)

POLARS_ARRAY_TYPE = ["List", "Array"]

try:
    import pyspark as ps
    import pyspark.sql.functions as psf

    from datacompy.comparator.utility import get_spark_column_dtypes
except ImportError:
    ps = None
    psf = None

try:
    import snowflake.snowpark as sp
    import snowflake.snowpark.functions as spf
    import snowflake.snowpark.types as spt

    from datacompy.comparator.utility import get_snowflake_column_dtypes

except ImportError:
    sp = None
    spf = None
    spt = None


class PandasArrayLikeComparator(BaseComparator):
    """Comparator for array-like columns in Pandas."""

    def compare(self, col1: pd.Series, col2: pd.Series) -> pd.Series | None:
        """
        Compare two array like columns for equality.

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
            are equal.
        None
            if the columns are not comparable.
        """
        if col1.shape != col2.shape:
            return None

        if (
            pd.api.types.infer_dtype(col1).startswith("mixed")
            or pd.api.types.infer_dtype(col2).startswith("mixed")
        ) and (
            # Using any() instead of all() for early termination
            not any(not isinstance(item, list | np.ndarray) for item in col1)
            and not any(not isinstance(item, list | np.ndarray) for item in col2)
        ):
            temp_df = pd.DataFrame({"col1": col1, "col2": col2})
            return temp_df.apply(
                lambda row: np.array_equal(row.col1, row.col2, equal_nan=True), axis=1
            )
        else:
            return None


class PolarsArrayLikeComparator(BaseComparator):
    """Comparator for array-like columns in Polars."""

    def compare(self, col1: pl.Series, col2: pl.Series) -> pl.Series | None:
        """
        Compare two array like columns for equality.

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
            are equal.
        None
            if the columns are not comparable.
        """
        if col1.shape != col2.shape:
            return None

        if (
            str(col1.dtype.base_type()) in POLARS_ARRAY_TYPE
            and str(col2.dtype.base_type()) in POLARS_ARRAY_TYPE
        ):
            # For Polars list comparison, we can use the eq_missing operator
            # which handles null values correctly
            return pl.Series(col1.eq_missing(col2))
        else:
            return None


class SparkArrayLikeComparator(BaseComparator):
    """Comparator for array-like columns in PySpark."""

    def compare(
        self, dataframe: "ps.sql.DataFrame", col1: str, col2: str
    ) -> "ps.sql.Column | None":
        """
        Compare two array like columns for equality.

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
        pyspark.sql.Column
            A PySpark Column containing boolean values indicating whether the values in
            `col_1` and `col_2` are equal.
        None
            if the columns are not comparable.
        """
        base_dtype, compare_dtype = get_spark_column_dtypes(dataframe, col1, col2)
        if base_dtype.startswith("array") and compare_dtype.startswith("array"):
            when_clause = psf.col(col1).eqNullSafe(psf.col(col2))
            return psf.when(when_clause, psf.lit(True)).otherwise(psf.lit(False))
        else:
            return None


class SnowflakeArrayLikeComparator(BaseComparator):
    """Comparator for array-like columns in Snowflake."""

    def compare(
        self, dataframe: "sp.DataFrame", col1: str, col2: str, col_match: str
    ) -> "sp.DataFrame | None":
        """
        Compare two array like columns for equality.

        Parameters
        ----------
        dataframe: snowflake.snowpark.DataFrame
            DataFrame to do comparison on
        col1 : str
            The first column to look at
        col2 : str
            The second column
        col_match : str
            The matching column denoting if the compare was a match or not

        Returns
        -------
        snowflake.snowpark.DataFrame
            A PySpark DataFrame with an additional column (`col_match`) containing
            boolean values indicating whether the values in `col1` and `col2` are
            equal.
        None
            if the columns are not comparable.
        """
        base_dtype, compare_dtype = get_snowflake_column_dtypes(dataframe, col1, col2)
        if base_dtype.startswith("array") and compare_dtype.startswith("array"):
            when_clause = spf.col(col1).eqNullSafe(spf.col(col2))
            return dataframe.withColumn(
                col_match,
                spf.when(when_clause, spf.lit(True)).otherwise(spf.lit(False)),
            )
        else:
            return None
