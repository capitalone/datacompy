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

"""Boolean comparator classes."""

from typing import Any

import pandas as pd
import polars as pl

from datacompy.comparator.base import BaseComparator

try:
    import pyspark.sql.functions as psf

    from datacompy.comparator.utility import get_spark_column_dtypes
except ImportError:
    psf = None
    get_spark_column_dtypes = None

try:
    import snowflake.snowpark.functions as spf

    from datacompy.comparator.utility import get_snowflake_column_dtypes
except ImportError:
    spf = None
    get_snowflake_column_dtypes = None

_NUMERIC_SPARK_TYPES = (
    "tinyint",
    "smallint",
    "int",
    "bigint",
    "float",
    "double",
    "decimal",
)
_NUMERIC_SNOWFLAKE_TYPES = (
    "tinyint",
    "smallint",
    "int",
    "bigint",
    "float",
    "double",
    "decimal",
)


class PandasBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in Pandas."""

    def compare(
        self,
        col1: pd.Series,
        col2: pd.Series,
        **kwargs: Any,
    ) -> pd.Series | None:
        """Compare columns when either Pandas dtype is Boolean."""
        if not (
            pd.api.types.is_bool_dtype(col1.dtype)
            or pd.api.types.is_bool_dtype(col2.dtype)
        ):
            return None
        if col1.shape != col2.shape:
            return None
        return (col1.eq(col2) | (col1.isna() & col2.isna())).fillna(False).astype(bool)


class PolarsBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in Polars."""

    def compare(
        self, col1: pl.Series, col2: pl.Series, **kwargs: Any
    ) -> pl.Series | None:
        """Compare columns when either Polars dtype is Boolean."""
        col1_is_bool = col1.dtype == pl.Boolean
        col2_is_bool = col2.dtype == pl.Boolean
        if not (col1_is_bool or col2_is_bool):
            return None
        if col1.shape != col2.shape:
            return None
        if col1_is_bool and col2_is_bool:
            return col1.eq_missing(col2)
        if (col1_is_bool and col2.dtype.is_numeric()) or (
            col2_is_bool and col1.dtype.is_numeric()
        ):
            left = col1.cast(pl.Int8) if col1_is_bool else col1
            right = col2.cast(pl.Int8) if col2_is_bool else col2
            return left.eq_missing(right)
        return col1.is_null() & col2.is_null()


class SparkBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in PySpark."""

    @staticmethod
    def _compare_boolean_to_numeric(boolean_col: str, numeric_col: str) -> Any:
        """Compare Boolean values with numeric 1/0 without casting the numeric column."""
        boolean = psf.col(boolean_col)
        numeric = psf.col(numeric_col)
        both_null = boolean.isNull() & numeric.isNull()
        values_equal = (
            boolean.eqNullSafe(psf.lit(True)) & numeric.eqNullSafe(psf.lit(1))
        ) | (boolean.eqNullSafe(psf.lit(False)) & numeric.eqNullSafe(psf.lit(0)))
        return both_null | values_equal

    def compare(
        self, dataframe: Any, col1: str, col2: str, **kwargs: Any
    ) -> Any | None:
        """Compare columns when either Spark dtype is Boolean."""
        if get_spark_column_dtypes is None or psf is None:
            return None
        dtype1, dtype2 = get_spark_column_dtypes(dataframe, col1, col2)
        col1_is_bool = dtype1 == "boolean"
        col2_is_bool = dtype2 == "boolean"
        if not (col1_is_bool or col2_is_bool):
            return None
        if col1_is_bool and col2_is_bool:
            return psf.col(col1).eqNullSafe(psf.col(col2))
        dtype1_is_numeric = dtype1.startswith(_NUMERIC_SPARK_TYPES)
        dtype2_is_numeric = dtype2.startswith(_NUMERIC_SPARK_TYPES)
        if col1_is_bool and dtype2_is_numeric:
            return self._compare_boolean_to_numeric(col1, col2)
        if col2_is_bool and dtype1_is_numeric:
            return self._compare_boolean_to_numeric(col2, col1)
        return psf.col(col1).isNull() & psf.col(col2).isNull()


class SnowflakeBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in Snowpark."""

    @staticmethod
    def _compare_boolean_to_numeric(boolean_col: str, numeric_col: str) -> Any:
        """Compare Boolean values with numeric 1/0 without casting the numeric column."""
        boolean = spf.col(boolean_col)
        numeric = spf.col(numeric_col)
        both_null = spf.is_null(boolean) & spf.is_null(numeric)
        values_equal = (
            boolean.eqNullSafe(spf.lit(True)) & numeric.eqNullSafe(spf.lit(1))
        ) | (boolean.eqNullSafe(spf.lit(False)) & numeric.eqNullSafe(spf.lit(0)))
        return both_null | values_equal

    def compare(
        self,
        dataframe: Any,
        col1: str,
        col2: str,
        col_match: str,
        **kwargs: Any,
    ) -> Any | None:
        """Compare columns when either Snowflake dtype is Boolean."""
        if get_snowflake_column_dtypes is None or spf is None:
            return None
        dtype1, dtype2 = get_snowflake_column_dtypes(dataframe, col1, col2)
        col1_is_bool = dtype1 == "boolean"
        col2_is_bool = dtype2 == "boolean"
        if not (col1_is_bool or col2_is_bool):
            return None
        if col1_is_bool and col2_is_bool:
            expression = spf.col(col1).eqNullSafe(spf.col(col2))
        elif col1_is_bool and dtype2.startswith(_NUMERIC_SNOWFLAKE_TYPES):
            expression = self._compare_boolean_to_numeric(col1, col2)
        elif col2_is_bool and dtype1.startswith(_NUMERIC_SNOWFLAKE_TYPES):
            expression = self._compare_boolean_to_numeric(col2, col1)
        else:
            expression = spf.is_null(spf.col(col1)) & spf.is_null(spf.col(col2))
        return dataframe.withColumn(col_match, expression)
