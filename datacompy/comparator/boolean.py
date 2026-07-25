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

PYSPARK_BOOLEAN_TYPE = "boolean"
SNOWFLAKE_BOOLEAN_TYPE = "boolean"

# Optional Spark dependencies
try:
    import pyspark as ps
    import pyspark.sql.functions as psf

    from datacompy.comparator.numeric import NUMERIC_PYSPARK_TYPES
    from datacompy.comparator.utility import get_spark_column_dtypes
except ImportError:
    ps = None
    psf = None
    NUMERIC_PYSPARK_TYPES = None

# Optional Snowflake dependencies
try:
    import snowflake.snowpark as sp
    import snowflake.snowpark.functions as spf

    from datacompy.comparator.utility import get_snowflake_column_dtypes
except ImportError:
    sp = None
    spf = None


class PandasBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in Pandas."""

    def compare(
        self,
        col1: pd.Series,
        col2: pd.Series,
        **kwargs: Any,
    ) -> pd.Series | None:
        """Compare columns when either column holds Boolean values.

        Boolean comparisons are exact and null-safe. When a Boolean column is
        compared with another dtype, normal Pandas equality semantics are used;
        for example, ``True`` matches ``1`` and ``False`` matches ``0``.

        Parameters
        ----------
        col1 : pd.Series
            The first Pandas Series to compare.
        col2 : pd.Series
            The second Pandas Series to compare.
        **kwargs : Any
            Unused; accepted so this comparator matches the pipeline signature.

        Returns
        -------
        pd.Series
            A Pandas Series of booleans indicating whether the values in `col1`
            and `col2` are equal. Two nulls are treated as equal.
        None
            if the columns are not comparable.

        Notes
        -----
        Detection uses ``infer_dtype`` rather than the column ``dtype`` because
        Pandas represents Boolean data as ``object`` in common cases: a Boolean
        column containing a null, and any Boolean column that has been through
        an outer merge (which upcasts ``bool`` to ``object``).
        """
        if col1.shape != col2.shape:
            return None

        if not (
            pd.api.types.infer_dtype(col1, skipna=True) == "boolean"
            or pd.api.types.infer_dtype(col2, skipna=True) == "boolean"
        ):
            return None

        return (col1.eq(col2) | (col1.isna() & col2.isna())).fillna(False).astype(bool)


class PolarsBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in Polars."""

    def compare(
        self,
        col1: pl.Series,
        col2: pl.Series,
        **kwargs: Any,
    ) -> pl.Series | None:
        """Compare columns when either Polars dtype is Boolean.

        Boolean comparisons are exact and null-safe. When a Boolean column is
        compared with another dtype, normal Polars equality semantics are used;
        for example, ``True`` matches ``1`` and ``False`` matches ``0``.

        Parameters
        ----------
        col1 : pl.Series
            The first Polars Series to compare.
        col2 : pl.Series
            The second Polars Series to compare.
        **kwargs : Any
            Unused; accepted so this comparator matches the pipeline signature.

        Returns
        -------
        pl.Series
            A Polars Series of booleans indicating whether the values in `col1`
            and `col2` are equal. Two nulls are treated as equal.
        None
            if the columns are not comparable.

        Notes
        -----
        ``eq_missing`` is used rather than ``==`` because Polars propagates
        nulls through ``==``; ``eq_missing`` treats two nulls as equal and a
        null against a value as unequal.
        """
        if col1.shape != col2.shape:
            return None

        if not (col1.dtype == pl.Boolean or col2.dtype == pl.Boolean):
            return None

        try:
            return col1.eq_missing(col2)
        except Exception:
            # Polars raises when the dtypes have no comparison supertype (a
            # Boolean against a List, Date, Struct, ...). Signal that this
            # comparator cannot handle the pair so the pipeline continues.
            return None


class SparkBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in PySpark."""

    def compare(
        self,
        dataframe: "ps.sql.DataFrame",
        col1: str,
        col2: str,
        **kwargs: Any,
    ) -> "ps.sql.Column | None":
        """Compare two columns in a PySpark DataFrame when either is Boolean.

        Boolean comparisons are exact and null-safe. A Boolean column may also
        be compared against a numeric column, in which case Spark casts the
        Boolean to the numeric type; for example, ``True`` matches ``1`` and
        ``False`` matches ``0``.

        Parameters
        ----------
        dataframe : pyspark.sql.DataFrame
            The PySpark DataFrame containing the columns to compare.
        col1 : str
            The name of the first column to compare.
        col2 : str
            The name of the second column to compare.
        **kwargs : Any
            Unused; accepted so this comparator matches the pipeline signature.

        Returns
        -------
        pyspark.sql.Column
            A Column containing boolean values indicating whether the values in
            `col1` and `col2` are equal. Two nulls are treated as equal.
        None
            Columns are not comparable if neither is Boolean, or if a Boolean is
            paired with anything other than a numeric type.

        Notes
        -----
        Unlike the Pandas and Polars comparators, this one claims only
        Boolean/Boolean and Boolean/numeric pairs. Spark builds the comparison
        lazily, so an unsupported pair (Boolean against a date, array, or
        struct) raises ``AnalysisException`` when the plan is analysed rather
        than when the Column is constructed, which is too late for this method to
        catch. Unsupported pairs are therefore rejected up front. Boolean
        against string is also declined, because Spark would implicitly cast
        the string to a Boolean and report ``True == 'yes'`` as a match, which
        the other backends do not.

        The Boolean/numeric case casts both sides to ``double`` rather than
        relying on Spark's implicit coercion, which is only available with
        ``spark.sql.ansi.enabled=false``. Under ANSI mode, the default from
        Spark 4 comparing a Boolean against a numeric raises
        ``DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES``. The explicit cast behaves
        identically under both settings.
        """
        base_dtype, compare_dtype = get_spark_column_dtypes(dataframe, col1, col2)
        base_boolean_type = base_dtype == PYSPARK_BOOLEAN_TYPE
        compare_boolean_type = compare_dtype == PYSPARK_BOOLEAN_TYPE
        base_numeric_type = any(base_dtype.startswith(t) for t in NUMERIC_PYSPARK_TYPES)
        compare_numeric_type = any(
            compare_dtype.startswith(t) for t in NUMERIC_PYSPARK_TYPES
        )

        if base_boolean_type and compare_boolean_type:
            when_clause = psf.col(col1).eqNullSafe(psf.col(col2))
        elif (base_boolean_type and compare_numeric_type) or (
            compare_boolean_type and base_numeric_type
        ):
            when_clause = (
                psf.col(col1).cast("double").eqNullSafe(psf.col(col2).cast("double"))
            )
        else:
            return None

        return psf.when(when_clause, psf.lit(True)).otherwise(psf.lit(False))


class SnowflakeBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in Snowflake."""

    def compare(
        self,
        dataframe: "sp.DataFrame",
        col1: str,
        col2: str,
        col_match: str,
        **kwargs: Any,
    ) -> "sp.DataFrame | None":
        """Compare two Boolean columns in a Snowflake DataFrame.

        Boolean comparisons are exact and null-safe.

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
        **kwargs : Any
            Unused; accepted so this comparator matches the pipeline signature.

        Returns
        -------
        snowflake.snowpark.DataFrame
            The DataFrame with an additional column containing the comparison
            results. Two nulls are treated as equal.
        None
            if both columns are not Boolean.

        Notes
        -----
        Unlike the other backends, this comparator claims only Boolean/Boolean
        pairs. Snowflake implicitly converts between BOOLEAN and NUMBER, but the
        direction of that conversion decides whether ``2`` matches ``True``:
        converting the Boolean gives ``1 = 2`` (no match, which is what Pandas,
        Polars, and Spark report), while converting the number gives
        ``TRUE = TRUE`` (a match). Boolean/numeric pairs are therefore left to
        fall through, preserving existing behaviour, until the semantics can be
        confirmed against a live Snowflake session.
        """
        base_dtype, compare_dtype = get_snowflake_column_dtypes(dataframe, col1, col2)

        if (
            base_dtype == SNOWFLAKE_BOOLEAN_TYPE
            and compare_dtype == SNOWFLAKE_BOOLEAN_TYPE
        ):
            try:
                return dataframe.withColumn(
                    col_match,
                    spf.when(
                        spf.col(col1).eqNullSafe(spf.col(col2)), spf.lit(True)
                    ).otherwise(spf.lit(False)),
                )
            except Exception:
                return dataframe.withColumn(col_match, spf.lit(False))
        else:
            return None
