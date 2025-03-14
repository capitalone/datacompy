"""
Helper functions for DataComPy.

This module contains standalone utility functions that can be used
independently of the main comparison classes.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Union

import pandas as pd

if TYPE_CHECKING:
    import polars as pl
    import pyspark.sql
    import snowflake.snowpark as sp

LOG = logging.getLogger(__name__)


def analyze_join_columns(
    df1: Union[pd.DataFrame, "pl.DataFrame", "pyspark.sql.DataFrame", "sp.DataFrame"],
    df2: Union[pd.DataFrame, "pl.DataFrame", "pyspark.sql.DataFrame", "sp.DataFrame"],
    uniqueness_threshold=90,
    allow_nulls=False,
) -> List[Dict[str, Any]]:
    """Get statistics for potential join columns to help select appropriate join keys.

    Analyzes columns present in both DataFrames to help users select
    appropriate join keys. Provides statistics about uniqueness and null
    values, and recommends columns suitable for joining.

    Parameters
    ----------
    df1 : Union[pd.DataFrame, pl.DataFrame, pyspark.sql.DataFrame, sp.DataFrame]
        First DataFrame to compare (pandas, polars, spark, or snowflake)
    df2 : Union[pd.DataFrame, pl.DataFrame, pyspark.sql.DataFrame, sp.DataFrame]
        Second DataFrame to compare (pandas, polars, spark, or snowflake)
    uniqueness_threshold : float, default=90
        Minimum percentage of uniqueness required in both dataframes for a column
        to be recommended as a join key
    allow_nulls : bool, default=False
        Whether to recommend columns containing null values

    Returns
    -------
    List[Dict]
        List of dictionaries containing column statistics:
            - column: Column name
            - type: Data type
            - unique_values_df1: Count and percentage for first DataFrame
            - unique_values_df2: Count and percentage for second DataFrame
            - nulls_df1: Null count in first DataFrame
            - nulls_df2: Null count in second DataFrame
            - recommended: Boolean indicating if column is recommended for joining
              (high uniqueness, no nulls by default)
    """
    try:
        if df1 is None or df2 is None:
            LOG.warning("Cannot analyze join columns: one or both DataFrames are None")
            return []

        df_type = _detect_dataframe_type(df1)

        if df_type == "pandas":
            return _analyze_join_columns_pandas(df1, df2, uniqueness_threshold, allow_nulls)
        elif df_type == "polars":
            return _analyze_join_columns_polars(df1, df2, uniqueness_threshold, allow_nulls)
        elif df_type == "spark":
            return _analyze_join_columns_spark(df1, df2, uniqueness_threshold, allow_nulls)
        elif df_type == "snowflake":
            return _analyze_join_columns_snowflake(df1, df2, uniqueness_threshold, allow_nulls)
        else:
            LOG.error(f"Unsupported DataFrame type: {df_type}")
            return []

    except Exception as e:
        LOG.error(f"Error analyzing join columns: {e}")
        return []


def _detect_dataframe_type(df):
    """Detect the type of DataFrame being used.

    Returns
    -------
        str: One of 'pandas', 'polars', 'spark', 'snowflake', or 'unknown'
    """
    df_type = type(df).__module__.split('.')[0]

    type_map = {
        "pandas": "pandas",
        "polars": "polars",
        "pyspark": "spark",
        "snowflake": "snowflake"
    }

    return type_map.get(df_type, "unknown")


def _analyze_join_columns_pandas(
    df1: pd.DataFrame, df2: pd.DataFrame, uniqueness_threshold=90, allow_nulls=False
) -> List[Dict[str, Any]]:
    """Analyze potential join columns in pandas DataFrames."""
    common_columns = sorted(set(df1.columns) & set(df2.columns))
    df1_len = len(df1)
    df2_len = len(df2)

    join_stats = []
    for col in common_columns:
        unique_vals_base = df1[col].nunique()
        unique_vals_compare = df2[col].nunique()

        null_count_base = df1[col].isna().sum()
        null_count_compare = df2[col].isna().sum()

        uniqueness_base = (unique_vals_base / df1_len * 100) if df1_len > 0 else 0
        uniqueness_compare = (
            (unique_vals_compare / df2_len * 100) if df2_len > 0 else 0
        )

        is_good_key = (
            uniqueness_base > uniqueness_threshold
            and uniqueness_compare > uniqueness_threshold
            and (allow_nulls or (null_count_base == 0 and null_count_compare == 0))
        )

        join_stats.append(
            {
                "column": col,
                "type": str(df1[col].dtype),
                "unique_values_df1": f"{unique_vals_base:,} ({uniqueness_base:.1f}%)",
                "unique_values_df2": f"{unique_vals_compare:,} ({uniqueness_compare:.1f}%)",
                "nulls_df1": null_count_base,
                "nulls_df2": null_count_compare,
                "recommended": is_good_key,
                "uniqueness_score": min(uniqueness_base, uniqueness_compare),
            }
        )

    join_stats.sort(key=lambda x: (-x["uniqueness_score"], x["column"]))

    return [
        {k: v for k, v in stat.items() if k != "uniqueness_score"}
        for stat in join_stats
    ]


def _analyze_join_columns_polars(
    df1, df2, uniqueness_threshold=90, allow_nulls=False
) -> List[Dict[str, Any]]:
    """Analyze potential join columns in polars DataFrames."""
    common_columns = sorted(set(df1.columns) & set(df2.columns))
    df1_len = len(df1)
    df2_len = len(df2)

    join_stats = []
    for col in common_columns:
        unique_vals_base = df1[col].n_unique()
        unique_vals_compare = df2[col].n_unique()

        null_count_base = df1[col].is_null().sum()
        null_count_compare = df2[col].is_null().sum()

        uniqueness_base = (unique_vals_base / df1_len * 100) if df1_len > 0 else 0
        uniqueness_compare = (
            (unique_vals_compare / df2_len * 100) if df2_len > 0 else 0
        )

        is_good_key = (
            uniqueness_base > uniqueness_threshold
            and uniqueness_compare > uniqueness_threshold
            and (allow_nulls or (null_count_base == 0 and null_count_compare == 0))
        )

        join_stats.append(
            {
                "column": col,
                "type": str(df1[col].dtype),
                "unique_values_df1": f"{unique_vals_base:,} ({uniqueness_base:.1f}%)",
                "unique_values_df2": f"{unique_vals_compare:,} ({uniqueness_compare:.1f}%)",
                "nulls_df1": null_count_base,
                "nulls_df2": null_count_compare,
                "recommended": is_good_key,
                "uniqueness_score": min(uniqueness_base, uniqueness_compare),
            }
        )

    join_stats.sort(key=lambda x: (-x["uniqueness_score"], x["column"]))

    return [
        {k: v for k, v in stat.items() if k != "uniqueness_score"}
        for stat in join_stats
    ]


def _analyze_join_columns_spark(
    df1, df2, uniqueness_threshold=90, allow_nulls=False
) -> List[Dict[str, Any]]:
    """Analyze potential join columns in spark DataFrames."""
    import pyspark.sql.functions as F

    common_columns = sorted(set(df1.columns) & set(df2.columns))
    df1_len = df1.count()
    df2_len = df2.count()

    join_stats = []
    for col in common_columns:
        unique_vals_base = df1.select(col).distinct().count()
        unique_vals_compare = df2.select(col).distinct().count()

        null_count_base = df1.filter(F.col(col).isNull()).count()
        null_count_compare = df2.filter(F.col(col).isNull()).count()

        uniqueness_base = (unique_vals_base / df1_len * 100) if df1_len > 0 else 0
        uniqueness_compare = (
            (unique_vals_compare / df2_len * 100) if df2_len > 0 else 0
        )

        is_good_key = (
            uniqueness_base > uniqueness_threshold
            and uniqueness_compare > uniqueness_threshold
            and (allow_nulls or (null_count_base == 0 and null_count_compare == 0))
        )

        col_type = next(f.dataType for f in df1.schema.fields if f.name == col)

        join_stats.append(
            {
                "column": col,
                "type": str(col_type),
                "unique_values_df1": f"{unique_vals_base:,} ({uniqueness_base:.1f}%)",
                "unique_values_df2": f"{unique_vals_compare:,} ({uniqueness_compare:.1f}%)",
                "nulls_df1": null_count_base,
                "nulls_df2": null_count_compare,
                "recommended": is_good_key,
                "uniqueness_score": min(uniqueness_base, uniqueness_compare),
            }
        )

    join_stats.sort(key=lambda x: (-x["uniqueness_score"], x["column"]))

    return [
        {k: v for k, v in stat.items() if k != "uniqueness_score"}
        for stat in join_stats
    ]


def _analyze_join_columns_snowflake(
    df1, df2, uniqueness_threshold=90, allow_nulls=False
) -> List[Dict[str, Any]]:
    """Analyze potential join columns in snowflake DataFrames."""
    import snowflake.snowpark.functions as F

    common_columns = sorted(set(df1.columns) & set(df2.columns))
    df1_len = df1.count()
    df2_len = df2.count()

    join_stats = []
    for col in common_columns:
        unique_vals_base = df1.select(col).distinct().count()
        unique_vals_compare = df2.select(col).distinct().count()

        null_count_base = df1.filter(F.col(col).is_null()).count()
        null_count_compare = df2.filter(F.col(col).is_null()).count()

        uniqueness_base = (unique_vals_base / df1_len * 100) if df1_len > 0 else 0
        uniqueness_compare = (
            (unique_vals_compare / df2_len * 100) if df2_len > 0 else 0
        )

        is_good_key = (
            uniqueness_base > uniqueness_threshold
            and uniqueness_compare > uniqueness_threshold
            and (allow_nulls or (null_count_base == 0 and null_count_compare == 0))
        )

        col_type = next(f.datatype for f in df1.schema.fields if f.name.upper() == col.upper())

        join_stats.append(
            {
                "column": col,
                "type": str(col_type),
                "unique_values_df1": f"{unique_vals_base:,} ({uniqueness_base:.1f}%)",
                "unique_values_df2": f"{unique_vals_compare:,} ({uniqueness_compare:.1f}%)",
                "nulls_df1": null_count_base,
                "nulls_df2": null_count_compare,
                "recommended": is_good_key,
                "uniqueness_score": min(uniqueness_base, uniqueness_compare),
            }
        )

    join_stats.sort(key=lambda x: (-x["uniqueness_score"], x["column"]))

    return [
        {k: v for k, v in stat.items() if k != "uniqueness_score"}
        for stat in join_stats
    ]


def get_recommended_join_columns(
    df1: Union[pd.DataFrame, "pl.DataFrame", "pyspark.sql.DataFrame", "sp.DataFrame"],
    df2: Union[pd.DataFrame, "pl.DataFrame", "pyspark.sql.DataFrame", "sp.DataFrame"],
    uniqueness_threshold=90,
    allow_nulls=False
) -> List[str]:
    """
    Get a list of recommended join columns.

    Parameters
    ----------
    df1 : Union[pd.DataFrame, pl.DataFrame, pyspark.sql.DataFrame, sp.DataFrame]
        First DataFrame to compare (pandas, polars, spark, or snowflake)
    df2 : Union[pd.DataFrame, pl.DataFrame, pyspark.sql.DataFrame, sp.DataFrame]
        Second DataFrame to compare (pandas, polars, spark, or snowflake)
    uniqueness_threshold : float, default=90
        Minimum percentage of uniqueness required in both dataframes for a column
        to be recommended as a join key
    allow_nulls : bool, default=False
        Whether to recommend columns containing null values

    Returns
    -------
    List[str]
        List of column names that are recommended for joining
    """
    stats = analyze_join_columns(df1, df2, uniqueness_threshold, allow_nulls)
    return [stat["column"] for stat in stats if stat["recommended"]]
