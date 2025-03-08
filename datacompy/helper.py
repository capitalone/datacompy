"""
Helper functions for DataComPy.

This module contains standalone utility functions that can be used
independently of the main comparison classes.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

LOG = logging.getLogger(__name__)


def analyze_join_columns(
    df1: pd.DataFrame, df2: pd.DataFrame, uniqueness_threshold=90, allow_nulls=False
) -> List[Dict[str, Any]]:
    """Get statistics for potential join columns to help select appropriate join keys.

    Analyzes columns present in both DataFrames to help users select
    appropriate join keys. Provides statistics about uniqueness and null
    values, and recommends columns suitable for joining.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame to compare
    df2 : pd.DataFrame
        Second DataFrame to compare
    uniqueness_threshold : float, default=90
        Minimum percentage of uniqueness required in both dataframes for a column
        to be recommended as a join key
    allow_nulls : bool, default=False
        Whether to recommend columns containing null values

    Returns
    -------
    List[Dict]
        List of dictionaries containing column statistics:
            - Column: Column name
            - Type: Data type
            - Unique Values: Count and percentage for both frames
            - Nulls: Null counts
            - Recommended: Indicator if column is recommended for joining
              (high uniqueness, no nulls by default)
    """
    try:
        if df1 is None or df2 is None:
            LOG.warning("Cannot analyze join columns: one or both DataFrames are None")
            return []

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

    except Exception as e:
        LOG.error(f"Error analyzing join columns: {e}")
        return []


def get_recommended_join_columns(
    df1: pd.DataFrame, df2: pd.DataFrame, uniqueness_threshold=90, allow_nulls=False
) -> List[str]:
    """Get a list of column names recommended for joining the DataFrames.

    Returns columns that have high uniqueness and no null values
    based on the analysis from analyze_join_columns().

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame to compare
    df2 : pd.DataFrame
        Second DataFrame to compare
    uniqueness_threshold : float, default=90
        Minimum percentage of uniqueness required in both dataframes for a column
        to be recommended as a join key
    allow_nulls : bool, default=False
        Whether to recommend columns containing null values

    Returns
    -------
    List[str]
        List of column names recommended for joining
    """
    stats = analyze_join_columns(df1, df2, uniqueness_threshold, allow_nulls)
    return [stat["column"] for stat in stats if stat["recommended"]]
