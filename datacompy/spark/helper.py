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

"""
Helper function module contributed by Capital One's Hydra Team.

Helper functions to assist in specific usecases where there is no columns to join
and use the row order of the datasets.
"""

import logging

from datacompy.spark.sql import SparkSQLCompare

LOG = logging.getLogger(__name__)

try:
    import pyspark.sql
    from pyspark.sql import Window
    from pyspark.sql import types as T
    from pyspark.sql.functions import col, format_number, row_number
except ImportError:
    LOG.warning(
        "Please note that you are missing the optional dependency: spark. "
        "If you need to use this functionality it must be installed."
    )


def compare_by_row(
    spark_session: "pyspark.sql.SparkSession",
    base_dataframe: "pyspark.sql.DataFrame",
    compare_dataframe: "pyspark.sql.DataFrame",
    string2double_cols: list[str] | None,
    abs_tol: float = 0,
    rel_tol: float = 0,
    df1_name: str = "df1",
    df2_name: str = "df2",
    ignore_spaces: bool = False,
    ignore_case: bool = False,
    cast_column_names_lower: bool = True,
) -> SparkSQLCompare:
    """Run a detailed analysis on specific usecases where there is no columns to join and use the row order of the datasets.

    If you know which columns to join on then please use ``SparkSQLCompare`` directly as this is meant to help
    support very specific helper usecases using row order contributed by Capital One's Hydra Team.

    Parameters
    ----------
    spark_session : pyspark.sql.SparkSession
        A ``SparkSession`` to be used to execute Spark commands in the comparison.
    base_dataframe: pyspark.sql.DataFrame
        Dataset to be compared against
    compare_dataframe: pyspark.sql.DataFrame
        dataset to compare
    string2double_cols: List[str], optional
        The columns that contain numeric values but are stored as string types
    abs_tol : float, optional
        Absolute tolerance between two values.
    rel_tol : float, optional
        Relative tolerance between two values.
    df1_name : str, optional
        A string name for the first dataframe.  This allows the reporting to
        print out an actual name instead of "df1", and allows human users to
        more easily track the dataframes.
    df2_name : str, optional
        A string name for the second dataframe
    ignore_spaces : bool, optional
        Flag to strip whitespace (including newlines) from string columns (including any join
        columns)
    ignore_case : bool, optional
        Flag to ignore the case of string columns
    cast_column_names_lower: bool, optional
        Boolean indicator that controls of column names will be cast into lower case

    Returns
    -------
    datacompy.spark.sql.SparkSQLCompare
    """
    # Convert fields that contain numeric values stored as strings to numeric types for comparison
    if len(string2double_cols) != 0:
        base_dataframe = handle_numeric_strings(base_dataframe, string2double_cols)
        compare_dataframe = handle_numeric_strings(
            compare_dataframe, string2double_cols
        )

    sorted_base_df, sorted_compare_df = sort_rows(base_dataframe, compare_dataframe)
    column_to_join = ["row"]

    LOG.info("Compared by column(s): ", column_to_join)
    if string2double_cols:
        LOG.info(
            "String column(s) cast to doubles for numeric comparison: ",
            string2double_cols,
        )
    return SparkSQLCompare(
        spark_session=spark_session,
        df1=sorted_base_df,
        df2=sorted_compare_df,
        join_columns=column_to_join,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        df1_name=df1_name,
        df2_name=df2_name,
        ignore_spaces=ignore_spaces,
        ignore_case=ignore_case,
        cast_column_names_lower=cast_column_names_lower,
    )


def handle_numeric_strings(
    df: "pyspark.sql.DataFrame", field_list: list[str]
) -> "pyspark.sql.DataFrame":
    """Convert columns in field_list from numeric strings to DoubleType.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
        The DataFrame to be converted
    field_list: List[str]
        List of StringType columns to be converted to DoubleType

    Returns
    -------
    pyspark.sql.DataFrame
    """
    for this_col in field_list:
        df = df.withColumn(this_col, col(this_col).cast(T.DoubleType()))
    return df


def sort_rows(
    base_df: "pyspark.sql.DataFrame", compare_df: "pyspark.sql.DataFrame"
) -> "pyspark.sql.DataFrame":
    """Add new column to each DataFrame that numbers the rows, so they can be compared by row number.

    Parameters
    ----------
    base_df: pyspark.sql.DataFrame
        The base DataFrame to be sorted
    compare_df: pyspark.sql.DataFrame
        The compare DataFrame to be sorted

    Returns
    -------
    pyspark.sql.DataFrame, pyspark.sql.DataFrame


    """
    base_cols = base_df.columns
    compare_cols = compare_df.columns

    # Ensure both DataFrames have the same columns
    for x in base_cols:
        if x not in compare_cols:
            raise Exception(
                f"{x} is present in base_df but does not exist in compare_df"
            )

    if set(base_cols) != set(compare_cols):
        LOG.warning(
            "WARNING: There are columns present in Compare df that do not exist in Base df. "
            "The Base df columns will be used for row-wise sorting and may produce unanticipated "
            "report output if the extra fields are not null."
        )

    w = Window.orderBy(*base_cols)
    sorted_base_df = base_df.select("*", row_number().over(w).alias("row"))
    sorted_compare_df = compare_df.select("*", row_number().over(w).alias("row"))
    return sorted_base_df, sorted_compare_df


def sort_columns(
    base_df: "pyspark.sql.DataFrame", compare_df: "pyspark.sql.DataFrame"
) -> "pyspark.sql.DataFrame":
    """Sort both DataFrames by their columns to ensure consistent order.

    Parameters
    ----------
    base_df: pyspark.sql.DataFrame
        The base DataFrame to be sorted
    compare_df: pyspark.sql.DataFrame
        The compare DataFrame to be sorted

    Returns
    -------
    pyspark.sql.DataFrame, pyspark.sql.DataFrame
    """
    # Ensure both DataFrames have the same columns
    common_columns = set(base_df.columns)
    for x in common_columns:
        if x not in compare_df.columns:
            raise Exception(
                f"{x} is present in base_df but does not exist in compare_df"
            )
    # Sort both DataFrames to ensure consistent order
    base_df = base_df.orderBy(*common_columns)
    compare_df = compare_df.orderBy(*common_columns)
    return base_df, compare_df


def format_numeric_fields(df: "pyspark.sql.DataFrame") -> "pyspark.sql.DataFrame":
    """Round and truncate numeric fields to 5 decimal places.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
        The DataFrame to be formatted

    Returns
    -------
    pyspark.sql.DataFrame
    """
    fixed_cols = []
    numeric_types = [
        "tinyint",
        "smallint",
        "int",
        "bigint",
        "float",
        "double",
        "decimal",
    ]

    for c in df.dtypes:
        # do not change non-numeric fields
        if c[1] not in numeric_types:
            fixed_cols.append(col(c[0]))
        # round & truncate numeric fields
        else:
            new_val = format_number(col(c[0]), 5).alias(c[0])
            fixed_cols.append(new_val)

    formatted_df = df.select(*fixed_cols)
    return formatted_df
