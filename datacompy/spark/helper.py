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
Helper function module.

Helper functions to assist in specific usecases where there is no columns to join
and use the row order of the datasets.
"""

from datacompy.logger import INFO, get_logger
from datacompy.spark.sql import SparkSQLCompare

LOG = get_logger(__name__, INFO)

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


def detailed_compare(
    spark_session,
    prod_dataframe: pyspark.sql.DataFrame,
    release_dataframe: pyspark.sql.DataFrame,
    column_to_join,
    string2double_cols=None,
) -> SparkSQLCompare:
    """Run a detailed analysis on results.

    Parameters
    ----------
    spark_session : pyspark.sql.SparkSession
        A ``SparkSession`` to be used to execute Spark commands in the comparison.

    prod_dataframe: pyspark.sql.DataFrame
        Dataset to be compared against

    release_dataframe: pyspark.sql.DataFrame
        dataset to compare

    column_to_join: List[str], optional
        the column by which the two datasets can be joined, an identifier that indicates which
        rows in both datasets should be compared. If null, the rows are compared in the order
        they are given dataset to compare

    string2double_cols: List[str], optional
        The columns that contain numeric values but are stored as string types

    Returns
    -------
    datacompy.spark.sql.SparkSQLCompare
    """
    # Convert fields that contain numeric values stored as strings to numeric types for comparison
    if len(string2double_cols) != 0:
        prod_dataframe = handle_numeric_strings(prod_dataframe, string2double_cols)
        release_dataframe = handle_numeric_strings(
            release_dataframe, string2double_cols
        )

    if len(column_to_join) == 0:
        # will add a new column that numbers the rows so datasets can be compared by row number instead of by a
        # common column
        sorted_prod_df, sorted_release_df = sort_rows(prod_dataframe, release_dataframe)
        column_to_join = ["row"]
    else:
        sorted_prod_df = prod_dataframe
        sorted_release_df = release_dataframe

    LOG.info("Compared by column(s): ", column_to_join)
    if string2double_cols:
        LOG.info(
            "String column(s) cast to doubles for numeric comparison: ",
            string2double_cols,
        )
    compared_data = SparkSQLCompare(
        spark_session,
        sorted_prod_df,
        sorted_release_df,
        join_columns=column_to_join,
        abs_tol=0.0001,
    )
    return compared_data


def handle_numeric_strings(df, field_list):
    """"""
    for this_col in field_list:
        df = df.withColumn(this_col, col(this_col).cast(T.DoubleType()))
    return df


def sort_rows(prod_df, release_df):
    """"""
    prod_cols = prod_df.columns
    release_cols = release_df.columns

    # Ensure both DataFrames have the same columns
    for x in prod_cols:
        if x not in release_cols:
            raise Exception(
                f"{x} is present in prod_df but does not exist in release_df"
            )

    if set(prod_cols) != set(release_cols):
        LOG.warning(
            "WARNING: There are columns present in Compare df that do not exist in Base df. "
            "The Base df columns will be used for row-wise sorting and may produce unanticipated "
            "report output if the extra fields are not null."
        )

    w = Window.orderBy(*prod_cols)
    sorted_prod_df = prod_df.select("*", row_number().over(w).alias("row"))
    sorted_release_df = release_df.select("*", row_number().over(w).alias("row"))
    return sorted_prod_df, sorted_release_df


def sort_columns(prod_df, release_df):
    """"""
    # Ensure both DataFrames have the same columns
    common_columns = set(prod_df.columns)
    for x in common_columns:
        if x not in release_df.columns:
            raise Exception(
                f"{x} is present in prod_df but does not exist in release_df"
            )
    # Sort both DataFrames to ensure consistent order
    prod_df = prod_df.orderBy(*common_columns)
    release_df = release_df.orderBy(*common_columns)
    return prod_df, release_df


def format_numeric_fields(df):
    """"""
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
