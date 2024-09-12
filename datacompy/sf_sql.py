#
# Copyright 2024 Capital One Services, LLC
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
Compare two Snowpark SQL DataFrames and Snowflake tables.

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd
import snowflake.snowpark as sp
from ordered_set import OrderedSet
from snowflake.snowpark import Window
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.functions import (
    abs,
    col,
    concat,
    contains,
    is_null,
    lit,
    monotonically_increasing_id,
    row_number,
    trim,
    when,
)

from datacompy.base import BaseCompare
from datacompy.spark.sql import decimal_comparator

LOG = logging.getLogger(__name__)


NUMERIC_SNOWPARK_TYPES = [
    "tinyint",
    "smallint",
    "int",
    "bigint",
    "float",
    "double",
    decimal_comparator(),
]


class SFTableCompare(BaseCompare):
    """Comparison class to be used to compare whether two Snowpark dataframes are equal.

    df1 and df2 can refer to either a Snowpark dataframe or the name of a valid Snowflake table.
    The data structures which df1 and df2 represent must contain all of the join_columns,
    with unique column names. Differences between values are compared to
    abs_tol + rel_tol * abs(df2['value']).

    Parameters
    ----------
    session: snowflake.snowpark.session
        Session with the required connection session info for user and targeted tables
    df1 : Union[str, sp.Dataframe]
        First table to check, provided either as the table's name or as a Snowpark DF.
    df2 : Union[str, sp.Dataframe]
        Second table to check, provided either as the table's name or as a Snowpark DF.
    join_columns : list or str, optional
        Column(s) to join dataframes on.  If a string is passed in, that one
        column will be used.
    abs_tol : float, optional
        Absolute tolerance between two values.
    rel_tol : float, optional
        Relative tolerance between two values.
    ignore_spaces : bool, optional
        Flag to strip whitespace (including newlines) from string columns (including any join
        columns).
    cast_join_columns_upper : bool, optional
        Uppercases joined columns, enabled by default as columns are by default uppercased in
        Snowpark.

    Attributes
    ----------
    df1_unq_rows : sp.DataFrame
        All records that are only in df1 (based on a join on join_columns)
    df2_unq_rows : sp.DataFrame
        All records that are only in df2 (based on a join on join_columns)
    """

    def __init__(
        self,
        session: sp.Session,
        df1: Union[str, sp.DataFrame],
        df2: Union[str, sp.DataFrame],
        join_columns: Optional[Union[List[str], str]],
        abs_tol: float = 0,
        rel_tol: float = 0,
        ignore_spaces: bool = False,
        cast_join_columns_upper=True,
    ) -> None:
        if join_columns is None:
            errmsg = "join_columns cannot be None"
            raise ValueError(errmsg)
        elif not join_columns:
            errmsg = "join_columns is empty"
            raise ValueError(errmsg)
        elif isinstance(join_columns, (str, int, float)):
            self.join_columns = (
                [str(join_columns).upper()]
                if cast_join_columns_upper
                else [str(join_columns)]
            )
        else:
            self.join_columns = (
                [str(col).upper() for col in cast(List[str], join_columns)]
                if cast_join_columns_upper
                else [str(col) for col in cast(List[str], join_columns)]
            )

        self._any_dupes: bool = False
        self.session = session
        self.df1 = df1
        self.df2 = df2
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.ignore_spaces = ignore_spaces
        self.df1_unq_rows: sp.DataFrame
        self.df2_unq_rows: sp.DataFrame
        self.intersect_rows: sp.DataFrame
        self.column_stats: List[Dict[str, Any]] = []
        self._compare(ignore_spaces=ignore_spaces)

    @property
    def df1(self) -> sp.DataFrame:
        """Get the first dataframe."""
        return self._df1

    @df1.setter
    def df1(self, df1: Union[str, sp.DataFrame]) -> None:
        """Check that df1 is either a Snowpark DF or the name of a valid Snowflake table."""
        if isinstance(df1, str):
            table_name = [table_comp.upper() for table_comp in df1.split(".")]
            if len(table_name) == 3:
                self.df1_name = table_name[2]
            else:
                errmsg = f"{df1} is not a valid table name. Be sure to include the target db and schema."
                raise ValueError(errmsg)
            self._df1 = self.session.table(df1)
        else:
            self._df1 = df1
            self.df1_name = "DF1"
        self._validate_dataframe(self.df1, self.df1_name)

    @property
    def df2(self) -> sp.DataFrame:
        """Get the second dataframe."""
        return self._df2

    @df2.setter
    def df2(self, df2: Union[str, sp.DataFrame]) -> None:
        """Check that df2 is either a Snowpark DF or the name of a valid Snowflake table."""
        if isinstance(df2, str):
            table_name = [table_comp.upper() for table_comp in df2.split(".")]
            if len(table_name) == 3:
                self.df2_name = table_name[2]
            else:
                errmsg = f"{df2} is not a valid table name. Be sure to include the target db and schema."
                raise ValueError(errmsg)
            self._df2 = self.session.table(df2)
        else:
            self._df2 = df2
            self.df2_name = "DF2"
        self._validate_dataframe(self.df2, self.df2_name)

    def _validate_dataframe(self, df: sp.DataFrame, df_name: str) -> None:
        """Validate the provided Snowpark dataframe.

        The dataframe can either be a standalone Snowpark dataframe or a representative
        of a Snowflake table - in the latter case we check that the table it represents
        is a valid table by forcing a collection.

        Parameters
        ----------
        df : sp.DataFrame
            Snowpark Dataframe (either directly instantiated or as a Snowpark Table object).
        """
        if not isinstance(df, sp.DataFrame):
            raise TypeError(f"{df_name} must be a valid sp.Dataframe")
        # Check that the dataframe actually exists by forcing collection
        df.columns  # noqa: B018
        # Check if join_columns are present in the dataframe
        if not set(self.join_columns).issubset(set(df.columns)):
            raise ValueError(f"{df_name} must have all columns from join_columns")

        if df.drop_duplicates(self.join_columns).count() < df.count():
            self._any_dupes = True

    def _compare(self, ignore_spaces: bool) -> None:
        """Actually run the comparison.

        This method will log out information about what is different between
        the two dataframes.
        """
        LOG.info(f"Number of columns in common: {len(self.intersect_columns())}")
        LOG.debug("Checking column overlap")
        for column in self.df1_unq_columns():
            LOG.info(f"Column in df1 and not in df2: {column}")
        LOG.info(
            f"Number of columns in df1 and not in df2: {len(self.df1_unq_columns())}"
        )
        for column in self.df2_unq_columns():
            LOG.info(f"Column in df2 and not in df1: {column}")
        LOG.info(
            f"Number of columns in df2 and not in df1: {len(self.df2_unq_columns())}"
        )
        LOG.debug("Merging dataframes")
        self._dataframe_merge(ignore_spaces)
        self._intersect_compare(ignore_spaces)
        if self.matches():
            LOG.info("df1 matches df2")
        else:
            LOG.info("df1 does not match df2")

    def df1_unq_columns(self) -> OrderedSet[str]:
        """Get columns that are unique to df1."""
        return cast(
            OrderedSet[str], OrderedSet(self.df1.columns) - OrderedSet(self.df2.columns)
        )

    def df2_unq_columns(self) -> OrderedSet[str]:
        """Get columns that are unique to df2."""
        return cast(
            OrderedSet[str], OrderedSet(self.df2.columns) - OrderedSet(self.df1.columns)
        )

    def intersect_columns(self) -> OrderedSet[str]:
        """Get columns that are shared between the two dataframes."""
        return OrderedSet(self.df1.columns) & OrderedSet(self.df2.columns)

    def _dataframe_merge(self, ignore_spaces: bool) -> None:
        """Merge df1 to df2 on the join columns.

        Gets df1 - df2, df2 - df1, and df1 & df2
        joining on the ``join_columns``.
        """
        LOG.debug("Outer joining")

        df1 = self.df1
        df2 = self.df2
        temp_join_columns = deepcopy(self.join_columns)

        if self._any_dupes:
            LOG.debug("Duplicate rows found, deduping by order of remaining fields")
            # setting internal index
            LOG.info("Adding internal index to dataframes")
            df1 = df1.withColumn("__index", monotonically_increasing_id())
            df2 = df2.withColumn("__index", monotonically_increasing_id())

            # Create order column for uniqueness of match
            order_column = temp_column_name(df1, df2)
            df1 = df1.join(
                _generate_id_within_group(df1, temp_join_columns, order_column),
                on="__index",
                how="inner",
            ).drop("__index")
            df2 = df2.join(
                _generate_id_within_group(df2, temp_join_columns, order_column),
                on="__index",
                how="inner",
            ).drop("__index")
            temp_join_columns.append(order_column)

            # drop index
            LOG.info("Dropping internal index")
            df1 = df1.drop("__index")
            df2 = df2.drop("__index")

        if ignore_spaces:
            for column in self.join_columns:
                if "string" in next(
                    dtype for name, dtype in df1.dtypes if name == column
                ):
                    df1 = df1.withColumn(column, trim(col(column)))
                if "string" in next(
                    dtype for name, dtype in df2.dtypes if name == column
                ):
                    df2 = df2.withColumn(column, trim(col(column)))

        df1 = df1.with_column("merge", lit(True))
        df2 = df2.with_column("merge", lit(True))

        for c in df1.columns:
            df1 = df1.withColumnRenamed(c, c + "_" + self.df1_name)
        for c in df2.columns:
            df2 = df2.withColumnRenamed(c, c + "_" + self.df2_name)

        # NULL SAFE Outer join, not possible with Snowpark Dataframe join
        df1.createOrReplaceTempView("df1")
        df2.createOrReplaceTempView("df2")
        on = " and ".join(
            [
                f"EQUAL_NULL(df1.{c}_{self.df1_name}, df2.{c}_{self.df2_name})"
                for c in temp_join_columns
            ]
        )
        outer_join = self.session.sql(
            """
        SELECT * FROM
        df1 FULL OUTER JOIN df2
        ON
        """
            + on
        )
        # Create join indicator
        outer_join = outer_join.with_column(
            "_merge",
            when(
                outer_join[f"MERGE_{self.df1_name}"]
                & outer_join[f"MERGE_{self.df2_name}"],
                lit("BOTH"),
            )
            .when(
                outer_join[f"MERGE_{self.df1_name}"]
                & outer_join[f"MERGE_{self.df2_name}"].is_null(),
                lit("LEFT_ONLY"),
            )
            .when(
                outer_join[f"MERGE_{self.df1_name}"].is_null()
                & outer_join[f"MERGE_{self.df2_name}"],
                lit("RIGHT_ONLY"),
            ),
        )

        df1 = df1.drop(f"MERGE_{self.df1_name}")
        df2 = df2.drop(f"MERGE_{self.df2_name}")

        # Clean up temp columns for duplicate row matching
        if self._any_dupes:
            outer_join = outer_join.select_expr(
                f"* EXCLUDE ({order_column}_{self.df1_name}, {order_column}_{self.df2_name})"
            )
            df1 = df1.drop(f"{order_column}_{self.df1_name}")
            df2 = df2.drop(f"{order_column}_{self.df2_name}")

        # Capitalization required - clean up
        df1_cols = get_merged_columns(df1, outer_join, self.df1_name)
        df2_cols = get_merged_columns(df2, outer_join, self.df2_name)

        LOG.debug("Selecting df1 unique rows")
        self.df1_unq_rows = outer_join[outer_join["_merge"] == "LEFT_ONLY"][df1_cols]

        LOG.debug("Selecting df2 unique rows")
        self.df2_unq_rows = outer_join[outer_join["_merge"] == "RIGHT_ONLY"][df2_cols]
        LOG.info(f"Number of rows in df1 and not in df2: {self.df1_unq_rows.count()}")
        LOG.info(f"Number of rows in df2 and not in df1: {self.df2_unq_rows.count()}")

        LOG.debug("Selecting intersecting rows")
        self.intersect_rows = outer_join[outer_join["_merge"] == "BOTH"]
        LOG.info(
            f"Number of rows in df1 and df2 (not necessarily equal): {self.intersect_rows.count()}"
        )
        self.intersect_rows = self.intersect_rows.cache_result()

    def _intersect_compare(self, ignore_spaces: bool) -> None:
        """Run the comparison on the intersect dataframe.

        This loops through all columns that are shared between df1 and df2, and
        creates a column column_match which is True for matches, False
        otherwise.
        """
        LOG.debug("Comparing intersection")
        max_diff: float
        null_diff: int
        row_cnt = self.intersect_rows.count()
        for column in self.intersect_columns():
            if column in self.join_columns:
                match_cnt = row_cnt
                col_match = ""
                max_diff = 0
                null_diff = 0
            else:
                col_1 = column + "_" + self.df1_name
                col_2 = column + "_" + self.df2_name
                col_match = column + "_MATCH"
                self.intersect_rows = columns_equal(
                    self.intersect_rows,
                    col_1,
                    col_2,
                    col_match,
                    self.rel_tol,
                    self.abs_tol,
                    ignore_spaces,
                )
                match_cnt = (
                    self.intersect_rows.select(col_match)
                    .where(col(col_match) == True)  # noqa: E712
                    .count()
                )
                max_diff = calculate_max_diff(
                    self.intersect_rows,
                    col_1,
                    col_2,
                )
                null_diff = calculate_null_diff(self.intersect_rows, col_1, col_2)

            if row_cnt > 0:
                match_rate = float(match_cnt) / row_cnt
            else:
                match_rate = 0
            LOG.info(f"{column}: {match_cnt} / {row_cnt} ({match_rate:.2%}) match")

            col1_dtype, _ = _get_column_dtypes(self.df1, column, column)
            col2_dtype, _ = _get_column_dtypes(self.df2, column, column)

            self.column_stats.append(
                {
                    "column": column,
                    "match_column": col_match,
                    "match_cnt": match_cnt,
                    "unequal_cnt": row_cnt - match_cnt,
                    "dtype1": str(col1_dtype),
                    "dtype2": str(col2_dtype),
                    "all_match": all(
                        (
                            col1_dtype == col2_dtype,
                            row_cnt == match_cnt,
                        )
                    ),
                    "max_diff": max_diff,
                    "null_diff": null_diff,
                }
            )

    def all_columns_match(self) -> bool:
        """Whether the columns all match in the dataframes.

        Returns
        -------
        bool
        True if all columns in df1 are in df2 and vice versa
        """
        return self.df1_unq_columns() == self.df2_unq_columns() == set()

    def all_rows_overlap(self) -> bool:
        """Whether the rows are all present in both dataframes.

        Returns
        -------
        bool
            True if all rows in df1 are in df2 and vice versa (based on
            existence for join option)
        """
        return self.df1_unq_rows.count() == self.df2_unq_rows.count() == 0

    def count_matching_rows(self) -> int:
        """Count the number of rows match (on overlapping fields).

        Returns
        -------
        int
            Number of matching rows
        """
        conditions = []
        match_columns = []
        for column in self.intersect_columns():
            if column not in self.join_columns:
                match_columns.append(column + "_MATCH")
                conditions.append(f"{column}_MATCH = True")
        if len(conditions) > 0:
            match_columns_count = self.intersect_rows.filter(
                " and ".join(conditions)
            ).count()
        else:
            match_columns_count = 0
        return match_columns_count

    def intersect_rows_match(self) -> bool:
        """Check whether the intersect rows all match."""
        actual_length = self.intersect_rows.count()
        return self.count_matching_rows() == actual_length

    def matches(self, ignore_extra_columns: bool = False) -> bool:
        """Return True or False if the dataframes match.

        Parameters
        ----------
        ignore_extra_columns : bool
            Ignores any columns in one dataframe and not in the other.

        Returns
        -------
        bool
            True or False if the dataframes match.
        """
        return not (
            (not ignore_extra_columns and not self.all_columns_match())
            or not self.all_rows_overlap()
            or not self.intersect_rows_match()
        )

    def subset(self) -> bool:
        """Return True if dataframe 2 is a subset of dataframe 1.

        Dataframe 2 is considered a subset if all of its columns are in
        dataframe 1, and all of its rows match rows in dataframe 1 for the
        shared columns.

        Returns
        -------
        bool
            True if dataframe 2 is a subset of dataframe 1.
        """
        return not (
            self.df2_unq_columns() != set()
            or self.df2_unq_rows.count() != 0
            or not self.intersect_rows_match()
        )

    def sample_mismatch(
        self, column: str, sample_count: int = 10, for_display: bool = False
    ) -> sp.DataFrame:
        """Return sample mismatches.

        Gets a sub-dataframe which contains the identifying
        columns, and df1 and df2 versions of the column.

        Parameters
        ----------
        column : str
            The raw column name (i.e. without ``_df1`` appended)
        sample_count : int, optional
            The number of sample records to return.  Defaults to 10.
        for_display : bool, optional
            Whether this is just going to be used for display (overwrite the
            column names)

        Returns
        -------
        sp.DataFrame
            A sample of the intersection dataframe, containing only the
            "pertinent" columns, for rows that don't match on the provided
            column.
        """
        row_cnt = self.intersect_rows.count()
        col_match = self.intersect_rows.select(column + "_MATCH")
        match_cnt = col_match.where(
            col(column + "_MATCH") == True  # noqa: E712
        ).count()
        sample_count = min(sample_count, row_cnt - match_cnt)
        sample = (
            self.intersect_rows.where(col(column + "_MATCH") == False)  # noqa: E712
            .drop(column + "_MATCH")
            .limit(sample_count)
        )

        for c in self.join_columns:
            sample = sample.withColumnRenamed(c + "_" + self.df1_name, c)

        return_cols = [
            *self.join_columns,
            column + "_" + self.df1_name,
            column + "_" + self.df2_name,
        ]
        to_return = sample.select(return_cols)

        if for_display:
            return to_return.toDF(
                *[
                    *self.join_columns,
                    column + " (" + self.df1_name + ")",
                    column + " (" + self.df2_name + ")",
                ]
            )
        return to_return

    def all_mismatch(self, ignore_matching_cols: bool = False) -> sp.DataFrame:
        """Get all rows with any columns that have a mismatch.

        Returns all df1 and df2 versions of the columns and join
        columns.

        Parameters
        ----------
        ignore_matching_cols : bool, optional
            Whether showing the matching columns in the output or not. The default is False.

        Returns
        -------
        sp.DataFrame
            All rows of the intersection dataframe, containing any columns, that don't match.
        """
        match_list = []
        return_list = []
        for c in self.intersect_rows.columns:
            if c.endswith("_MATCH"):
                orig_col_name = c[:-6]

                col_comparison = columns_equal(
                    self.intersect_rows,
                    orig_col_name + "_" + self.df1_name,
                    orig_col_name + "_" + self.df2_name,
                    c,
                    self.rel_tol,
                    self.abs_tol,
                    self.ignore_spaces,
                )

                if not ignore_matching_cols or (
                    ignore_matching_cols
                    and col_comparison.select(c)
                    .where(col(c) == False)  # noqa: E712
                    .count()
                    > 0
                ):
                    LOG.debug(f"Adding column {orig_col_name} to the result.")
                    match_list.append(c)
                    return_list.extend(
                        [
                            orig_col_name + "_" + self.df1_name,
                            orig_col_name + "_" + self.df2_name,
                        ]
                    )
                elif ignore_matching_cols:
                    LOG.debug(
                        f"Column {orig_col_name} is equal in df1 and df2. It will not be added to the result."
                    )

        mm_rows = self.intersect_rows.withColumn(
            "match_array", concat(*match_list)
        ).where(contains(col("match_array"), lit("false")))

        for c in self.join_columns:
            mm_rows = mm_rows.withColumnRenamed(c + "_" + self.df1_name, c)

        return mm_rows.select(self.join_columns + return_list)

    def report(
        self,
        sample_count: int = 10,
        column_count: int = 10,
        html_file: Optional[str] = None,
    ) -> str:
        """Return a string representation of a report.

        The representation can
        then be printed or saved to a file.

        Parameters
        ----------
        sample_count : int, optional
            The number of sample records to return.  Defaults to 10.

        column_count : int, optional
            The number of columns to display in the sample records output.  Defaults to 10.

        html_file : str, optional
            HTML file name to save report output to. If ``None`` the file creation will be skipped.

        Returns
        -------
        str
            The report, formatted kinda nicely.
        """
        # Header
        report = render("header.txt")
        df_header = pd.DataFrame(
            {
                "DataFrame": [self.df1_name, self.df2_name],
                "Columns": [len(self.df1.columns), len(self.df2.columns)],
                "Rows": [self.df1.count(), self.df2.count()],
            }
        )
        report += df_header[["DataFrame", "Columns", "Rows"]].to_string()
        report += "\n\n"

        # Column Summary
        report += render(
            "column_summary.txt",
            len(self.intersect_columns()),
            len(self.df1_unq_columns()),
            len(self.df2_unq_columns()),
            self.df1_name,
            self.df2_name,
        )

        # Row Summary
        match_on = ", ".join(self.join_columns)
        report += render(
            "row_summary.txt",
            match_on,
            self.abs_tol,
            self.rel_tol,
            self.intersect_rows.count(),
            self.df1_unq_rows.count(),
            self.df2_unq_rows.count(),
            self.intersect_rows.count() - self.count_matching_rows(),
            self.count_matching_rows(),
            self.df1_name,
            self.df2_name,
            "Yes" if self._any_dupes else "No",
        )

        # Column Matching
        report += render(
            "column_comparison.txt",
            len([col for col in self.column_stats if col["unequal_cnt"] > 0]),
            len([col for col in self.column_stats if col["unequal_cnt"] == 0]),
            sum(col["unequal_cnt"] for col in self.column_stats),
        )

        match_stats = []
        match_sample = []
        any_mismatch = False
        for column in self.column_stats:
            if not column["all_match"]:
                any_mismatch = True
                match_stats.append(
                    {
                        "Column": column["column"],
                        f"{self.df1_name} dtype": column["dtype1"],
                        f"{self.df2_name} dtype": column["dtype2"],
                        "# Unequal": column["unequal_cnt"],
                        "Max Diff": column["max_diff"],
                        "# Null Diff": column["null_diff"],
                    }
                )
                if column["unequal_cnt"] > 0:
                    match_sample.append(
                        self.sample_mismatch(
                            column["column"], sample_count, for_display=True
                        )
                    )

        if any_mismatch:
            report += "Columns with Unequal Values or Types\n"
            report += "------------------------------------\n"
            report += "\n"
            df_match_stats = pd.DataFrame(match_stats)
            df_match_stats.sort_values("Column", inplace=True)
            # Have to specify again for sorting
            report += df_match_stats[
                [
                    "Column",
                    f"{self.df1_name} dtype",
                    f"{self.df2_name} dtype",
                    "# Unequal",
                    "Max Diff",
                    "# Null Diff",
                ]
            ].to_string()
            report += "\n\n"

            if sample_count > 0:
                report += "Sample Rows with Unequal Values\n"
                report += "-------------------------------\n"
                report += "\n"
                for sample in match_sample:
                    report += sample.toPandas().to_string()
                    report += "\n\n"

        if min(sample_count, self.df1_unq_rows.count()) > 0:
            report += (
                f"Sample Rows Only in {self.df1_name} (First {column_count} Columns)\n"
            )
            report += (
                f"---------------------------------------{'-' * len(self.df1_name)}\n"
            )
            report += "\n"
            columns = self.df1_unq_rows.columns[:column_count]
            unq_count = min(sample_count, self.df1_unq_rows.count())
            report += (
                self.df1_unq_rows.limit(unq_count)
                .select(columns)
                .toPandas()
                .to_string()
            )
            report += "\n\n"

        if min(sample_count, self.df2_unq_rows.count()) > 0:
            report += (
                f"Sample Rows Only in {self.df2_name} (First {column_count} Columns)\n"
            )
            report += (
                f"---------------------------------------{'-' * len(self.df2_name)}\n"
            )
            report += "\n"
            columns = self.df2_unq_rows.columns[:column_count]
            unq_count = min(sample_count, self.df2_unq_rows.count())
            report += (
                self.df2_unq_rows.limit(unq_count)
                .select(columns)
                .toPandas()
                .to_string()
            )
            report += "\n\n"

        if html_file:
            html_report = report.replace("\n", "<br>").replace(" ", "&nbsp;")
            html_report = f"<pre>{html_report}</pre>"
            with open(html_file, "w") as f:
                f.write(html_report)

        return report


def render(filename: str, *fields: Union[int, float, str]) -> str:
    """Render out an individual template.

    This basically just reads in a
    template file, and applies ``.format()`` on the fields.

    Parameters
    ----------
    filename : str
        The file that contains the template.  Will automagically prepend the
        templates directory before opening
    fields : list
        Fields to be rendered out in the template

    Returns
    -------
    str
        The fully rendered out file.
    """
    this_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(this_dir, "templates", filename)) as file_open:
        return file_open.read().format(*fields)


def columns_equal(
    dataframe: sp.DataFrame,
    col_1: str,
    col_2: str,
    col_match: str,
    rel_tol: float = 0,
    abs_tol: float = 0,
    ignore_spaces: bool = False,
) -> sp.DataFrame:
    """Compare two columns from a dataframe.

    Returns a True/False series with the same index as column 1.

    - Two nulls (np.nan) will evaluate to True.
    - A null and a non-null value will evaluate to False.
    - Numeric values will use the relative and absolute tolerances.
    - Decimal values (decimal.Decimal) will attempt to be converted to floats
      before comparing
    - Non-numeric values (i.e. where np.isclose can't be used) will just
      trigger True on two nulls or exact matches.

    Parameters
    ----------
    dataframe: sp.DataFrame
        DataFrame to do comparison on
    col_1 : str
        The first column to look at
    col_2 : str
        The second column
    col_match : str
        The matching column denoting if the compare was a match or not
    rel_tol : float, optional
        Relative tolerance
    abs_tol : float, optional
        Absolute tolerance
    ignore_spaces : bool, optional
        Flag to strip whitespace (including newlines) from string columns

    Returns
    -------
    sp.DataFrame
        A column of boolean values are added.  True == the values match, False == the
        values don't match.
    """
    base_dtype, compare_dtype = _get_column_dtypes(dataframe, col_1, col_2)
    if _is_comparable(base_dtype, compare_dtype):
        if (base_dtype in NUMERIC_SNOWPARK_TYPES) and (
            compare_dtype in NUMERIC_SNOWPARK_TYPES
        ):  # numeric tolerance comparison
            dataframe = dataframe.withColumn(
                col_match,
                when(
                    (col(col_1).eqNullSafe(col(col_2)))
                    | (
                        abs(col(col_1) - col(col_2))
                        <= lit(abs_tol) + (lit(rel_tol) * abs(col(col_2)))
                    ),
                    # corner case of col1 != NaN and col2 == Nan returns True incorrectly
                    when(
                        (is_null(col(col_1)) == False)  # noqa: E712
                        & (is_null(col(col_2)) == True),  # noqa: E712
                        lit(False),
                    ).otherwise(lit(True)),
                ).otherwise(lit(False)),
            )
        else:  # non-numeric comparison
            if ignore_spaces:
                when_clause = trim(col(col_1)).eqNullSafe(trim(col(col_2)))
            else:
                when_clause = col(col_1).eqNullSafe(col(col_2))

            dataframe = dataframe.withColumn(
                col_match,
                when(when_clause, lit(True)).otherwise(lit(False)),
            )
    else:
        LOG.debug(
            f"Skipping {col_1}({base_dtype}) and {col_2}({compare_dtype}), columns are not comparable"
        )
        dataframe = dataframe.withColumn(col_match, lit(False))
    return dataframe


def get_merged_columns(
    original_df: sp.DataFrame, merged_df: sp.DataFrame, suffix: str
) -> List[str]:
    """Get the columns from an original dataframe, in the new merged dataframe.

    Parameters
    ----------
    original_df : sp.DataFrame
        The original, pre-merge dataframe
    merged_df : sp.DataFrame
        Post-merge with another dataframe, with suffixes added in.
    suffix : str
        What suffix was used to distinguish when the original dataframe was
        overlapping with the other merged dataframe.

    Returns
    -------
    List[str]
        Column list of the original dataframe pre suffix
    """
    columns = []
    for column in original_df.columns:
        if column in merged_df.columns:
            columns.append(column)
        elif f"{column}_{suffix}" in merged_df.columns:
            columns.append(f"{column}_{suffix}")
        else:
            raise ValueError("Column not found: %s", column)
    return columns


def calculate_max_diff(dataframe: sp.DataFrame, col_1: str, col_2: str) -> float:
    """Get a maximum difference between two columns.

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
    float
        max diff
    """
    # Attempting to coalesce maximum diff for non-numeric results in error, if error return 0 max diff.
    try:
        diff = dataframe.select(
            (col(col_1).astype("float") - col(col_2).astype("float")).alias("diff")
        )
        abs_diff = diff.select(abs(col("diff")).alias("abs_diff"))
        max_diff: float = (
            abs_diff.where(is_null(col("abs_diff")) == False)  # noqa: E712
            .agg({"abs_diff": "max"})
            .collect()[0][0]
        )
    except SnowparkSQLException:
        return None

    if pd.isna(max_diff) or pd.isnull(max_diff) or max_diff is None:
        return 0
    else:
        return max_diff


def calculate_null_diff(dataframe: sp.DataFrame, col_1: str, col_2: str) -> int:
    """Get the null differences between two columns.

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
    int
        null diff
    """
    nulls_df = dataframe.withColumn(
        "col_1_null",
        when(col(col_1).isNull() == True, lit(True)).otherwise(  # noqa: E712
            lit(False)
        ),
    )
    nulls_df = nulls_df.withColumn(
        "col_2_null",
        when(col(col_2).isNull() == True, lit(True)).otherwise(  # noqa: E712
            lit(False)
        ),
    ).select(["col_1_null", "col_2_null"])

    # (not a and b) or (a and not b)
    null_diff = nulls_df.where(
        ((col("col_1_null") == False) & (col("col_2_null") == True))  # noqa: E712
        | ((col("col_1_null") == True) & (col("col_2_null") == False))  # noqa: E712
    ).count()

    if pd.isna(null_diff) or pd.isnull(null_diff) or null_diff is None:
        return 0
    else:
        return null_diff


def _generate_id_within_group(
    dataframe: sp.DataFrame, join_columns: List[str], order_column_name: str
) -> sp.DataFrame:
    """Generate an ID column that can be used to deduplicate identical rows.

    The series generated
    is the order within a unique group, and it handles nulls. Requires a ``__index`` column.

    Parameters
    ----------
    dataframe : sp.DataFrame
        The dataframe to operate on
    join_columns : list
        List of strings which are the join columns
    order_column_name: str
        The name of the ``row_number`` column name

    Returns
    -------
    sp.DataFrame
        Original dataframe with the ID column that's unique in each group
    """
    default_value = "DATACOMPY_NULL"
    null_check = False
    default_check = False
    for c in join_columns:
        if dataframe.where(col(c).isNull()).limit(1).collect():
            null_check = True
            break
    for c in [
        column for column, type in dataframe[join_columns].dtypes if "string" in type
    ]:
        if dataframe.where(col(c).isin(default_value)).limit(1).collect():
            default_check = True
            break

    if null_check:
        if default_check:
            raise ValueError(f"{default_value} was found in your join columns")

        return (
            dataframe.select(
                *(col(c).cast("string").alias(c) for c in join_columns + ["__index"])  # noqa: RUF005
            )
            .fillna(default_value)
            .withColumn(
                order_column_name,
                row_number().over(Window.orderBy("__index").partitionBy(join_columns))
                - 1,
            )
            .select(["__index", order_column_name])
        )
    else:
        return (
            dataframe.select(join_columns + ["__index"])  # noqa: RUF005
            .withColumn(
                order_column_name,
                row_number().over(Window.orderBy("__index").partitionBy(join_columns))
                - 1,
            )
            .select(["__index", order_column_name])
        )


def _get_column_dtypes(
    dataframe: sp.DataFrame, col_1: "str", col_2: "str"
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
    base_dtype = next(d[1] for d in dataframe.dtypes if d[0] == col_1)
    compare_dtype = next(d[1] for d in dataframe.dtypes if d[0] == col_2)
    return base_dtype, compare_dtype


def _is_comparable(type1: str, type2: str) -> bool:
    """Check if two SnowPark data types can be safely compared.

    Two data types are considered comparable if any of the following apply:
        1. Both data types are the same
        2. Both data types are numeric

    Parameters
    ----------
    type1 : str
        A string representation of a Snowpark data type
    type2 : str
        A string representation of a Snowpark data type

    Returns
    -------
    bool
        True if both data types are comparable
    """
    return (
        type1 == type2
        or (type1 in NUMERIC_SNOWPARK_TYPES and type2 in NUMERIC_SNOWPARK_TYPES)
        or ("string" in type1 and type2 == "date")
        or (type1 == "date" and "string" in type2)
        or ("string" in type1 and type2 == "timestamp")
        or (type1 == "timestamp" and "string" in type2)
    )


def temp_column_name(*dataframes) -> str:
    """Get a temp column name that isn't included in columns of any dataframes.

    Parameters
    ----------
    dataframes : list of DataFrames
        The DataFrames to create a temporary column name for

    Returns
    -------
    str
        String column name that looks like '_temp_x' for some integer x
    """
    i = 0
    columns = []
    for dataframe in dataframes:
        columns = columns + list(dataframe.columns)
    columns = set(columns)

    while True:
        temp_column = f"_TEMP_{i}"
        unique = True

        if temp_column in columns:
            i += 1
            unique = False
        if unique:
            return temp_column
