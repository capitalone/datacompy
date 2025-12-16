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
Compare two PySpark SQL DataFrames.

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import pandas as pd
import pyspark.sql
import pyspark.sql.functions as F
from ordered_set import OrderedSet
from pyspark.sql import Window
from pyspark.sql.connect.dataframe import DataFrame

from datacompy.base import (
    BaseCompare,
    df_to_str,
    get_column_tolerance,
    render,
    save_html_report,
    temp_column_name,
    validate_tolerance_parameter,
)
from datacompy.comparator import (
    SparkArrayLikeComparator,
    SparkNumericComparator,
    SparkStringComparator,
)
from datacompy.comparator.base import BaseComparator

LOG = logging.getLogger(__name__)


_SPARK_DEFAULT_COMPARATORS = [
    SparkArrayLikeComparator(),
    SparkNumericComparator(),
    SparkStringComparator(),
]


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


class SparkSQLCompare(BaseCompare):
    """Comparison class to be used to compare whether two Spark SQL dataframes are equal.

    Both df1 and df2 should be dataframes containing all of the join_columns,
    with unique column names. Differences between values are compared to
    abs_tol + rel_tol * abs(df2['value']).

    Parameters
    ----------
    spark_session : pyspark.sql.SparkSession
        A ``SparkSession`` to be used to execute Spark commands in the comparison.
    df1 : pyspark.sql.DataFrame
        First dataframe to check
    df2 : pyspark.sql.DataFrame
        Second dataframe to check
    join_columns : list or str, optional
        Column(s) to join dataframes on.  If a string is passed in, that one
        column will be used.
    abs_tol : float or dict, optional
        Absolute tolerance between two values. Can be either a float value applied to all columns,
        or a dictionary mapping column names to specific tolerance values. The special key "default"
        in the dictionary specifies the tolerance for columns not explicitly listed.
    rel_tol : float or dict, optional
        Relative tolerance between two values. Can be either a float value applied to all columns,
        or a dictionary mapping column names to specific tolerance values. The special key "default"
        in the dictionary specifies the tolerance for columns not explicitly listed.
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
    custom_comparators : list of ``BaseComparator``, optional
        A list of custom comparator classes to use to compare columns.
    """

    def __init__(
        self,
        spark_session: "pyspark.sql.SparkSession",
        df1: "pyspark.sql.DataFrame",
        df2: "pyspark.sql.DataFrame",
        join_columns: List[str] | str,
        abs_tol: float | Dict[str, float] = 0,
        rel_tol: float | Dict[str, float] = 0,
        df1_name: str = "df1",
        df2_name: str = "df2",
        ignore_spaces: bool = False,
        ignore_case: bool = False,
        cast_column_names_lower: bool = True,
        custom_comparators: List[BaseComparator] | None = None,
    ) -> None:
        self.cast_column_names_lower = cast_column_names_lower
        self.custom_comparators = custom_comparators or []

        # Validate tolerance parameters first
        self._abs_tol_dict = validate_tolerance_parameter(
            abs_tol, "abs_tol", "lower" if cast_column_names_lower else "preserve"
        )
        self._rel_tol_dict = validate_tolerance_parameter(
            rel_tol, "rel_tol", "lower" if cast_column_names_lower else "preserve"
        )

        if isinstance(join_columns, str | int | float):
            self.join_columns = [
                (
                    str(join_columns).lower()
                    if self.cast_column_names_lower
                    else str(join_columns)
                )
            ]
        else:
            self.join_columns = [
                str(c).lower() if self.cast_column_names_lower else str(c)
                for c in join_columns
            ]

        self.spark_session = spark_session
        self._any_dupes: bool = False
        self.df1 = df1
        self.df2 = df2
        self.df1_name = df1_name
        self.df2_name = df2_name
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.ignore_spaces = ignore_spaces
        self.ignore_case = ignore_case
        self.df1_unq_rows: pyspark.sql.DataFrame
        self.df2_unq_rows: pyspark.sql.DataFrame
        self.intersect_rows: pyspark.sql.DataFrame
        self.column_stats: List = []
        self._compare(ignore_spaces=ignore_spaces, ignore_case=ignore_case)

    def _get_comparators(self) -> List[BaseComparator]:
        """Build and return the list of comparators to be used.

        Custom comparators are placed first, followed by the default ones.
        """
        return self.custom_comparators + _SPARK_DEFAULT_COMPARATORS

    @property
    def df1(self) -> "pyspark.sql.DataFrame":
        """Get the first dataframe."""
        return self._df1

    @df1.setter
    def df1(self, df1: "pyspark.sql.DataFrame") -> None:
        """Check that it is a dataframe and has the join columns."""
        self._df1 = df1
        self._validate_dataframe(
            "df1", cast_column_names_lower=self.cast_column_names_lower
        )

    @property
    def df2(self) -> "pyspark.sql.DataFrame":
        """Get the second dataframe."""
        return self._df2

    @df2.setter
    def df2(self, df2: "pyspark.sql.DataFrame") -> None:
        """Check that it is a dataframe and has the join columns."""
        self._df2 = df2
        self._validate_dataframe(
            "df2", cast_column_names_lower=self.cast_column_names_lower
        )

    def _validate_dataframe(
        self, index: str, cast_column_names_lower: bool = True
    ) -> None:
        """Check that it is a dataframe and has the join columns.

        Parameters
        ----------
        index : str
            The "index" of the dataframe - df1 or df2.
        cast_column_names_lower: bool, optional
            Boolean indicator that controls of column names will be cast into lower case

        Return
        ------
        None
        """
        dataframe = getattr(self, index)
        instances = (pyspark.sql.DataFrame, DataFrame)

        if not isinstance(dataframe, instances):
            raise TypeError(
                f"{index} must be a pyspark.sql.DataFrame or pyspark.sql.connect.dataframe.DataFrame (Spark 3.4.0 and above)"
            )

        if cast_column_names_lower:
            if index == "df1":
                self._df1 = dataframe.toDF(*[str(c).lower() for c in dataframe.columns])
            if index == "df2":
                self._df2 = dataframe.toDF(*[str(c).lower() for c in dataframe.columns])

        # Check if join_columns are present in the dataframe
        dataframe = getattr(self, index)  # refresh
        if not set(self.join_columns).issubset(set(dataframe.columns)):
            missing_cols = set(self.join_columns) - set(dataframe.columns)
            raise ValueError(
                f"{index} must have all columns from join_columns: {missing_cols}"
            )

        if len(set(dataframe.columns)) < len(dataframe.columns):
            raise ValueError(f"{index} must have unique column names")

        if (
            dataframe.drop_duplicates(subset=self.join_columns).count()
            < dataframe.count()
        ):
            self._any_dupes = True

    def _compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        """Actually run the comparison.

        This tries to run df1.equals(df2)
        first so that if they're truly equal we can tell.

        This method will log out information about what is different between
        the two dataframes, and will also return a boolean.
        """
        LOG.info(f"Number of columns in common: {len(self.intersect_columns())}")
        LOG.debug("Checking column overlap")
        for c in self.df1_unq_columns():
            LOG.info(f"Column in df1 and not in df2: {c}")
        LOG.info(
            f"Number of columns in df1 and not in df2: {len(self.df1_unq_columns())}"
        )
        for c in self.df2_unq_columns():
            LOG.info(f"Column in df2 and not in df1: {c}")
        LOG.info(
            f"Number of columns in df2 and not in df1: {len(self.df2_unq_columns())}"
        )

        LOG.debug("Merging dataframes")
        self._dataframe_merge(ignore_spaces)
        self._intersect_compare(ignore_spaces, ignore_case)

        if self.matches():
            LOG.info("df1 matches df2")
        else:
            LOG.info("df1 does not match df2")

    def df1_unq_columns(self) -> OrderedSet[str]:
        """Get columns that are unique to df1."""
        return OrderedSet(self.df1.columns) - OrderedSet(self.df2.columns)

    def df2_unq_columns(self) -> OrderedSet[str]:
        """Get columns that are unique to df2."""
        return OrderedSet(self.df2.columns) - OrderedSet(self.df1.columns)

    def intersect_columns(self) -> OrderedSet[str]:
        """Get columns that are shared between the two dataframes."""
        return OrderedSet(self.df1.columns) & OrderedSet(self.df2.columns)

    def _dataframe_merge(self, ignore_spaces: bool) -> None:
        """Merge df1 to df2 on the join columns.

        To get df1 - df2, df2 - df1 and df1 & df2.
        """
        LOG.debug("Outer joining")

        df1 = self.df1
        df2 = self.df2
        temp_join_columns = deepcopy(self.join_columns)

        if self._any_dupes:
            LOG.debug("Duplicate rows found, deduping by order of remaining fields")
            # setting internal index
            LOG.info("Adding internal index to dataframes")
            df1 = df1.withColumn("__index", F.monotonically_increasing_id())
            df2 = df2.withColumn("__index", F.monotonically_increasing_id())

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

        params = {"on": temp_join_columns}

        if ignore_spaces:
            for column in self.join_columns:
                if (
                    next(dtype for name, dtype in df1.dtypes if name == column)
                    == "string"
                ):
                    df1 = df1.withColumn(column, F.trim(F.col(column)))
                if (
                    next(dtype for name, dtype in df2.dtypes if name == column)
                    == "string"
                ):
                    df2 = df2.withColumn(column, F.trim(F.col(column)))

        df1_non_join_columns = OrderedSet(df1.columns) - OrderedSet(temp_join_columns)
        df2_non_join_columns = OrderedSet(df2.columns) - OrderedSet(temp_join_columns)

        df1 = df1.withColumnsRenamed(
            {c: f"{c}_{self.df1_name}" for c in df1_non_join_columns}
        )
        df2 = df2.withColumnsRenamed(
            {c: f"{c}_{self.df2_name}" for c in df2_non_join_columns}
        )

        # generate merge indicator
        df1 = df1.withColumn("_merge_left", F.lit(True))
        df2 = df2.withColumn("_merge_right", F.lit(True))

        df1 = df1.withColumnsRenamed(
            {c: f"{c}_{self.df1_name}" for c in temp_join_columns}
        )
        df2 = df2.withColumnsRenamed(
            {c: f"{c}_{self.df2_name}" for c in temp_join_columns}
        )

        # cache
        df1.cache()
        df2.cache()

        # NULL SAFE Outer join using ON
        df1.createOrReplaceTempView("df1")
        df2.createOrReplaceTempView("df2")
        on = " and ".join(
            [
                f"df1.`{c}_{self.df1_name}` <=> df2.`{c}_{self.df2_name}`"
                for c in params["on"]
            ]
        )
        outer_join = self.spark_session.sql(
            """
        SELECT * FROM
        df1 FULL OUTER JOIN df2
        ON
        """
            + on
        )

        outer_join = outer_join.withColumn("_merge", F.lit(None))  # initialize col

        # process merge indicator
        outer_join = outer_join.withColumn(
            "_merge",
            F.when(
                (outer_join["_merge_left"] == True)  # noqa: E712
                & (F.isnull(outer_join["_merge_right"])),
                "left_only",
            )
            .when(
                (F.isnull(outer_join["_merge_left"]))
                & (outer_join["_merge_right"] == True),  # noqa: E712
                "right_only",
            )
            .otherwise("both"),
        )

        # Clean up temp columns for duplicate row matching
        if self._any_dupes:
            outer_join = outer_join.drop(
                *[
                    f"{order_column}_{self.df1_name}",
                    f"{order_column}_{self.df2_name}",
                ],
            )
            df1 = df1.drop(
                *[
                    f"{order_column}_{self.df1_name}",
                    f"{order_column}_{self.df2_name}",
                ],
            )
            df2 = df2.drop(
                *[
                    f"{order_column}_{self.df1_name}",
                    f"{order_column}_{self.df2_name}",
                ],
            )

        df1_cols = get_merged_columns(df1, outer_join, self.df1_name)
        df2_cols = get_merged_columns(df2, outer_join, self.df2_name)

        LOG.debug("Selecting df1 unique rows")
        self.df1_unq_rows = outer_join[outer_join["_merge"] == "left_only"][df1_cols]

        LOG.debug("Selecting df2 unique rows")
        self.df2_unq_rows = outer_join[outer_join["_merge"] == "right_only"][df2_cols]

        LOG.info(f"Number of rows in df1 and not in df2: {self.df1_unq_rows.count()}")
        LOG.info(f"Number of rows in df2 and not in df1: {self.df2_unq_rows.count()}")

        LOG.debug("Selecting intersecting rows")
        self.intersect_rows = outer_join[outer_join["_merge"] == "both"]
        LOG.info(
            f"Number of rows in df1 and df2 (not necessarily equal): {self.intersect_rows.count()}"
        )
        # cache
        self.intersect_rows.cache()

    def _intersect_compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        """Run the comparison on the intersect dataframe.

        This loops through all columns that are shared between df1 and df2, and
        creates a column column_match which is True for matches, False
        otherwise.
        """
        LOG.debug("Comparing intersection")
        max_diff: float
        null_diff: int
        exprs = {}
        LOG.info("Generating expression columns for comparison")
        for column in self.intersect_columns():
            if column in self.join_columns:
                continue  # same as original: no per-column comparison for join cols

            col_1 = f"{column}_{self.df1_name}"
            col_2 = f"{column}_{self.df2_name}"
            col_match = f"{column}_match"

            exprs[col_match] = columns_equal(
                self.intersect_rows,
                col_1,
                col_2,
                rel_tol=get_column_tolerance(column, self._rel_tol_dict),
                abs_tol=get_column_tolerance(column, self._abs_tol_dict),
                ignore_spaces=ignore_spaces,
                ignore_case=ignore_case,
                comparators=self._get_comparators(),
            )

        self.intersect_rows = self.intersect_rows.withColumns(exprs)
        all_rows_count = self.intersect_rows.count()

        if exprs:
            agg_exprs = [
                F.sum(F.when(F.col(c) == True, 1).otherwise(0)).alias(f"{c}_count")  # noqa: E712
                for c in exprs
            ]
            match_counts = self.intersect_rows.agg(*agg_exprs).first()
        else:
            match_counts = {}

        # metric calculation
        for column in self.intersect_columns():
            col_1 = f"{column}_{self.df1_name}"
            col_2 = f"{column}_{self.df2_name}"
            col_match = f"{column}_match"

            if column in self.join_columns:
                if not self.only_join_columns():
                    row_cnt = all_rows_count
                else:
                    row_cnt = (
                        all_rows_count
                        + self.df1_unq_rows.count()
                        + self.df2_unq_rows.count()
                    )
                match_cnt = all_rows_count
                max_diff = 0
                null_diff = 0
            else:
                row_cnt = all_rows_count
                # Lookup pre-calculated col_match_count instead of querying,
                # with default 0 to avoid None for empty rows
                match_cnt = match_counts[f"{col_match}_count"] or 0
                max_diff = calculate_max_diff(self.intersect_rows, col_1, col_2)
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
                    "rel_tol": get_column_tolerance(column, self._rel_tol_dict),
                    "abs_tol": get_column_tolerance(column, self._abs_tol_dict),
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
                match_columns.append(column + "_match")
                conditions.append(f"`{column}_match` == True")
        if len(conditions) > 0:
            match_columns_count = self.intersect_rows.filter(
                " and ".join(conditions)
            ).count()
        else:
            match_columns_count = self.intersect_rows.count()
        return match_columns_count

    def intersect_rows_match(self) -> bool:
        """Check whether the intersect rows all match."""
        if self.intersect_rows.count() == 0:
            return False
        actual_length = self.intersect_rows.count()
        return self.count_matching_rows() == actual_length

    def matches(self, ignore_extra_columns: bool = False) -> bool:
        """Return True or False if the dataframes match.

        Parameters
        ----------
        ignore_extra_columns : bool
            Ignores any columns in one dataframe and not in the other.
        """
        return (
            (ignore_extra_columns or self.all_columns_match())
            and self.all_rows_overlap()
            and self.intersect_rows_match()
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
        return (
            self.df2_unq_columns() == set()
            and self.df2_unq_rows.count() == 0
            and self.intersect_rows_match()
        )

    def sample_mismatch(
        self, column: str, sample_count: int = 10, for_display: bool = False
    ) -> "pyspark.sql.DataFrame":
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
        pyspark.sql.DataFrame
            A sample of the intersection dataframe, containing only the
            "pertinent" columns, for rows that don't match on the provided
            column.
        """
        if not self.only_join_columns() and column not in self.join_columns:
            row_cnt = self.intersect_rows.count()
            col_match = self.intersect_rows.select(f"{column}_match")
            match_cnt = col_match.where(
                F.col(f"{column}_match") == True  # noqa: E712
            ).count()
            sample_count = min(sample_count, row_cnt - match_cnt)
            sample = (
                self.intersect_rows.where(F.col(f"{column}_match") == False)  # noqa: E712
                .drop(f"{column}_match")
                .limit(sample_count)
            )

            sample = sample.withColumnsRenamed(
                {f"{c}_{self.df1_name}": c for c in self.join_columns}
            )

            return_cols = [
                *self.join_columns,
                f"{column}_{self.df1_name}",
                f"{column}_{self.df2_name}",
            ]
            to_return = sample.select(return_cols)

            if for_display:
                return to_return.toDF(
                    *[
                        *self.join_columns,
                        f"{column} ({self.df1_name})",
                        f"{column} ({self.df2_name})",
                    ]
                )
            return to_return
        else:
            row_cnt = (
                self.intersect_rows.count()
                + self.df1_unq_rows.count()
                + self.df2_unq_rows.count()
            )
            match_cnt = self.intersect_rows.count()
            sample_count = min(sample_count, row_cnt - match_cnt)
            df1_col = f"{column}_{self.df1_name}"
            df2_col = f"{column}_{self.df2_name}"
            sample = (
                self.df1_unq_rows[[df1_col]]
                .union(self.df2_unq_rows[[df2_col]])
                .limit(sample_count)
            )
            return sample.toDF(column)

    def all_mismatch(
        self, ignore_matching_cols: bool = False
    ) -> "pyspark.sql.DataFrame":
        """Get all rows with any columns that have a mismatch.

        Returns all df1 and df2 versions of the columns and join
        columns.

        Parameters
        ----------
        ignore_matching_cols : bool, optional
            Whether showing the matching columns in the output or not. The default is False.

        Returns
        -------
        pyspark.sql.DataFrame
            All rows of the intersection dataframe, containing any columns, that don't match.
        """
        match_list = []
        return_list = []
        if self.only_join_columns():
            LOG.info("Only join keys in data, returning mismatches based on unq_rows")
            df1_cols = [f"{cols}_{self.df1_name}" for cols in self.join_columns]
            df2_cols = [f"{cols}_{self.df2_name}" for cols in self.join_columns]
            to_return = self.df1_unq_rows[df1_cols].union(self.df2_unq_rows[df2_cols])
            to_return = to_return.withColumnsRenamed(
                {f"{c}_{self.df1_name}": c for c in self.join_columns}
            )
            return to_return

        # get the match cols to aggregate
        match_cols = [c for c in self.intersect_rows.columns if c.endswith("_match")]

        if match_cols:
            agg_exprs = [
                F.sum(F.when(F.col(c) == False, 1).otherwise(0)).alias(f"{c}_count")  # noqa: E712
                for c in match_cols
            ]
            mismatch_counts = self.intersect_rows.agg(*agg_exprs).first()
        else:
            mismatch_counts = {}

        for c in self.intersect_rows.columns:
            if c.endswith("_match"):
                orig_col_name = c[:-6]

                if not ignore_matching_cols or (
                    ignore_matching_cols and mismatch_counts[f"{c}_count"] > 0
                ):
                    LOG.debug(f"Adding column {orig_col_name} to the result.")
                    match_list.append(c)
                    return_list.extend(
                        [
                            f"{orig_col_name}_{self.df1_name}",
                            f"{orig_col_name}_{self.df2_name}",
                        ]
                    )
                elif ignore_matching_cols:
                    LOG.debug(
                        f"Column {orig_col_name} is equal in df1 and df2. It will not be added to the result."
                    )

        if len(match_list) == 0:
            LOG.info("No match columns found, returning mismatches based on unq_rows")
            df1_cols = [f"{cols}_{self.df1_name}" for cols in self.join_columns]
            df2_cols = [f"{cols}_{self.df2_name}" for cols in self.join_columns]
            to_return = self.df1_unq_rows[df1_cols].union(self.df2_unq_rows[df2_cols])
            to_return = to_return.withColumnsRenamed(
                {f"{c}_{self.df1_name}": c for c in self.join_columns}
            )
            return to_return

        mm_rows = self.intersect_rows.withColumn(
            "match_array", F.array(match_list)
        ).where(F.array_contains("match_array", False))

        mm_rows = mm_rows.withColumnsRenamed(
            {f"{c}_{self.df1_name}": c for c in self.join_columns}
        )

        return mm_rows.select(self.join_columns + return_list)

    def _get_column_summary(self) -> dict:
        """Generate column summary data for the report.

        Returns
        -------
        dict
            Dictionary containing column summary information.
        """
        return {
            "column_summary": {
                "common_columns": len(self.intersect_columns()),
                "df1_unique": f"{len(self.df1_unq_columns())} {self.df1_unq_columns().items}",
                "df2_unique": f"{len(self.df2_unq_columns())} {self.df2_unq_columns().items}",
                "df1_name": self.df1_name,
                "df2_name": self.df2_name,
            }
        }

    def _get_row_summary(self) -> dict:
        """Generate row summary data for the report.

        Returns
        -------
        dict
            Dictionary containing row summary information.
        """
        intersect_count = self.intersect_rows.count()
        df1_unq_count = self.df1_unq_rows.count()
        df2_unq_count = self.df2_unq_rows.count()
        matching_rows = self.count_matching_rows()

        return {
            "row_summary": {
                "match_columns": ", ".join(self.join_columns),
                "abs_tol": self.abs_tol,
                "rel_tol": self.rel_tol,
                "common_rows": intersect_count,
                "df1_unique": df1_unq_count,
                "df2_unique": df2_unq_count,
                "unequal_rows": intersect_count - matching_rows,
                "equal_rows": matching_rows,
                "df1_name": self.df1_name,
                "df2_name": self.df2_name,
                "has_duplicates": "Yes" if self._any_dupes else "No",
            }
        }

    def _get_column_comparison(self) -> dict:
        """Generate column comparison statistics for the report.

        Returns
        -------
        dict
            Dictionary containing column comparison information.
        """
        return {
            "column_comparison": {
                "unequal_columns": len(
                    [col for col in self.column_stats if col["unequal_cnt"] > 0]
                ),
                "equal_columns": len(
                    [col for col in self.column_stats if col["unequal_cnt"] == 0]
                ),
                "unequal_values": sum(col["unequal_cnt"] for col in self.column_stats),
            }
        }

    def _get_mismatch_stats(self, sample_count: int) -> dict:
        """Generate mismatch statistics for the report.

        Parameters
        ----------
        sample_count : int
            Number of samples to include in the report.

        Returns
        -------
        dict
            Dictionary containing mismatch statistics.
        """
        match_stats = []
        match_sample = []
        any_mismatch = False

        for column in self.column_stats:
            if not column["all_match"]:
                any_mismatch = True
                match_stats.append(
                    {
                        "column": column["column"],
                        "dtype1": column["dtype1"],
                        "dtype2": column["dtype2"],
                        "unequal_cnt": column["unequal_cnt"],
                        "max_diff": column["max_diff"],
                        "null_diff": column["null_diff"],
                        "rel_tol": column["rel_tol"],
                        "abs_tol": column["abs_tol"],
                    }
                )
                if column["unequal_cnt"] > 0:
                    match_sample.append(
                        self.sample_mismatch(
                            column["column"], sample_count, for_display=True
                        )
                    )

        if any_mismatch:
            return {
                "mismatch_stats": {
                    "has_mismatches": True,
                    "stats": match_stats,
                    "df1_name": self.df1_name,
                    "df2_name": self.df2_name,
                    "samples": [df_to_str(sample) for sample in match_sample],
                    "has_samples": len(match_sample) > 0 and sample_count > 0,
                }
            }
        return {
            "mismatch_stats": {
                "has_mismatches": False,
                "has_samples": False,
            }
        }

    def _get_unique_rows_data(self, sample_count: int, column_count: int) -> dict:
        """Generate data for unique rows in both dataframes.

        Parameters
        ----------
        sample_count : int
            Number of samples to include.
        column_count : int
            Number of columns to include.

        Returns
        -------
        dict
            Dictionary containing unique rows data for both dataframes.
        """
        df1_unq_count = self.df1_unq_rows.count()
        df2_unq_count = self.df2_unq_rows.count()

        min_sample_count_df1 = min(sample_count, df1_unq_count)
        min_sample_count_df2 = min(sample_count, df2_unq_count)
        min_column_count_df1 = min(column_count, len(self.df1_unq_rows.columns))
        min_column_count_df2 = min(column_count, len(self.df2_unq_rows.columns))

        return {
            "sample_count": sample_count,
            "column_count": column_count,
            "df1_unique_rows": {
                "has_rows": min_sample_count_df1 > 0,
                "rows": (
                    df_to_str(
                        self.df1_unq_rows.select(
                            self.df1_unq_rows.columns[:min_column_count_df1]
                        ),
                        sample_count=min_sample_count_df1,
                    )
                    if df1_unq_count > 0
                    else ""
                ),
                "columns": (
                    self.df1_unq_rows.columns[:min_column_count_df1]
                    if df1_unq_count > 0
                    else ""
                ),
            },
            "df2_unique_rows": {
                "has_rows": min_sample_count_df2 > 0,
                "rows": (
                    df_to_str(
                        self.df2_unq_rows.select(
                            self.df2_unq_rows.columns[:min_column_count_df2]
                        ),
                        sample_count=min_sample_count_df2,
                    )
                    if df2_unq_count > 0
                    else ""
                ),
                "columns": (
                    self.df2_unq_rows.columns[:min_column_count_df2]
                    if df2_unq_count > 0
                    else ""
                ),
            },
        }

    def report(
        self,
        sample_count: int = 10,
        column_count: int = 10,
        html_file: str | None = None,
        template_path: str | None = None,
    ) -> str:
        """Return a string representation of a report.

        The representation can then be printed or saved to a file. You can customize the
        report's appearance by providing a custom Jinja2 template.

        Parameters
        ----------
        sample_count : int, optional
            The number of sample records to return. Defaults to 10.

        column_count : int, optional
            The number of columns to display in the sample records output. Defaults to 10.

        html_file : str, optional
            HTML file name to save report output to. If ``None`` the file creation will be skipped.

        template_path : str, optional
            Path to a custom Jinja2 template file to use for report generation.
            If ``None``, the default template will be used. The template receives the
            following context variables:

            - ``column_summary``: Dict with column statistics including: ``common_columns``, ``df1_unique``, ``df2_unique``, ``df1_name``, ``df2_name``
            - ``row_summary``: Dict with row statistics including: ``match_columns``, ``equal_rows``, ``unequal_rows``
            - ``column_comparison``: Dict with column comparison statistics including: ``unequal_columns``, ``equal_columns``, ``unequal_values``
            - ``mismatch_stats``: Dict containing:
                - ``stats``: List of dicts with column mismatch statistics (column, match, mismatch, null_diff, etc.)
                - ``samples``: Sample rows with mismatched values
                - ``has_samples``: Boolean indicating if there are any mismatch samples
                - ``has_mismatches``: Boolean indicating if there are any mismatches
            - ``df1_unique_rows``: Dict with unique rows in df1 including: ``has_rows``, ``rows``, ``columns``
            - ``df2_unique_rows``: Dict with unique rows in df2 including: ``has_rows``, ``rows``, ``columns``

        Returns
        -------
        str
            The report, formatted according to the template.
        """
        # Get counts for the dataframes
        df1_count = self.df1.count()
        df2_count = self.df2.count()

        # Prepare template data
        template_data: Dict[str, Any] = {
            **self._get_column_summary(),
            **self._get_row_summary(),
            **self._get_column_comparison(),
            **self._get_mismatch_stats(sample_count),
            **self._get_unique_rows_data(sample_count, column_count),
            "df1_name": self.df1_name,
            "df2_name": self.df2_name,
            "df1_shape": (df1_count, len(self.df1.columns)),
            "df2_shape": (df2_count, len(self.df2.columns)),
            "column_count": column_count,
        }

        # Determine which template to use
        template_name = template_path if template_path else "report_template.j2"

        # Render the main report
        report = render(template_name, **template_data)

        if html_file:
            save_html_report(report, html_file)

        return report


def columns_equal(
    dataframe: "pyspark.sql.DataFrame",
    col_1: str,
    col_2: str,
    rel_tol: float = 0,
    abs_tol: float = 0,
    ignore_spaces: bool = False,
    ignore_case: bool = False,
    comparators: List[BaseComparator] | None = None,
    **kwargs,
) -> "pyspark.sql.Column":
    """Compare if two columns are considered equal, returns a boolean Spark Column to be used in a `.withColumn(...)` statement.

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
    dataframe: pyspark.sql.DataFrame
        DataFrame to do comparison on
    col_1 : str
        The first column to look at
    col_2 : str
        The second column
    rel_tol : float, optional
        Relative tolerance
    abs_tol : float, optional
        Absolute tolerance
    ignore_spaces : bool, optional
        Flag to strip whitespace (including newlines) from string columns
    ignore_case : bool, optional
        Flag to ignore the case of string columns
    comparators: List[BaseComparator] | None = None,
        A list of custom comparator classes to use to compare columns.
    **kwargs
        Additional keyword arguments to pass to custom comparators.

    Returns
    -------
    pyspark.sql.Column
        Boolean Spark Column: True if values match according to the rules above, False otherwise.

    Example
    -------
    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([
    ...     (1, 1.1, "ABC", None),
    ...     (2, 2.0, "DEF ", 4.0),
    ...     (3, None, "ghi", 5.0)
    ... ], ["id", "col1", "col2", "col3"])
    >>> # Compare numeric columns with tolerance
    >>> df = df.withColumn("col1_equals_col3", columns_equal(df, "col1", "col3", rel_tol=0.1))
    >>> # Compare string columns ignoring case and spaces
    >>> df = df.withColumn("col2_normalized", columns_equal(df, "col2", "col2", ignore_spaces=True, ignore_case=True))

    Note
    ----
    Starting in version 0.18.0, the behavior of this function was changed so rather than returning
    a DataFrame a Column expression is returned.
    """
    comparators_ = comparators

    if not comparators_:
        # If no comparators are passed, behave as before.
        comparators_ = _SPARK_DEFAULT_COMPARATORS

    for comparator in comparators_:
        if isinstance(comparator, SparkNumericComparator):
            compare = comparator.compare(
                dataframe, col_1, col_2, rtol=rel_tol, atol=abs_tol
            )
        elif isinstance(comparator, SparkStringComparator):
            compare = comparator.compare(
                dataframe,
                col_1,
                col_2,
                ignore_space=ignore_spaces,
                ignore_case=ignore_case,
            )
        elif isinstance(comparator, SparkArrayLikeComparator):
            compare = comparator.compare(dataframe, col_1, col_2)
        else:
            # for custom comparators pass all the available parameters
            # custom comparators can ignore what they don't need.
            compare = comparator.compare(dataframe, col_1, col_2, **kwargs)

        if compare is not None:
            LOG.info(
                f"Using comparator: {comparator.__class__.__name__} for column ({col_1}, {col_2}) comparison."
            )
            return compare

    compare = F.lit(False)
    return compare


def get_merged_columns(
    original_df: "pyspark.sql.DataFrame",
    merged_df: "pyspark.sql.DataFrame",
    suffix: str,
) -> List[str]:
    """Get the columns from an original dataframe, in the new merged dataframe.

    Parameters
    ----------
    original_df : pyspark.sql.DataFrame
        The original, pre-merge dataframe
    merged_df : pyspark.sql.DataFrame
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
    for c in original_df.columns:
        if c in merged_df.columns:
            columns.append(c)
        elif c + "_" + suffix in merged_df.columns:
            columns.append(c + "_" + suffix)
        else:
            raise ValueError("Column not found: %s", c)
    return columns


def calculate_max_diff(
    dataframe: "pyspark.sql.DataFrame", col_1: str, col_2: str
) -> float:
    """Get a maximum difference between two columns.

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
    float
        max diff
    """
    diff = dataframe.select(
        (F.col(col_1).astype("float") - F.col(col_2).astype("float")).alias("diff")
    )
    abs_diff = diff.select(F.abs(F.col("diff")).alias("abs_diff"))
    max_diff: float = (
        abs_diff.where(F.isnan(F.col("abs_diff")) == False)  # noqa: E712
        .agg({"abs_diff": "max"})
        .collect()[0][0]
    )

    if pd.isna(max_diff) or pd.isnull(max_diff) or max_diff is None:
        return 0
    else:
        return max_diff


def calculate_null_diff(
    dataframe: "pyspark.sql.DataFrame", col_1: str, col_2: str
) -> int:
    """Get the null differences between two columns.

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
    int
        null diff
    """
    nulls_df = dataframe.withColumn(
        "col_1_null",
        F.when(F.col(col_1).isNull() == True, F.lit(True)).otherwise(  # noqa: E712
            F.lit(False)
        ),
    )
    nulls_df = nulls_df.withColumn(
        "col_2_null",
        F.when(F.col(col_2).isNull() == True, F.lit(True)).otherwise(  # noqa: E712
            F.lit(False)
        ),
    ).select(["col_1_null", "col_2_null"])

    # (not a and b) or (a and not b)
    null_diff = nulls_df.where(
        ((F.col("col_1_null") == False) & (F.col("col_2_null") == True))  # noqa: E712
        | ((F.col("col_1_null") == True) & (F.col("col_2_null") == False))  # noqa: E712
    ).count()

    if pd.isna(null_diff) or pd.isnull(null_diff) or null_diff is None:
        return 0
    else:
        return null_diff


def _generate_id_within_group(
    dataframe: "pyspark.sql.DataFrame", join_columns: List[str], order_column_name: str
) -> "pyspark.sql.DataFrame":
    """Generate an ID column that can be used to deduplicate identical rows.

    The series generated
    is the order within a unique group, and it handles nulls. Requires a ``__index`` column.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        The dataframe to operate on
    join_columns : list
        List of strings which are the join columns
    order_column_name: str
        The name of the ``row_number`` column name

    Returns
    -------
    pyspark.sql.DataFrame
        Original dataframe with the ID column that's unique in each group
    """
    default_value = "DATACOMPY_NULL"
    null_cols = [f"any(isnull({c}))" for c in join_columns]
    default_cols = [f"any({c} == '{default_value}')" for c in join_columns]

    null_check = any(list(dataframe.selectExpr(null_cols).first()))
    default_check = any(list(dataframe.selectExpr(default_cols).first()))

    if null_check:
        if default_check:
            raise ValueError(f"{default_value} was found in your join columns")

        return (
            dataframe.select(
                *(F.col(c).cast("string").alias(c) for c in [*join_columns, "__index"])
            )
            .fillna(default_value)
            .withColumn(
                order_column_name,
                F.row_number().over(Window.orderBy("__index").partitionBy(join_columns))
                - 1,
            )
            .select(["__index", order_column_name])
        )
    else:
        return (
            dataframe.select([*join_columns, "__index"])
            .withColumn(
                order_column_name,
                F.row_number().over(Window.orderBy("__index").partitionBy(join_columns))
                - 1,
            )
            .select(["__index", order_column_name])
        )


def _get_column_dtypes(
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


def _is_comparable(type1: str, type2: str) -> bool:
    """Check if two Spark data types can be safely compared.

    Two data types are considered comparable if any of the following apply:
        1. Both data types are the same
        2. Both data types are numeric

    Parameters
    ----------
    type1 : str
        A string representation of a Spark data type
    type2 : str
        A string representation of a Spark data type

    Returns
    -------
    bool
        True if both data types are comparable
    """
    return (
        type1 == type2
        or (type1 in NUMERIC_SPARK_TYPES and type2 in NUMERIC_SPARK_TYPES)
        or ({type1, type2} == {"string", "timestamp"})
        or ({type1, type2} == {"string", "date"})
    )
