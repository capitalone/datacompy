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

"""
Compare two Polars DataFrames.

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""

import inspect
import logging
from copy import deepcopy
from typing import Any, Dict, List, cast

import polars as pl
from ordered_set import OrderedSet

from datacompy.base import (
    BaseCompare,
    ColumnStat,
    get_column_tolerance,
    temp_column_name,
    validate_tolerance_parameter,
)
from datacompy.comparator import (
    PolarsArrayLikeComparator,
    PolarsBooleanComparator,
    PolarsNumericComparator,
    PolarsStringComparator,
)
from datacompy.comparator.base import BaseComparator
from datacompy.comparator.string import polars_normalize_string_column

LOG = logging.getLogger(__name__)

STRING_TYPE = ["String", "Utf8"]
LIST_TYPE = ["List", "Array"]

# ``DataFrame.join`` spells its null-equality argument ``join_nulls`` up to polars
# 1.23, ``nulls_equal`` from polars 1.24, and drops ``join_nulls`` entirely in
# polars 2.0.  Resolving the name from the installed signature keeps a single
# join path across the whole supported range.
_JOIN_NULLS_KWARG = (
    "nulls_equal"
    if "nulls_equal" in inspect.signature(pl.DataFrame.join).parameters
    else "join_nulls"
)

_POLARS_DEFAULT_COMPARATORS = [
    PolarsArrayLikeComparator(),
    PolarsBooleanComparator(),
    PolarsNumericComparator(),
    PolarsStringComparator(),
]


class PolarsCompare(BaseCompare):
    """Comparison class to be used to compare whether two dataframes as equal.

    Both df1 and df2 should be dataframes containing all of the join_columns,
    with unique column names. Differences between values are compared to
    abs_tol + rel_tol * abs(df2['value']).

    Parameters
    ----------
    df1 : Polars ``DataFrame``
        First dataframe to check
    df2 : Polars ``DataFrame``
        Second dataframe to check
    join_columns : list or str
        Column(s) to join dataframes on.  If a string is passed in, that one
        column will be used.
    abs_tol : float or dict, optional
        Absolute tolerance between two values. Can be either a float value applied to all columns,
        or a dictionary mapping column names to specific tolerance values. The special key "default"
        in the dictionary specifies the tolerance for columns not explicitly listed.
    rel_tol : float or dict, optional
        Relative tolerance between two values. Can be either a float value applied to all columns,
        or a dictionary mapping column names to specific tolerance values. The special key "default"
    df1_name : str, optional
        A string name for the first dataframe.  This allows the reporting to
        print out an actual name instead of "df1", and allows human users to
        more easily track the dataframes.
    df2_name : str, optional
        A string name for the second dataframe
    ignore_spaces : bool, optional
        Flag to strip whitespace (including newlines) from string columns (including any join
        columns). Excludes categoricals.
    ignore_case : bool, optional
        Flag to ignore the case of string columns. Excludes categoricals.
    cast_column_names_lower: bool, optional
        Boolean indicator that controls of column names will be cast into lower case
    custom_comparators : list of ``BaseComparator``, optional
        A list of custom comparator classes to use to compare columns.
    """

    def __init__(
        self,
        df1: pl.DataFrame,
        df2: pl.DataFrame,
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
        self._sensitive_columns: List[str] | None = None

        # Validate tolerance parameters first
        self._abs_tol_dict = validate_tolerance_parameter(
            abs_tol, "abs_tol", "lower" if cast_column_names_lower else "preserve"
        )
        self._rel_tol_dict = validate_tolerance_parameter(
            rel_tol, "rel_tol", "lower" if cast_column_names_lower else "preserve"
        )

        if isinstance(join_columns, str):
            self.join_columns = [
                str(join_columns).lower()
                if self.cast_column_names_lower
                else str(join_columns)
            ]
        elif isinstance(join_columns, list):
            self.join_columns = [
                str(col).lower() if self.cast_column_names_lower else str(col)
                for col in join_columns
            ]
        else:
            raise TypeError(f"{join_columns} must be a string or list of string(s)")

        self._any_dupes: bool = False
        self.df1 = df1
        self.df2 = df2
        self.df1_name = df1_name
        self.df2_name = df2_name
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.ignore_spaces = ignore_spaces
        self.ignore_case = ignore_case
        self.df1_unq_rows: pl.DataFrame
        self.df2_unq_rows: pl.DataFrame
        self.intersect_rows: pl.DataFrame
        self.column_stats: List[ColumnStat] = []
        self._compare(ignore_spaces=ignore_spaces, ignore_case=ignore_case)

    def _get_comparators(self) -> List[BaseComparator]:
        """Build and return the list of comparators to be used.

        Custom comparators are placed first, followed by the default ones.
        """
        return self.custom_comparators + _POLARS_DEFAULT_COMPARATORS

    @property
    def df1(self) -> pl.DataFrame:
        """The first dataframe."""
        return self._df1

    @df1.setter
    def df1(self, df1: pl.DataFrame) -> None:
        """Check that it is a dataframe and has the join columns."""
        self._df1 = df1
        self._validate_dataframe(
            "df1", cast_column_names_lower=self.cast_column_names_lower
        )

    @property
    def df2(self) -> pl.DataFrame:
        """The second dataframe."""
        return self._df2

    @df2.setter
    def df2(self, df2: pl.DataFrame) -> None:
        """Check that it is a dataframe and has the join columns."""
        self._df2 = df2
        self._validate_dataframe(
            "df2", cast_column_names_lower=self.cast_column_names_lower
        )

    def hide_sensitive_columns(self, sensitive_columns: List[str]) -> None:
        """Hides sensitive columns of df1 or df2 if applicable in the compare."""
        # Don't allow hiding columns again before first revealing
        if self.sensitive_columns:
            raise ValueError(
                "sensitive columns are already hidden, call reveal_sensitive_columns() first"
            )

        self._set_and_validate_sensitive_columns(sensitive_columns)
        # Don't do anything if [] is passed (normalized to None)
        if not self.sensitive_columns:
            return
        # Derived stats (max_diff, null_diff) come from raw values and must not
        # survive masking (issue #565).
        self._mask_sensitive_column_stats()
        sensitive = set(self.sensitive_columns)  # Otherwise this fails due to None
        sensitive_with_suffixes = (
            sensitive
            | {f"{c}_{self.df1_name}" for c in sensitive}
            | {f"{c}_{self.df2_name}" for c in sensitive}
        )

        # Hide columns in unq_rows
        for df_name in ("df1_unq_rows", "df2_unq_rows"):
            df = getattr(self, df_name)
            LOG.debug(f"Hiding sensitive columns in {df_name}")
            cols_to_hide = [col for col in df.columns if col in sensitive]
            if not cols_to_hide:  # skip if empty
                continue
            setattr(
                self,
                df_name,
                df.with_columns([pl.lit("*******").alias(col) for col in cols_to_hide]),
            )

        # Hide columns in intersect_rows
        LOG.debug("Hiding sensitive columns in intersect_rows")
        cols_to_hide = [
            col for col in self.intersect_rows.columns if col in sensitive_with_suffixes
        ]
        if not cols_to_hide:  # skip if empty
            return
        self.intersect_rows = self.intersect_rows.with_columns(
            [pl.lit("*******").alias(col) for col in cols_to_hide]
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
        """
        dataframe = getattr(self, index)
        if not isinstance(dataframe, pl.DataFrame):
            raise TypeError(f"{index} must be a Polars DataFrame")

        if cast_column_names_lower:
            # Assigning duplicate names raises pl.exceptions.DuplicateError.
            # Reject that case here with the same ValueError the other backends use.
            columns = [str(col).lower() for col in dataframe.columns]
            if len(set(columns)) < len(columns):
                raise ValueError(f"{index} must have unique column names")
            dataframe.columns = columns

        # Check if join_columns are present in the dataframe
        if not set(self.join_columns).issubset(set(dataframe.columns)):
            missing_cols = set(self.join_columns) - set(dataframe.columns)
            raise ValueError(
                f"{index} must have all columns from join_columns: {missing_cols}"
            )

        if len(dataframe.unique(subset=self.join_columns)) < len(dataframe):
            self._any_dupes = True

    def _compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        """Run the comparison.

        This tries to run df1.equals(df2)
        first so that if they're truly equal we can tell.

        This method will log out information about what is different between
        the two dataframes, and will also return a boolean.
        """
        LOG.debug("Checking equality")
        if self.df1.equals(self.df2):
            LOG.info("df1 pl.DataFrame.equals df2")
        else:
            LOG.info("df1 does not pl.DataFrame.equals df2")
        LOG.info(f"Number of columns in common: {len(self.intersect_columns())}")
        LOG.debug("Checking column overlap")
        for col in self.df1_unq_columns():
            LOG.info(f"Column in df1 and not in df2: {col}")
        LOG.info(
            f"Number of columns in df1 and not in df2: {len(self.df1_unq_columns())}"
        )
        for col in self.df2_unq_columns():
            LOG.info(f"Column in df2 and not in df1: {col}")
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
        """Perform an outer join between two dataframes and categorize rows into unique and intersecting groups based on the join columns.

        Parameters
        ----------
        ignore_spaces : bool
            If True, normalizes string columns by ignoring spaces during the join operation.

        Notes
        -----
        - Temporary columns may be added to the dataframes during processing
          and are cleaned up before final output.
        - The method assumes that `self.df1`, `self.df2`, and `self.join_columns`
          are properly initialized before calling this method.
        """
        params: Dict[str, Any]
        LOG.debug("Outer joining")

        df1 = self.df1.clone()
        df2 = self.df2.clone()
        temp_join_columns = deepcopy(self.join_columns)

        if self._any_dupes:
            LOG.debug("Duplicate rows found, deduping by order of remaining fields")
            # Create order column for uniqueness of match
            order_column = temp_column_name(df1, df2)
            df1 = df1.with_columns(
                generate_id_within_group(df1, temp_join_columns).alias(order_column)
            )
            df2 = df2.with_columns(
                generate_id_within_group(df2, temp_join_columns).alias(order_column)
            )
            temp_join_columns.append(order_column)

        params = {"on": temp_join_columns}

        schema1 = df1.schema
        schema2 = df2.schema

        df1_exprs = []
        df2_exprs = []
        if ignore_spaces:
            for column in self.join_columns:
                df1_exprs.append(
                    polars_normalize_string_column(
                        pl.col(column),
                        schema=schema1,
                        ignore_spaces=ignore_spaces,
                        ignore_case=False,
                    ).alias(column)
                )
                df2_exprs.append(
                    polars_normalize_string_column(
                        pl.col(column),
                        schema=schema2,
                        ignore_spaces=ignore_spaces,
                        ignore_case=False,
                    ).alias(column)
                )
        df1 = df1.with_columns(df1_exprs)
        df2 = df2.with_columns(df2_exprs)

        df1_non_join_columns = OrderedSet(schema1.names()) - OrderedSet(
            temp_join_columns
        )
        df2_non_join_columns = OrderedSet(schema2.names()) - OrderedSet(
            temp_join_columns
        )

        df1 = df1.rename({c: f"{c}_{self.df1_name}" for c in df1_non_join_columns})
        df2 = df2.rename({c: f"{c}_{self.df2_name}" for c in df2_non_join_columns})

        # generate merge indicator
        df1 = df1.with_columns(_merge_left=pl.lit(True))
        df2 = df2.with_columns(_merge_right=pl.lit(True))

        null_equality: Dict[str, Any] = {_JOIN_NULLS_KWARG: True}
        outer_join = df1.join(df2, how="full", coalesce=True, **null_equality, **params)

        # process merge indicator
        outer_join = outer_join.with_columns(
            pl.when(pl.col("_merge_left") & pl.col("_merge_right"))
            .then(pl.lit("both"))
            .when(pl.col("_merge_left") & pl.col("_merge_right").is_null())
            .then(pl.lit("left_only"))
            .when(pl.col("_merge_left").is_null() & pl.col("_merge_right"))
            .then(pl.lit("right_only"))
            .alias("_merge")
        )

        # Clean up temp columns for duplicate row matching
        if self._any_dupes:
            outer_join = outer_join.drop(order_column)

        df1_cols = get_merged_columns(self.df1, outer_join, self.df1_name)
        df2_cols = get_merged_columns(self.df2, outer_join, self.df2_name)

        LOG.debug("Selecting df1 unique rows")
        self.df1_unq_rows = outer_join.filter(
            outer_join["_merge"] == "left_only"
        ).select(df1_cols)
        self.df1_unq_rows.columns = self.df1.columns

        LOG.debug("Selecting df2 unique rows")
        self.df2_unq_rows = outer_join.filter(
            outer_join["_merge"] == "right_only"
        ).select(df2_cols)
        self.df2_unq_rows.columns = self.df2.columns

        LOG.info(f"Number of rows in df1 and not in df2: {self.df1_unq_rows.height}")
        LOG.info(f"Number of rows in df2 and not in df1: {self.df2_unq_rows.height}")

        LOG.debug("Selecting intersecting rows")
        self.intersect_rows = outer_join.filter(outer_join["_merge"] == "both")
        LOG.info(
            f"Number of rows in df1 and df2 (not necessarily equal): {self.intersect_rows.height}"
        )

    def _intersect_compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        """Run the comparison on the intersect dataframe.

        This loops through all columns that are shared between df1 and df2, and
        creates a column column_match which is True for matches, False
        otherwise.
        """
        match_cnt: int
        null_diff: int
        max_diff: float | None

        LOG.debug("Comparing intersection")

        schema = self.intersect_rows.schema
        intersect_columns = self.intersect_columns()

        # Build comparison expressions for the columns to match on
        compare_exprs: List[pl.Expr] = []
        for column in intersect_columns:
            if column not in self.join_columns:
                col_1 = column + "_" + self.df1_name
                col_2 = column + "_" + self.df2_name
                col_match = column + "_match"
                compare_exprs.append(
                    columns_equal(
                        col_1=pl.col(col_1),
                        col_2=pl.col(col_2),
                        schema=schema,
                        rel_tol=get_column_tolerance(column, self._rel_tol_dict),
                        abs_tol=get_column_tolerance(column, self._abs_tol_dict),
                        ignore_spaces=ignore_spaces,
                        ignore_case=ignore_case,
                        comparators=self._get_comparators(),
                    ).alias(col_match)
                )
        self.intersect_rows = self.intersect_rows.with_columns(compare_exprs)

        # Calculate all non-join column statistics in a single dataframe pass.
        stat_exprs: List[pl.Expr] = []
        stat_aliases: Dict[str, tuple[str, str, str]] = {}
        for index, column in enumerate(intersect_columns):
            if column in self.join_columns:
                continue
            col_1 = column + "_" + self.df1_name
            col_2 = column + "_" + self.df2_name
            col_match = column + "_match"
            match_alias = f"__match_cnt_{index}"
            max_diff_alias = f"__max_diff_{index}"
            null_diff_alias = f"__null_diff_{index}"
            stat_aliases[column] = (
                match_alias,
                max_diff_alias,
                null_diff_alias,
            )
            stat_exprs.extend(
                [
                    pl.col(col_match).sum().alias(match_alias),
                    _calculate_max_diff_expr(
                        pl.col(col_1),
                        pl.col(col_2),
                        schema[col_1],
                        schema[col_2],
                    ).alias(max_diff_alias),
                    (pl.col(col_1).is_null() ^ pl.col(col_2).is_null())
                    .sum()
                    .alias(null_diff_alias),
                ]
            )
        aggregated_stats: Dict[str, Any] = (
            self.intersect_rows.select(stat_exprs).row(0, named=True)
            if stat_exprs
            else {}
        )

        # Calculate the column stats for each column
        for column in intersect_columns:
            if column in self.join_columns:
                col_match = column + "_match"
                match_cnt = self.intersect_rows.height
                if not self.only_join_columns():
                    row_cnt = self.intersect_rows.height
                else:
                    row_cnt = (
                        self.intersect_rows.height
                        + self.df1_unq_rows.height
                        + self.df2_unq_rows.height
                    )
                max_diff = 0.0
                null_diff = 0
            else:
                row_cnt = self.intersect_rows.height
                col_1 = column + "_" + self.df1_name
                col_2 = column + "_" + self.df2_name
                col_match = column + "_match"
                match_alias, max_diff_alias, null_diff_alias = stat_aliases[column]
                match_cnt = int(aggregated_stats[match_alias])
                max_diff = cast(float | None, aggregated_stats[max_diff_alias])
                null_diff = int(aggregated_stats[null_diff_alias])
            if row_cnt > 0:
                match_rate = float(match_cnt) / row_cnt
            else:
                match_rate = 0
            LOG.info(f"{column}: {match_cnt} / {row_cnt} ({match_rate:.2%}) match")

            self.column_stats.append(
                {
                    "column": column,
                    "match_column": col_match,
                    "match_cnt": match_cnt,
                    "unequal_cnt": row_cnt - match_cnt,
                    "dtype1": str(self.df1[column].dtype),
                    "dtype2": str(self.df2[column].dtype),
                    "all_match": all(
                        (
                            self.df1[column].dtype == self.df2[column].dtype,
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
        """Whether the columns all match in the dataframes."""
        return self.df1_unq_columns() == self.df2_unq_columns() == set()

    def all_rows_overlap(self) -> bool:
        """Whether the rows are all present in both dataframes.

        Returns
        -------
        bool
            True if all rows in df1 are in df2 and vice versa (based on
            existence for join option)
        """
        return len(self.df1_unq_rows) == len(self.df2_unq_rows) == 0

    def count_matching_rows(self) -> int:
        """Count the number of rows match (on overlapping fields).

        Returns
        -------
        int
            Number of matching rows
        """
        match_columns = []
        for column in self.intersect_columns():
            if column not in self.join_columns:
                match_columns.append(column + "_match")

        if len(match_columns) > 0:
            return int(
                self.intersect_rows[match_columns]
                .select(pl.all_horizontal(match_columns).alias("__sum"))
                .sum()
                .item()
            )
        else:
            # corner case where it is just the join columns that make the dataframes
            if len(self.intersect_rows) > 0:
                return len(self.intersect_rows)
            else:
                return 0

    def intersect_rows_match(self) -> bool:
        """Check whether the intersect rows all match."""
        if self.intersect_rows.is_empty():
            return False
        actual_length = self.intersect_rows.shape[0]
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
            and len(self.df2_unq_rows) == 0
            and self.intersect_rows_match()
        )

    def sample_mismatch(
        self, column: str, sample_count: int = 10, for_display: bool = False
    ) -> pl.DataFrame | None:
        """Return sample mismatches.

        Get a sub-dataframe which contains the identifying
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
        pl.DataFrame
            A sample of the intersection dataframe, containing only the
            "pertinent" columns, for rows that don't match on the provided
            column.

        None
            When the column being requested is not an intersecting column between dataframes.
        """
        if not self.only_join_columns() and column not in self.join_columns:
            row_cnt = self.intersect_rows.shape[0]
            col_match = self.intersect_rows[column + "_match"]
            match_cnt = col_match.sum()
            sample_count = min(sample_count, row_cnt - match_cnt)  # type: ignore
            sample = self.intersect_rows.filter(
                pl.col(column + "_match") != True  # noqa: E712
            ).sample(sample_count)
            return_cols = [
                *self.join_columns,
                column + "_" + self.df1_name,
                column + "_" + self.df2_name,
            ]
            to_return = sample[return_cols]
            if for_display:
                to_return.columns = [
                    *self.join_columns,
                    column + " (" + self.df1_name + ")",
                    column + " (" + self.df2_name + ")",
                ]
            return to_return
        else:
            row_cnt = (
                len(self.intersect_rows)
                + len(self.df1_unq_rows)
                + len(self.df2_unq_rows)
            )
            match_cnt = len(self.intersect_rows)
            sample_count = min(sample_count, row_cnt - match_cnt)
            sample = pl.concat(
                [self.df1_unq_rows[[column]], self.df2_unq_rows[[column]]]
            ).sample(sample_count)
            return sample

    def all_mismatch(self, ignore_matching_cols: bool = False) -> pl.DataFrame:
        """Get all rows with any columns that have a mismatch.

        Returns all df1 and df2 versions of the columns and join
        columns.

        Parameters
        ----------
        ignore_matching_cols : bool, optional
            Whether showing the matching columns in the output or not. The default is False.

        Returns
        -------
        pl.DataFrame
            All rows of the intersection dataframe, containing any columns, that don't match.
        """
        match_list = []
        return_list = []
        if self.only_join_columns():
            LOG.info("Only join keys in data, returning mismatches based on unq_rows")
            return pl.concat([self.df1_unq_rows, self.df2_unq_rows])

        # Walk the compared columns rather than every intersect_rows column
        # ending in "_match": join columns and columns unique to one frame can
        # have that suffix too, and are not compared.
        for orig_col_name in self.intersect_columns():
            if orig_col_name not in self.join_columns:
                col = orig_col_name + "_match"
                schema = self.intersect_rows.schema

                col_comparison = self.intersect_rows.select(
                    columns_equal(
                        col_1=pl.col(orig_col_name + "_" + self.df1_name),
                        col_2=pl.col(orig_col_name + "_" + self.df2_name),
                        schema=schema,
                        rel_tol=get_column_tolerance(orig_col_name, self._rel_tol_dict),
                        abs_tol=get_column_tolerance(orig_col_name, self._abs_tol_dict),
                        ignore_spaces=self.ignore_spaces,
                        ignore_case=self.ignore_case,
                        comparators=self._get_comparators(),
                    ).alias(col)
                ).to_series()

                if not ignore_matching_cols or (
                    ignore_matching_cols and not col_comparison.all()
                ):
                    LOG.debug(f"Adding column {orig_col_name} to the result.")
                    match_list.append(col)
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
        if len(match_list) == 0:
            LOG.info("No match columns found, returning mismatches based on unq_rows")
            return pl.concat(
                [
                    self.df1_unq_rows.select(self.join_columns),
                    self.df2_unq_rows.select(self.join_columns),
                ]
            )

        return (
            self.intersect_rows.with_columns(__all=pl.all_horizontal(match_list))
            .filter(pl.col("__all") != True)  # noqa: E712
            .select(self.join_columns + return_list)
        )


def columns_equal(
    col_1: pl.Expr,
    col_2: pl.Expr,
    schema: pl.Schema,
    rel_tol: float = 0,
    abs_tol: float = 0,
    ignore_spaces: bool = False,
    ignore_case: bool = False,
    comparators: List[BaseComparator] | None = None,
    **kwargs,
) -> pl.Expr:
    """Compare two columns from a dataframe.

    Returns a True/False column,

    - Two nulls (``np.nan``) will evaluate to True.
    - A null and a non-null value will evaluate to False.
    - Numeric values will use the relative and absolute tolerances.
    - Decimal values (``decimal.Decimal``) will attempt to be converted to floats
      before comparing
    - Non-numeric values (i.e. where ``np.isclose`` semantics can't be used) will just
      trigger True on two nulls or exact matches.

    Parameters
    ----------
    col_1 : pl.Expr
        The first column to look at
    col_2 : pl.Expr
        The second column
    schema : pl.Schema
        The schema of the columns to compare
    rel_tol : float, optional
        Relative tolerance
    abs_tol : float, optional
        Absolute tolerance
    ignore_spaces : bool, optional
        Flag to strip whitespace (including newlines) from string columns
    ignore_case : bool, optional
        Flag to ignore the case of string columns
    comparators : list of ``BaseComparator``, optional
        A list of custom comparator classes to use to compare columns.
    **kwargs
        Additional keyword arguments to pass to custom comparators.

    Returns
    -------
    pl.Expr
        A column of Boolean values.  True == the values match, False == the
        values don't match.
    """
    compare: pl.Expr

    comparators_ = comparators
    if not comparators_:
        # If no comparators are passed, behave as before.
        comparators_ = _POLARS_DEFAULT_COMPARATORS

    for comparator in comparators_:
        if isinstance(comparator, PolarsBooleanComparator):
            compare = comparator.compare(col_1, col_2, schema)
        elif isinstance(comparator, PolarsNumericComparator):
            compare = comparator.compare(
                col_1, col_2, schema, rtol=rel_tol, atol=abs_tol
            )
        elif isinstance(comparator, PolarsStringComparator):
            compare = comparator.compare(
                col_1,
                col_2,
                schema,
                ignore_space=ignore_spaces,
                ignore_case=ignore_case,
            )
        elif isinstance(comparator, PolarsArrayLikeComparator):
            compare = comparator.compare(col_1, col_2, schema)
        else:
            # for custom comparators pass all the available parameters
            # custom comparators can ignore what they don't need.
            compare = comparator.compare(col_1, col_2, schema, **kwargs)

        if compare is not None:
            LOG.info(
                f"Using comparator: {comparator.__class__.__name__} for column ({col_1.meta.output_name()}, {col_2.meta.output_name()}) comparison."
            )
            return compare

    return pl.lit(False, dtype=pl.Boolean)


def get_merged_columns(
    original_df: pl.DataFrame, merged_df: pl.DataFrame, suffix: str
) -> List[str]:
    """Get the columns from an original dataframe, in the new merged dataframe.

    Parameters
    ----------
    original_df : pl.DataFrame
        The original, pre-merge dataframe
    merged_df : pl.DataFrame
        Post-merge with another dataframe, with suffixes added in.
    suffix : str
        What suffix was used to distinguish when the original dataframe was
        overlapping with the other merged dataframe.
    """
    columns = []
    for col in original_df.columns:
        if col in merged_df.columns:
            columns.append(col)
        elif col + "_" + suffix in merged_df.columns:
            columns.append(col + "_" + suffix)
        else:
            raise ValueError("Column not found: %s", col)
    return columns


def _calculate_max_diff_expr(
    col_1: pl.Expr,
    col_2: pl.Expr,
    dtype_1: pl.DataType,
    dtype_2: pl.DataType,
) -> pl.Expr:
    """Build an aggregate expression for the maximum difference."""
    if dtype_1.is_temporal() or dtype_2.is_temporal():
        return pl.lit(0.0)

    castable_types = {*STRING_TYPE, "Boolean", "Null"}
    if not (
        (dtype_1.is_numeric() or str(dtype_1.base_type()) in castable_types)
        and (dtype_2.is_numeric() or str(dtype_2.base_type()) in castable_types)
    ):
        return pl.lit(0.0)

    cast_1 = col_1.cast(pl.Float64, strict=False)
    cast_2 = col_2.cast(pl.Float64, strict=False)
    cast_failed = (col_1.is_not_null() & cast_1.is_null()).any() | (
        col_2.is_not_null() & cast_2.is_null()
    ).any()
    max_diff = (cast_1.fill_null(0.0) - cast_2.fill_null(0.0)).abs().max()
    return pl.when(cast_failed).then(pl.lit(0.0)).otherwise(max_diff)


def generate_id_within_group(
    dataframe: pl.DataFrame, join_columns: List[str]
) -> pl.Series:
    """Generate an ID column that can be used to deduplicate identical rows.

    The series generated
    is the order within a unique group, and it handles nulls.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The dataframe to operate on
    join_columns : list
        List of strings which are the join columns

    Returns
    -------
    pl.Series
        The ID column that's unique in each group.
    """
    default_value = "DATACOMPY_NULL"
    if (
        dataframe.select(pl.any_horizontal(pl.col(join_columns).is_null()))
        .to_series()
        .any()
    ):
        if (
            dataframe.select(
                pl.any_horizontal(pl.col(join_columns).cast(pl.String) == default_value)
            )
            .to_series()
            .any()
        ):
            raise ValueError(f"{default_value} was found in your join columns")
        return (
            dataframe[join_columns]
            .cast(pl.String)
            .fill_null(default_value)
            .select(rn=pl.col(join_columns[0]).cum_count().over(join_columns))
            .to_series()
        )
    else:
        return dataframe.select(
            rn=pl.col(join_columns[0]).cum_count().over(join_columns)
        ).to_series()
