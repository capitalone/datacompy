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
Compare two Pandas DataFrames.

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""

import logging
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from datacompy.base import (
    BaseCompare,
    _validate_tolerance_parameter,
    df_to_str,
    get_column_tolerance,
    render,
    save_html_report,
    temp_column_name,
)

LOG = logging.getLogger(__name__)


class Compare(BaseCompare):
    """Comparison class to be used to compare whether two dataframes as equal.

    Both df1 and df2 should be dataframes containing all of the join_columns,
    with unique column names. Differences between values are compared to
    abs_tol + rel_tol * abs(df2['value']).

    Parameters
    ----------
    df1 : pandas ``DataFrame``
        First dataframe to check
    df2 : pandas ``DataFrame``
        Second dataframe to check
    join_columns : list or str, optional
        Column(s) to join dataframes on.  If a string is passed in, that one
        column will be used.
    on_index : bool, optional
        If True, the index will be used to join the two dataframes.  If both
        ``join_columns`` and ``on_index`` are provided, an exception will be
        raised.
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
        columns). Excludes categoricals.
    ignore_case : bool, optional
        Flag to ignore the case of string columns. Excludes categoricals.
    cast_column_names_lower: bool, optional
        Boolean indicator that controls of column names will be cast into lower case

    Attributes
    ----------
    df1_unq_rows : pandas ``DataFrame``
        All records that are only in df1 (based on a join on join_columns)
    df2_unq_rows : pandas ``DataFrame``
        All records that are only in df2 (based on a join on join_columns)
    """

    def __init__(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        join_columns: List[str] | str | None = None,
        on_index: bool = False,
        abs_tol: float | Dict[str, float] = 0,
        rel_tol: float | Dict[str, float] = 0,
        df1_name: str = "df1",
        df2_name: str = "df2",
        ignore_spaces: bool = False,
        ignore_case: bool = False,
        cast_column_names_lower: bool = True,
    ) -> None:
        self.cast_column_names_lower = cast_column_names_lower

        # Validate tolerance parameters first
        self._abs_tol_dict = _validate_tolerance_parameter(
            abs_tol, "abs_tol", "lower" if cast_column_names_lower else "preserve"
        )
        self._rel_tol_dict = _validate_tolerance_parameter(
            rel_tol, "rel_tol", "lower" if cast_column_names_lower else "preserve"
        )

        if on_index and join_columns is not None:
            raise Exception("Only provide on_index or join_columns")
        elif on_index:
            self.on_index = True
            self.join_columns = []
        elif isinstance(join_columns, str | int | float):
            self.join_columns = [
                str(join_columns).lower()
                if self.cast_column_names_lower
                else str(join_columns)
            ]
            self.on_index = False
        else:
            self.join_columns = [
                str(col).lower() if self.cast_column_names_lower else str(col)
                for col in cast(List[str], join_columns)
            ]
            self.on_index = False

        self._any_dupes: bool = False
        self.df1 = df1
        self.df2 = df2
        self.df1_name = df1_name
        self.df2_name = df2_name
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.ignore_spaces = ignore_spaces
        self.ignore_case = ignore_case
        self.df1_unq_rows: pd.DataFrame
        self.df2_unq_rows: pd.DataFrame
        self.intersect_rows: pd.DataFrame
        self.column_stats: List[Dict[str, Any]] = []
        self._compare(ignore_spaces=ignore_spaces, ignore_case=ignore_case)

    @property
    def df1(self) -> pd.DataFrame:
        """Get the first dataframe."""
        return self._df1

    @df1.setter
    def df1(self, df1: pd.DataFrame) -> None:
        """Check that it is a dataframe and has the join columns."""
        self._df1 = df1
        self._validate_dataframe(
            "df1", cast_column_names_lower=self.cast_column_names_lower
        )

    @property
    def df2(self) -> pd.DataFrame:
        """Get the second dataframe."""
        return self._df2

    @df2.setter
    def df2(self, df2: pd.DataFrame) -> None:
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
        """
        dataframe = getattr(self, index)
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"{index} must be a pandas DataFrame")

        if cast_column_names_lower:
            dataframe.columns = pd.Index(
                [str(col).lower() for col in dataframe.columns]
            )
        else:
            dataframe.columns = pd.Index([str(col) for col in dataframe.columns])
        # Check if join_columns are present in the dataframe
        if not set(self.join_columns).issubset(set(dataframe.columns)):
            missing_cols = set(self.join_columns) - set(dataframe.columns)
            raise ValueError(
                f"{index} must have all columns from join_columns: {missing_cols}"
            )

        if len(set(dataframe.columns)) < len(dataframe.columns):
            raise ValueError(f"{index} must have unique column names")

        if self.on_index:
            if dataframe.index.duplicated().sum() > 0:
                self._any_dupes = True
        else:
            if len(dataframe.drop_duplicates(subset=self.join_columns)) < len(
                dataframe
            ):
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
            LOG.info("df1 Pandas.DataFrame.equals df2")
        else:
            LOG.info("df1 does not Pandas.DataFrame.equals df2")
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
        """Merge df1 to df2 on the join columns.

        To get df1 - df2, df2 - df1
        and df1 & df2.

        If ``on_index`` is True, this will join on index values, otherwise it
        will join on the ``join_columns``.
        """
        params: Dict[str, Any]
        index_column: str
        LOG.debug("Outer joining")
        if self._any_dupes:
            LOG.debug("Duplicate rows found, deduping by order of remaining fields")
            # Bring index into a column
            if self.on_index:
                index_column = temp_column_name(self.df1, self.df2)
                self.df1[index_column] = self.df1.index
                self.df2[index_column] = self.df2.index
                temp_join_columns = [index_column]
            else:
                temp_join_columns = list(self.join_columns)

            # Create order column for uniqueness of match
            order_column = temp_column_name(self.df1, self.df2)
            self.df1[order_column] = generate_id_within_group(
                self.df1, temp_join_columns
            )
            self.df2[order_column] = generate_id_within_group(
                self.df2, temp_join_columns
            )
            temp_join_columns.append(order_column)

            params = {"on": temp_join_columns}
        elif self.on_index:
            params = {"left_index": True, "right_index": True}
        else:
            params = {"on": self.join_columns}

        for column in self.join_columns:
            self.df1[column] = normalize_string_column(
                self.df1[column], ignore_spaces=ignore_spaces, ignore_case=False
            )
            self.df2[column] = normalize_string_column(
                self.df2[column], ignore_spaces=ignore_spaces, ignore_case=False
            )

        outer_join = self.df1.merge(
            self.df2,
            how="outer",
            suffixes=("_" + self.df1_name, "_" + self.df2_name),
            indicator=True,
            **params,
        )
        # Clean up temp columns for duplicate row matching
        if self._any_dupes:
            if self.on_index:
                outer_join.set_index(keys=index_column, drop=True, inplace=True)
                self.df1.drop(index_column, axis=1, inplace=True)
                self.df2.drop(index_column, axis=1, inplace=True)
            outer_join.drop(labels=order_column, axis=1, inplace=True)
            self.df1.drop(order_column, axis=1, inplace=True)
            self.df2.drop(order_column, axis=1, inplace=True)

        df1_cols = get_merged_columns(self.df1, outer_join, self.df1_name)
        df2_cols = get_merged_columns(self.df2, outer_join, self.df2_name)

        LOG.debug("Selecting df1 unique rows")
        self.df1_unq_rows = outer_join[outer_join["_merge"] == "left_only"][
            df1_cols
        ].copy()
        self.df1_unq_rows.columns = self.df1.columns

        LOG.debug("Selecting df2 unique rows")
        self.df2_unq_rows = outer_join[outer_join["_merge"] == "right_only"][
            df2_cols
        ].copy()
        self.df2_unq_rows.columns = self.df2.columns
        LOG.info(f"Number of rows in df1 and not in df2: {len(self.df1_unq_rows)}")
        LOG.info(f"Number of rows in df2 and not in df1: {len(self.df2_unq_rows)}")

        LOG.debug("Selecting intersecting rows")
        self.intersect_rows = outer_join[outer_join["_merge"] == "both"].copy()
        LOG.info(
            f"Number of rows in df1 and df2 (not necessarily equal): {len(self.intersect_rows)}"
        )

    def _intersect_compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        """Run the comparison on the intersect dataframe.

        This loops through all columns that are shared between df1 and df2, and
        creates a column column_match which is True for matches, False
        otherwise.
        """
        LOG.debug("Comparing intersection")
        for column in self.intersect_columns():
            if column in self.join_columns:
                col_match = column + "_match"
                match_cnt = len(self.intersect_rows)
                if not self.only_join_columns():
                    row_cnt = len(self.intersect_rows)
                else:
                    row_cnt = (
                        len(self.intersect_rows)
                        + len(self.df1_unq_rows)
                        + len(self.df2_unq_rows)
                    )
                max_diff = 0.0
                null_diff = 0
            else:
                row_cnt = len(self.intersect_rows)
                col_1 = column + "_" + self.df1_name
                col_2 = column + "_" + self.df2_name
                col_match = column + "_match"
                self.intersect_rows = pd.concat(
                    [
                        self.intersect_rows,
                        columns_equal(
                            col_1=self.intersect_rows[col_1],
                            col_2=self.intersect_rows[col_2],
                            rel_tol=get_column_tolerance(column, self._rel_tol_dict),
                            abs_tol=get_column_tolerance(column, self._abs_tol_dict),
                            ignore_spaces=ignore_spaces,
                            ignore_case=ignore_case,
                        ).to_frame(name=col_match),
                    ],
                    axis=1,
                )
                match_cnt = self.intersect_rows[col_match].sum()
                max_diff = calculate_max_diff(
                    self.intersect_rows[col_1], self.intersect_rows[col_2]
                )
                null_diff = (
                    (self.intersect_rows[col_1].isnull())
                    ^ (self.intersect_rows[col_2].isnull())
                ).sum()

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
                    "dtype1": str(self.df1[column].dtype.__repr__())
                    if str(self.df1[column].dtype) == "string"
                    else str(self.df1[column].dtype),
                    "dtype2": str(self.df2[column].dtype.__repr__())
                    if str(self.df2[column].dtype) == "string"
                    else str(self.df2[column].dtype),
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
        return self.intersect_rows[match_columns].all(axis=1).sum()

    def intersect_rows_match(self) -> bool:
        """Check whether the intersect rows all match."""
        if self.intersect_rows.empty:
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
    ) -> pd.DataFrame | None:
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
        Pandas.DataFrame
            A sample of the intersection dataframe, containing only the
            "pertinent" columns, for rows that don't match on the provided
            column.

        None
            When the column being requested is not an intersecting column between dataframes.
        """
        if not self.only_join_columns() and column not in self.join_columns:
            row_cnt = self.intersect_rows.shape[0]
            try:
                col_match = self.intersect_rows[column + "_match"]
            except KeyError:
                LOG.error(
                    f"Column: {column} is not an intersecting column. No mismatches can be generated."
                )
                return None
            match_cnt = col_match.sum()
            sample_count = min(sample_count, row_cnt - match_cnt)
            sample = self.intersect_rows[~col_match].sample(sample_count)
            return_cols = [
                *self.join_columns,
                column + "_" + self.df1_name,
                column + "_" + self.df2_name,
            ]
            to_return = sample[return_cols]
            if for_display:
                to_return.columns = pd.Index(
                    [
                        *self.join_columns,
                        column + " (" + self.df1_name + ")",
                        column + " (" + self.df2_name + ")",
                    ]
                )
            return to_return
        else:
            row_cnt = (
                len(self.intersect_rows)
                + len(self.df1_unq_rows)
                + len(self.df2_unq_rows)
            )
            col_match = self.intersect_rows[column]
            match_cnt = col_match.count()
            sample_count = min(sample_count, row_cnt - match_cnt)
            sample = pd.concat(
                [self.df1_unq_rows[[column]], self.df2_unq_rows[[column]]]
            ).sample(sample_count)
            to_return = sample
            if for_display:
                to_return.columns = pd.Index([column])
            return to_return

    def all_mismatch(self, ignore_matching_cols: bool = False) -> pd.DataFrame:
        """Get all rows with any columns that have a mismatch.

        Returns all df1 and df2 versions of the columns and join
        columns.

        Parameters
        ----------
        ignore_matching_cols : bool, optional
            Whether showing the matching columns in the output or not. The default is False.

        Returns
        -------
        Pandas.DataFrame
            All rows of the intersection dataframe, containing any columns, that don't match.
        """
        match_list = []
        return_list = []
        if self.only_join_columns():
            LOG.info("Only join keys in data, returning mismatches based on unq_rows")
            return pd.concat([self.df1_unq_rows, self.df2_unq_rows])

        for col in self.intersect_rows.columns:
            if col.endswith("_match"):
                orig_col_name = col[:-6]

                col_comparison = columns_equal(
                    col_1=self.intersect_rows[orig_col_name + "_" + self.df1_name],
                    col_2=self.intersect_rows[orig_col_name + "_" + self.df2_name],
                    rel_tol=get_column_tolerance(orig_col_name, self._rel_tol_dict),
                    abs_tol=get_column_tolerance(orig_col_name, self._abs_tol_dict),
                    ignore_spaces=self.ignore_spaces,
                    ignore_case=self.ignore_case,
                )

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
            return pd.concat(
                [
                    self.df1_unq_rows[self.join_columns],
                    self.df2_unq_rows[self.join_columns],
                ]
            )

        mm_bool = self.intersect_rows[match_list].all(axis="columns")
        return self.intersect_rows[~mm_bool][self.join_columns + return_list]

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
                "df1_unique": len(self.df1_unq_columns()),
                "df2_unique": len(self.df2_unq_columns()),
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
        return {
            "row_summary": {
                "match_columns": "index"
                if self.on_index
                else ", ".join(self.join_columns),
                "abs_tol": self.abs_tol,
                "rel_tol": self.rel_tol,
                "common_rows": self.intersect_rows.shape[0],
                "df1_unique": self.df1_unq_rows.shape[0],
                "df2_unique": self.df2_unq_rows.shape[0],
                "unequal_rows": self.intersect_rows.shape[0]
                - self.count_matching_rows(),
                "equal_rows": self.count_matching_rows(),
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
        mismatch_stats = []
        match_sample = []
        any_mismatch = False

        for column in self.column_stats:
            if not column["all_match"]:
                any_mismatch = True
                mismatch_stats.append(
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
                    "stats": sorted(mismatch_stats, key=lambda x: x["column"]),
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
        min_sample_count_df1 = min(sample_count, self.df1_unq_rows.shape[0])
        min_sample_count_df2 = min(sample_count, self.df2_unq_rows.shape[0])
        min_column_count_df1 = min(column_count, self.df1_unq_rows.shape[1])
        min_column_count_df2 = min(column_count, self.df2_unq_rows.shape[1])

        return {
            "df1_unique_rows": {
                "has_rows": min_sample_count_df1 > 0,
                "rows": df_to_str(
                    self.df1_unq_rows.iloc[:, :min_column_count_df1],
                    sample_count=min_sample_count_df1,
                )
                if self.df1_unq_rows.shape[0] > 0
                else "",
                "columns": list(self.df1_unq_rows.columns[:min_column_count_df1])
                if self.df1_unq_rows.shape[0] > 0
                else "",
            },
            "df2_unique_rows": {
                "has_rows": min_sample_count_df2 > 0,
                "rows": df_to_str(
                    self.df2_unq_rows.iloc[:, :min_column_count_df2],
                    sample_count=min_sample_count_df2,
                )
                if self.df2_unq_rows.shape[0] > 0
                else "",
                "columns": list(self.df2_unq_rows.columns[:min_column_count_df2])
                if self.df2_unq_rows.shape[0] > 0
                else "",
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

        Examples
        --------
        Basic usage with default template:

        >>> compare = datacompy.Compare(df1, df2, join_columns=['id'])
        >>> report = compare.report()
        >>> print(report)

        Using a custom template file:

        >>> # Create a custom template file (custom_report.j2)
        >>> with open('custom_report.j2', 'w') as f:
        ...     f.write('''
        ...     Comparison Report
        ...     ================
        ...
        ...     DataFrames: {{ df1_name }} vs {{ df2_name }}
        ...
        ...     Shape Summary:
        ...     - {{ df1_name }}: {{ df1_shape[0] }} rows x {{ df1_shape[1] }} columns
        ...     - {{ df2_name }}: {{ df2_shape[0] }} rows x {{ df2_shape[1] }} columns
        ...
        ...     {% if mismatch_stats %}
        ...     Mismatched Columns ({{ mismatch_stats|length }}):
        ...     {% for col in mismatch_stats %}
        ...     - {{ col.column }} ({{ col.unequal_cnt }} mismatches)
        ...     {% endfor %}
        ...     {% else %}
        ...     No mismatches found in any columns!
        ...     {% endif %}
        ...     ''')
        ...
        >>> # Generate report with custom template
        >>> report = compare.report(template_path='custom_report.j2')
        >>> print(report)
        """
        # Prepare the template data by combining all sections
        template_data = {
            **self._get_column_summary(),
            **self._get_row_summary(),
            **self._get_column_comparison(),
            **self._get_mismatch_stats(sample_count),
            **self._get_unique_rows_data(sample_count, column_count),
            "df1_name": self.df1_name,
            "df2_name": self.df2_name,
            "df1_shape": self.df1.shape,
            "df2_shape": self.df2.shape,
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
    col_1: "pd.Series[Any]",
    col_2: "pd.Series[Any]",
    rel_tol: float = 0,
    abs_tol: float = 0,
    ignore_spaces: bool = False,
    ignore_case: bool = False,
) -> "pd.Series[bool]":
    """Compare two columns from a dataframe.

    Returns a True/False series,
    with the same index as column 1.

    - Two nulls (np.nan) will evaluate to True.
    - A null and a non-null value will evaluate to False.
    - Numeric values will use the relative and absolute tolerances.
    - Decimal values (decimal.Decimal) will attempt to be converted to floats
      before comparing
    - Non-numeric values (i.e. where np.isclose can't be used) will just
      trigger True on two nulls or exact matches.

    Notes
    -----
    As of version ``0.14.0`` If a column is of a mixed data type the compare will
    default to returning ``False``.

    Notes
    -----
    - ``list`` and ``np.array`` types will be compared row wise using ``np.array_equal``.
      Depending on the size of your data this might lead to performance issues.
    - All the rows must be of the same type otherwise it is considered "mixed"
      and will default to being ``False`` for everything.

    Parameters
    ----------
    col_1 : Pandas.Series
        The first column to look at
    col_2 : Pandas.Series
        The second column
    rel_tol : float, optional
        Relative tolerance
    abs_tol : float, optional
        Absolute tolerance
    ignore_spaces : bool, optional
        Flag to strip whitespace (including newlines) from string columns
    ignore_case : bool, optional
        Flag to ignore the case of string columns

    Returns
    -------
    pandas.Series
        A series of Boolean values.  True == the values match, False == the
        values don't match.
    """
    default_value = "DATACOMPY_NULL"
    compare: pd.Series[bool]

    col_1 = normalize_string_column(
        col_1, ignore_spaces=ignore_spaces, ignore_case=ignore_case
    )
    col_2 = normalize_string_column(
        col_2, ignore_spaces=ignore_spaces, ignore_case=ignore_case
    )

    # Rest of comparison logic using rel_tol and abs_tol
    # short circuit if comparing mixed type columns. Check list/arrrays or just return false for everything else.
    if pd.api.types.infer_dtype(col_1).startswith("mixed") or pd.api.types.infer_dtype(
        col_2
    ).startswith("mixed"):
        if all(isinstance(item, list | np.ndarray) for item in col_1) and all(
            isinstance(item, list | np.ndarray) for item in col_2
        ):  # compare list like
            # join together and apply np.array_equal
            temp_df = pd.DataFrame({"col_1": col_1, "col_2": col_2})
            compare = temp_df.apply(
                lambda row: np.array_equal(row.col_1, row.col_2, equal_nan=True), axis=1
            )
        else:
            compare = pd.Series(False, index=col_1.index)
    elif pd.api.types.is_string_dtype(col_1) and pd.api.types.is_string_dtype(col_2):
        try:
            compare = pd.Series(
                (col_1.fillna(default_value) == col_2.fillna(default_value))
                | (col_1.isnull() & col_2.isnull())
            )
        except TypeError:
            compare = pd.Series(col_1.astype(str) == col_2.astype(str))
    elif {col_1.dtype.kind, col_2.dtype.kind} == {"M", "O"}:
        compare = compare_string_and_date_columns(col_1, col_2)
    else:
        try:
            compare = pd.Series(
                np.isclose(col_1, col_2, rtol=rel_tol, atol=abs_tol, equal_nan=True)
            )
        except TypeError:
            try:
                compare = pd.Series(
                    np.isclose(
                        col_1.astype(float),
                        col_2.astype(float),
                        rtol=rel_tol,
                        atol=abs_tol,
                        equal_nan=True,
                    )
                )
            except Exception:
                try:  # last check where we just cast to strings
                    compare = pd.Series(col_1.astype(str) == col_2.astype(str))
                except Exception:  # Blanket exception should just return all False
                    compare = pd.Series(False, index=col_1.index)
    compare.index = col_1.index
    return compare


def compare_string_and_date_columns(
    col_1: "pd.Series[Any]", col_2: "pd.Series[Any]"
) -> "pd.Series[bool]":
    """Compare a string column and date column, value-wise.

    This tries to
    convert a string column to a date column and compare that way.

    Parameters
    ----------
    col_1 : Pandas.Series
        The first column to look at
    col_2 : Pandas.Series
        The second column

    Returns
    -------
    pandas.Series
        A series of Boolean values.  True == the values match, False == the
        values don't match.
    """
    if col_1.dtype.kind == "O":
        obj_column = col_1
        date_column = col_2
    else:
        obj_column = col_2
        date_column = col_1

    try:
        return pd.Series(
            (pd.to_datetime(obj_column) == date_column)
            | (obj_column.isnull() & date_column.isnull())
        )
    except Exception:
        try:
            return pd.Series(
                (pd.to_datetime(obj_column, format="mixed") == date_column)
                | (obj_column.isnull() & date_column.isnull())
            )
        except Exception:
            return pd.Series(False, index=col_1.index)


def get_merged_columns(
    original_df: pd.DataFrame, merged_df: pd.DataFrame, suffix: str
) -> List[str]:
    """Get the columns from an original dataframe, in the new merged dataframe.

    Parameters
    ----------
    original_df : Pandas.DataFrame
        The original, pre-merge dataframe
    merged_df : Pandas.DataFrame
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


def calculate_max_diff(col_1: "pd.Series[Any]", col_2: "pd.Series[Any]") -> float:
    """Get a maximum difference between two columns.

    Parameters
    ----------
    col_1 : Pandas.Series
        The first column
    col_2 : Pandas.Series
        The second column

    Returns
    -------
    Numeric
        Numeric field, or zero.
    """
    try:
        return cast(float, (col_1.astype(float) - col_2.astype(float)).abs().max())
    except Exception:
        return 0.0


def generate_id_within_group(
    dataframe: pd.DataFrame, join_columns: List[str]
) -> "pd.Series[int]":
    """Generate an ID column that can be used to deduplicate identical rows.

    The series generated
    is the order within a unique group, and it handles nulls.

    Parameters
    ----------
    dataframe : Pandas.DataFrame
        The dataframe to operate on
    join_columns : list
        List of strings which are the join columns

    Returns
    -------
    Pandas.Series
        The ID column that's unique in each group.
    """
    default_value = "DATACOMPY_NULL"
    if dataframe[join_columns].isnull().any().any():
        if (dataframe[join_columns] == default_value).any().any():
            raise ValueError(f"{default_value} was found in your join columns")
        return (
            dataframe[join_columns]
            .astype(str)
            .fillna(default_value)
            .groupby(join_columns)
            .cumcount()
        )
    else:
        return dataframe[join_columns].groupby(join_columns).cumcount()


def normalize_string_column(
    column: pd.Series, ignore_spaces: bool, ignore_case: bool
) -> pd.Series:
    """Normalize a string column by converting to upper case and stripping whitespace.

    Parameters
    ----------
    column : pd.Series
        The column to normalize
    ignore_spaces : bool
        Whether to ignore spaces when normalizing
    ignore_case : bool
        Whether to ignore case when normalizing

    Returns
    -------
    pd.Series
        The normalized column

    Notes
    -----
    Will not operate on categorical columns.
    """
    if (column.dtype.kind == "O" and pd.api.types.infer_dtype(column) == "string") or (
        pd.api.types.is_string_dtype(column)
        and not isinstance(column.dtype, pd.CategoricalDtype)
    ):
        column = column.str.strip() if ignore_spaces else column
        column = column.str.upper() if ignore_case else column
    return column
