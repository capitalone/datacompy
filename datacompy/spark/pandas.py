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
Compare two Pandas on Spark DataFrames

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""

import logging
import os
from typing import List, Optional, Union

import pandas as pd
from ordered_set import OrderedSet

from ..base import BaseCompare, temp_column_name

try:
    import pyspark.pandas as ps
    from pandas.api.types import is_numeric_dtype
except ImportError:
    pass  # Let non-Spark people at least enjoy the loveliness of the pandas datacompy functionality


LOG = logging.getLogger(__name__)


class SparkPandasCompare(BaseCompare):
    """Comparison class to be used to compare whether two Pandas on Spark dataframes are equal.

    Both df1 and df2 should be dataframes containing all of the join_columns,
    with unique column names. Differences between values are compared to
    abs_tol + rel_tol * abs(df2['value']).

    Parameters
    ----------
    df1 : pyspark.pandas.frame.DataFrame
        First dataframe to check
    df2 : pyspark.pandas.frame.DataFrame
        Second dataframe to check
    join_columns : list or str, optional
        Column(s) to join dataframes on.  If a string is passed in, that one
        column will be used.
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

    Attributes
    ----------
    df1_unq_rows : pyspark.pandas.frame.DataFrame
        All records that are only in df1 (based on a join on join_columns)
    df2_unq_rows : pyspark.pandas.frame.DataFrame
        All records that are only in df2 (based on a join on join_columns)
    """

    def __init__(
        self,
        df1: "ps.DataFrame",
        df2: "ps.DataFrame",
        join_columns: Union[List[str], str],
        abs_tol: float = 0,
        rel_tol: float = 0,
        df1_name: str = "df1",
        df2_name: str = "df2",
        ignore_spaces: bool = False,
        ignore_case: bool = False,
        cast_column_names_lower: bool = True,
    ) -> None:
        if pd.__version__ >= "2.0.0":
            raise Exception(
                "It seems like you are running Pandas 2+. Please note that Pandas 2+ will only be supported in Spark 4+. See: https://issues.apache.org/jira/browse/SPARK-44101. If you need to use Spark DataFrame with Pandas 2+ then consider using Fugue otherwise downgrade to Pandas 1.5.3"
            )

        ps.set_option("compute.ops_on_diff_frames", True)
        self.cast_column_names_lower = cast_column_names_lower
        if isinstance(join_columns, (str, int, float)):
            self.join_columns = [
                (
                    str(join_columns).lower()
                    if self.cast_column_names_lower
                    else str(join_columns)
                )
            ]
        else:
            self.join_columns = [
                str(col).lower() if self.cast_column_names_lower else str(col)
                for col in join_columns
            ]

        self._any_dupes: bool = False
        self.df1 = df1
        self.df2 = df2
        self.df1_name = df1_name
        self.df2_name = df2_name
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.ignore_spaces = ignore_spaces
        self.ignore_case = ignore_case
        self.df1_unq_rows = self.df2_unq_rows = self.intersect_rows = None
        self.column_stats: List = []
        self._compare(ignore_spaces, ignore_case)

    @property
    def df1(self) -> "ps.DataFrame":
        return self._df1

    @df1.setter
    def df1(self, df1: "ps.DataFrame") -> None:
        """Check that it is a dataframe and has the join columns"""
        self._df1 = df1
        self._validate_dataframe(
            "df1", cast_column_names_lower=self.cast_column_names_lower
        )

    @property
    def df2(self) -> "ps.DataFrame":
        return self._df2

    @df2.setter
    def df2(self, df2: "ps.DataFrame") -> None:
        """Check that it is a dataframe and has the join columns"""
        self._df2 = df2
        self._validate_dataframe(
            "df2", cast_column_names_lower=self.cast_column_names_lower
        )

    def _validate_dataframe(
        self, index: str, cast_column_names_lower: bool = True
    ) -> None:
        """Check that it is a dataframe and has the join columns

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
        if not isinstance(dataframe, (ps.DataFrame)):
            raise TypeError(f"{index} must be a pyspark.pandas.frame.DataFrame")

        if cast_column_names_lower:
            dataframe.columns = [str(col).lower() for col in dataframe.columns]
        else:
            dataframe.columns = [str(col) for col in dataframe.columns]
        # Check if join_columns are present in the dataframe
        if not set(self.join_columns).issubset(set(dataframe.columns)):
            raise ValueError(f"{index} must have all columns from join_columns")

        if len(set(dataframe.columns)) < len(dataframe.columns):
            raise ValueError(f"{index} must have unique column names")

        if len(dataframe.drop_duplicates(subset=self.join_columns)) < len(dataframe):
            self._any_dupes = True

    def _compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        """Actually run the comparison.  This tries to run df1.equals(df2)
        first so that if they're truly equal we can tell.

        This method will log out information about what is different between
        the two dataframes, and will also return a boolean.
        """
        LOG.debug("Checking equality")
        if self.df1.equals(self.df2).all().all():
            LOG.info("df1 pyspark.pandas.frame.DataFrame.equals df2")
        else:
            LOG.info("df1 does not pyspark.pandas.frame.DataFrame.equals df2")
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
        # cache
        self.df1.spark.cache()
        self.df2.spark.cache()

        LOG.debug("Merging dataframes")
        self._dataframe_merge(ignore_spaces)
        self._intersect_compare(ignore_spaces, ignore_case)
        if self.matches():
            LOG.info("df1 matches df2")
        else:
            LOG.info("df1 does not match df2")

    def df1_unq_columns(self) -> OrderedSet[str]:
        """Get columns that are unique to df1"""
        return OrderedSet(self.df1.columns) - OrderedSet(self.df2.columns)

    def df2_unq_columns(self) -> OrderedSet[str]:
        """Get columns that are unique to df2"""
        return OrderedSet(self.df2.columns) - OrderedSet(self.df1.columns)

    def intersect_columns(self) -> OrderedSet[str]:
        """Get columns that are shared between the two dataframes"""
        return OrderedSet(self.df1.columns) & OrderedSet(self.df2.columns)

    def _dataframe_merge(self, ignore_spaces: bool) -> None:
        """Merge df1 to df2 on the join columns, to get df1 - df2, df2 - df1
        and df1 & df2
        """

        LOG.debug("Outer joining")

        df1 = self.df1.copy()
        df2 = self.df2.copy()

        if self._any_dupes:
            LOG.debug("Duplicate rows found, deduping by order of remaining fields")
            temp_join_columns = list(self.join_columns)

            # Create order column for uniqueness of match
            order_column = temp_column_name(df1, df2)
            df1[order_column] = generate_id_within_group(df1, temp_join_columns)
            df2[order_column] = generate_id_within_group(df2, temp_join_columns)
            temp_join_columns.append(order_column)

            params = {"on": temp_join_columns}
        else:
            params = {"on": self.join_columns}

        if ignore_spaces:
            for column in self.join_columns:
                if df1[column].dtype.kind == "O":
                    df1[column] = df1[column].str.strip()
                if df2[column].dtype.kind == "O":
                    df2[column] = df2[column].str.strip()

        non_join_columns = (
            OrderedSet(df1.columns) | OrderedSet(df2.columns)
        ) - OrderedSet(self.join_columns)

        for c in non_join_columns:
            df1.rename(columns={c: c + "_" + self.df1_name}, inplace=True)
            df2.rename(columns={c: c + "_" + self.df2_name}, inplace=True)

        # generate merge indicator
        df1["_merge_left"] = True
        df2["_merge_right"] = True

        for c in self.join_columns:
            df1.rename(columns={c: c + "_" + self.df1_name}, inplace=True)
            df2.rename(columns={c: c + "_" + self.df2_name}, inplace=True)

        # cache
        df1.spark.cache()
        df2.spark.cache()

        # NULL SAFE Outer join using ON
        on = " and ".join(
            [
                f"df1.`{c}_{self.df1_name}` <=> df2.`{c}_{self.df2_name}`"
                for c in params["on"]
            ]
        )
        outer_join = ps.sql(
            """
        SELECT * FROM
        {df1} df1 FULL OUTER JOIN {df2} df2
        ON     
        """
            + on,
            df1=df1,
            df2=df2,
        )

        outer_join["_merge"] = None  # initialize col

        # process merge indicator
        outer_join["_merge"] = outer_join._merge.mask(
            (outer_join["_merge_left"] == True)  # noqa: E712
            & (outer_join["_merge_right"] == True),  # noqa: E712
            "both",
        )
        outer_join["_merge"] = outer_join._merge.mask(
            (outer_join["_merge_left"] == True)  # noqa: E712
            & (outer_join["_merge_right"] != True),  # noqa: E712
            "left_only",
        )
        outer_join["_merge"] = outer_join._merge.mask(
            (outer_join["_merge_left"] != True)  # noqa: E712
            & (outer_join["_merge_right"] == True),  # noqa: E712
            "right_only",
        )

        # Clean up temp columns for duplicate row matching
        if self._any_dupes:
            outer_join = outer_join.drop(
                [
                    order_column + "_" + self.df1_name,
                    order_column + "_" + self.df2_name,
                ],
                axis=1,
            )
            df1 = df1.drop(
                [
                    order_column + "_" + self.df1_name,
                    order_column + "_" + self.df2_name,
                ],
                axis=1,
            )
            df2 = df2.drop(
                [
                    order_column + "_" + self.df1_name,
                    order_column + "_" + self.df2_name,
                ],
                axis=1,
            )

        df1_cols = get_merged_columns(df1, outer_join, self.df1_name)
        df2_cols = get_merged_columns(df2, outer_join, self.df2_name)

        LOG.debug("Selecting df1 unique rows")
        self.df1_unq_rows = outer_join[outer_join["_merge"] == "left_only"][
            df1_cols
        ].copy()

        LOG.debug("Selecting df2 unique rows")
        self.df2_unq_rows = outer_join[outer_join["_merge"] == "right_only"][
            df2_cols
        ].copy()

        LOG.info(f"Number of rows in df1 and not in df2: {len(self.df1_unq_rows)}")
        LOG.info(f"Number of rows in df2 and not in df1: {len(self.df2_unq_rows)}")

        LOG.debug("Selecting intersecting rows")
        self.intersect_rows = outer_join[outer_join["_merge"] == "both"].copy()
        LOG.info(
            "Number of rows in df1 and df2 (not necessarily equal): {len(self.intersect_rows)}"
        )
        # cache
        self.intersect_rows.spark.cache()

    def _intersect_compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        """Run the comparison on the intersect dataframe

        This loops through all columns that are shared between df1 and df2, and
        creates a column column_match which is True for matches, False
        otherwise.
        """
        LOG.debug("Comparing intersection")
        max_diff: float
        null_diff: int
        row_cnt = len(self.intersect_rows)
        for column in self.intersect_columns():
            if column in self.join_columns:
                match_cnt = row_cnt
                col_match = ""
                max_diff = 0
                null_diff = 0
            else:
                col_1 = column + "_" + self.df1_name
                col_2 = column + "_" + self.df2_name
                col_match = column + "_match"
                self.intersect_rows[col_match] = columns_equal(
                    self.intersect_rows[col_1],
                    self.intersect_rows[col_2],
                    self.rel_tol,
                    self.abs_tol,
                    ignore_spaces,
                    ignore_case,
                )
                match_cnt = self.intersect_rows[col_match].sum()
                max_diff = calculate_max_diff(
                    self.intersect_rows[col_1], self.intersect_rows[col_2]
                )

                try:
                    null_diff = (
                        (self.intersect_rows[col_1].isnull())
                        ^ (self.intersect_rows[col_2].isnull())
                    ).sum()
                except TypeError:  # older pyspark compatibility
                    temp_null_diff = self.intersect_rows[[col_1, col_2]].isnull()
                    null_diff = (temp_null_diff[col_1] != temp_null_diff[col_2]).sum()

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
                }
            )

    def all_columns_match(self) -> bool:
        """Whether the columns all match in the dataframes"""
        return self.df1_unq_columns() == self.df2_unq_columns() == set()

    def all_rows_overlap(self) -> bool:
        """Whether the rows are all present in both dataframes

        Returns
        -------
        bool
            True if all rows in df1 are in df2 and vice versa (based on
            existence for join option)
        """
        return len(self.df1_unq_rows) == len(self.df2_unq_rows) == 0

    def count_matching_rows(self) -> bool:
        """Count the number of rows match (on overlapping fields)

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
            match_columns_count = (
                self.intersect_rows[match_columns]
                .query(" and ".join(conditions))
                .shape[0]
            )
        else:
            match_columns_count = 0
        return match_columns_count

    def intersect_rows_match(self) -> bool:
        """Check whether the intersect rows all match"""
        actual_length = self.intersect_rows.shape[0]
        return self.count_matching_rows() == actual_length

    def matches(self, ignore_extra_columns: bool = False) -> bool:
        """Return True or False if the dataframes match.

        Parameters
        ----------
        ignore_extra_columns : bool
            Ignores any columns in one dataframe and not in the other.
        """
        if not ignore_extra_columns and not self.all_columns_match():
            return False
        elif not self.all_rows_overlap():
            return False
        elif not self.intersect_rows_match():
            return False
        else:
            return True

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
        if not self.df2_unq_columns() == set():
            return False
        elif not len(self.df2_unq_rows) == 0:
            return False
        elif not self.intersect_rows_match():
            return False
        else:
            return True

    def sample_mismatch(
        self, column: str, sample_count: int = 10, for_display: bool = False
    ) -> "ps.DataFrame":
        """Returns a sample sub-dataframe which contains the identifying
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
        pyspark.pandas.frame.DataFrame
            A sample of the intersection dataframe, containing only the
            "pertinent" columns, for rows that don't match on the provided
            column.
        """
        row_cnt = self.intersect_rows.shape[0]
        col_match = self.intersect_rows[column + "_match"]
        match_cnt = col_match.sum()
        sample_count = min(sample_count, row_cnt - match_cnt)
        sample = self.intersect_rows[~col_match].head(sample_count)

        for c in self.join_columns:
            sample[c] = sample[c + "_" + self.df1_name]

        return_cols = self.join_columns + [
            column + "_" + self.df1_name,
            column + "_" + self.df2_name,
        ]
        to_return = sample[return_cols]
        if for_display:
            to_return.columns = self.join_columns + [
                column + " (" + self.df1_name + ")",
                column + " (" + self.df2_name + ")",
            ]
        return to_return

    def all_mismatch(self, ignore_matching_cols: bool = False) -> "ps.DataFrame":
        """All rows with any columns that have a mismatch. Returns all df1 and df2 versions of the columns and join
        columns.

        Parameters
        ----------
        ignore_matching_cols : bool, optional
            Whether showing the matching columns in the output or not. The default is False.

        Returns
        -------
        pyspark.pandas.frame.DataFrame
            All rows of the intersection dataframe, containing any columns, that don't match.
        """
        match_list = []
        return_list = []
        for col in self.intersect_rows.columns:
            if col.endswith("_match"):
                orig_col_name = col[:-6]

                col_comparison = columns_equal(
                    self.intersect_rows[orig_col_name + "_" + self.df1_name],
                    self.intersect_rows[orig_col_name + "_" + self.df2_name],
                    self.rel_tol,
                    self.abs_tol,
                    self.ignore_spaces,
                    self.ignore_case,
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

        mm_bool = self.intersect_rows[match_list].T.all()

        updated_join_columns = []
        for c in self.join_columns:
            updated_join_columns.append(c + "_" + self.df1_name)
            updated_join_columns.append(c + "_" + self.df2_name)

        return self.intersect_rows[~mm_bool][updated_join_columns + return_list]

    def report(
        self,
        sample_count: int = 10,
        column_count: int = 10,
        html_file: Optional[str] = None,
    ) -> str:
        """Returns a string representation of a report.  The representation can
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
        df_header = ps.DataFrame(
            {
                "DataFrame": [self.df1_name, self.df2_name],
                "Columns": [self.df1.shape[1], self.df2.shape[1]],
                "Rows": [self.df1.shape[0], self.df2.shape[0]],
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
            self.intersect_rows.shape[0],
            self.df1_unq_rows.shape[0],
            self.df2_unq_rows.shape[0],
            self.intersect_rows.shape[0] - self.count_matching_rows(),
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
            sum([col["unequal_cnt"] for col in self.column_stats]),
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
            df_match_stats = ps.DataFrame(match_stats)
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
                    report += sample.to_string()
                    report += "\n\n"

        if min(sample_count, self.df1_unq_rows.shape[0]) > 0:
            report += (
                f"Sample Rows Only in {self.df1_name} (First {column_count} Columns)\n"
            )
            report += (
                f"---------------------------------------{'-' * len(self.df1_name)}\n"
            )
            report += "\n"
            columns = self.df1_unq_rows.columns[:column_count]
            unq_count = min(sample_count, self.df1_unq_rows.shape[0])
            report += self.df1_unq_rows.head(unq_count)[columns].to_string()
            report += "\n\n"

        if min(sample_count, self.df2_unq_rows.shape[0]) > 0:
            report += (
                f"Sample Rows Only in {self.df2_name} (First {column_count} Columns)\n"
            )
            report += (
                f"---------------------------------------{'-' * len(self.df2_name)}\n"
            )
            report += "\n"
            columns = self.df2_unq_rows.columns[:column_count]
            unq_count = min(sample_count, self.df2_unq_rows.shape[0])
            report += self.df2_unq_rows.head(unq_count)[columns].to_string()
            report += "\n\n"

        if html_file:
            html_report = report.replace("\n", "<br>").replace(" ", "&nbsp;")
            html_report = f"<pre>{html_report}</pre>"
            with open(html_file, "w") as f:
                f.write(html_report)

        return report


def render(filename: str, *fields: Union[int, float, str]) -> str:
    """Renders out an individual template.  This basically just reads in a
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
    with open(os.path.join(this_dir, "..", "templates", filename)) as file_open:
        return file_open.read().format(*fields)


def columns_equal(
    col_1: "ps.Series",
    col_2: "ps.Series",
    rel_tol: float = 0,
    abs_tol: float = 0,
    ignore_spaces: bool = False,
    ignore_case: bool = False,
) -> "ps.Series":
    """Compares two columns from a dataframe, returning a True/False series,
    with the same index as column 1.

    - Two nulls (np.nan) will evaluate to True.
    - A null and a non-null value will evaluate to False.
    - Numeric values will use the relative and absolute tolerances.
    - Decimal values (decimal.Decimal) will attempt to be converted to floats
      before comparing
    - Non-numeric values (i.e. where np.isclose can't be used) will just
      trigger True on two nulls or exact matches.

    Parameters
    ----------
    col_1 : pyspark.pandas.series.Series
        The first column to look at
    col_2 : pyspark.pandas.series.Series
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
    pyspark.pandas.series.Series
        A series of Boolean values.  True == the values match, False == the
        values don't match.
    """
    try:
        compare = ((col_1 - col_2).abs() <= abs_tol + (rel_tol * col_2.abs())) | (
            col_1.isnull() & col_2.isnull()
        )
    except TypeError:
        if (
            is_numeric_dtype(col_1.dtype.kind) and is_numeric_dtype(col_2.dtype.kind)
        ) or (
            col_1.spark.data_type.typeName() == "decimal"
            and col_2.spark.data_type.typeName() == "decimal"
        ):
            compare = (
                (col_1.astype(float) - col_2.astype(float)).abs()
                <= abs_tol + (rel_tol * col_2.astype(float).abs())
            ) | (col_1.astype(float).isnull() & col_2.astype(float).isnull())
        else:
            try:
                col_1_temp = col_1.copy()
                col_2_temp = col_2.copy()
                if ignore_spaces:
                    if col_1.dtype.kind == "O":
                        col_1_temp = col_1_temp.str.strip()
                    if col_2.dtype.kind == "O":
                        col_2_temp = col_2_temp.str.strip()

                if ignore_case:
                    if col_1.dtype.kind == "O":
                        col_1_temp = col_1_temp.str.upper()
                    if col_2.dtype.kind == "O":
                        col_2_temp = col_2_temp.str.upper()

                if {col_1.dtype.kind, col_2.dtype.kind} == {"M", "O"}:
                    compare = compare_string_and_date_columns(col_1_temp, col_2_temp)
                else:
                    compare = (col_1_temp == col_2_temp) | (
                        col_1_temp.isnull() & col_2_temp.isnull()
                    )

            except Exception:
                # Blanket exception should just return all False
                compare = ps.Series(False, index=col_1.index.to_numpy())
    return compare


def compare_string_and_date_columns(
    col_1: "ps.Series", col_2: "ps.Series"
) -> "ps.Series":
    """Compare a string column and date column, value-wise.  This tries to
    convert a string column to a date column and compare that way.

    Parameters
    ----------
    col_1 : pyspark.pandas.series.Series
        The first column to look at
    col_2 : pyspark.pandas.series.Series
        The second column

    Returns
    -------
    pyspark.pandas.series.Series
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
        compare = ps.Series(
            (
                (ps.to_datetime(obj_column) == date_column)
                | (obj_column.isnull() & date_column.isnull())
            ).to_numpy()
        )  # force compute
    except Exception:
        compare = ps.Series(False, index=col_1.index.to_numpy())
    return compare


def get_merged_columns(
    original_df: "ps.DataFrame",
    merged_df: "ps.DataFrame",
    suffix: str,
) -> List[str]:
    """Gets the columns from an original dataframe, in the new merged dataframe

    Parameters
    ----------
    original_df : pyspark.pandas.frame.DataFrame
        The original, pre-merge dataframe
    merged_df : pyspark.pandas.frame.DataFrame
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
    for col in original_df.columns:
        if col in merged_df.columns:
            columns.append(col)
        elif col + "_" + suffix in merged_df.columns:
            columns.append(col + "_" + suffix)
        else:
            raise ValueError("Column not found: %s", col)
    return columns


def calculate_max_diff(col_1: "ps.DataFrame", col_2: "ps.DataFrame") -> float:
    """Get a maximum difference between two columns

    Parameters
    ----------
    col_1 : pyspark.pandas.series.Series
        The first column
    col_2 : pyspark.pandas.series.Series
        The second column

    Returns
    -------
    float
        max diff
    """
    try:
        return (col_1.astype(float) - col_2.astype(float)).abs().max()
    except Exception:
        return 0


def generate_id_within_group(
    dataframe: "ps.DataFrame", join_columns: List[str]
) -> "ps.Series":
    """Generate an ID column that can be used to deduplicate identical rows.  The series generated
    is the order within a unique group, and it handles nulls.

    Parameters
    ----------
    dataframe : pyspark.pandas.frame.DataFrame
        The dataframe to operate on
    join_columns : list
        List of strings which are the join columns

    Returns
    -------
    pyspark.pandas.series.Series
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
