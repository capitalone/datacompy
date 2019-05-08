# -*- coding: utf-8 -*-
#
# Copyright 2017 Capital One Services, LLC
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
Compare two Pandas DataFrames

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from datacompy import utils

LOG = logging.getLogger(__name__)


class Compare:
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
        Flag to strip whitespace (including newlines) from string columns
    ignore_case : bool, optional
        Flag to ignore the case of string columns

    Attributes
    ----------
    df1_unq_rows : pandas ``DataFrame``
        All records that are only in df1 (based on a join on join_columns)
    df2_unq_rows : pandas ``DataFrame``
        All records that are only in df2 (based on a join on join_columns)
    """

    def __init__(
        self,
        df1,
        df2,
        join_columns=None,
        on_index=False,
        abs_tol=0,
        rel_tol=0,
        df1_name="df1",
        df2_name="df2",
        ignore_spaces=False,
        ignore_case=False,
    ):

        if on_index and join_columns is not None:
            raise Exception("Only provide on_index or join_columns")
        elif on_index:
            self.on_index = True
            self.join_columns = []
        elif isinstance(join_columns, str):
            self.join_columns = [join_columns.lower()]
            self.on_index = False
        else:
            self.join_columns = [col.lower() for col in join_columns]
            self.on_index = False

        self._any_dupes = False
        self.df1 = df1
        self.df2 = df2
        self.df1_name = df1_name
        self.df2_name = df2_name
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.df1_unq_rows = self.df2_unq_rows = self.intersect_rows = None
        self._df1_row_count = self._df2_row_count = None
        self.column_stats = None
        self._compare(ignore_spaces, ignore_case)

    @property
    def df1(self):
        return self._df1

    @df1.setter
    def df1(self, df1):
        """Check that it is a dataframe and has the join columns"""
        self._df1 = df1
        self._validate_dataframe("df1")

    @property
    def df2(self):
        return self._df2

    @df2.setter
    def df2(self, df2):
        """Check that it is a dataframe and has the join columns"""
        self._df2 = df2
        self._validate_dataframe("df2")

    def _validate_dataframe(self, df1_or_df2):
        """Check that it is a dataframe and has the join columns

        Parameters
        ----------
        df1_or_df2 : str
            The "index" of the dataframe - df1 or df2.
        """
        dataframe = getattr(self, df1_or_df2)
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("{} must be a pandas DataFrame".format(df1_or_df2))

        dataframe.columns = [col.lower() for col in dataframe.columns]
        # Check if join_columns are present in the dataframe
        if not set(self.join_columns).issubset(set(dataframe.columns)):
            raise ValueError("{} must have all columns from join_columns".format(df1_or_df2))

        if len(set(dataframe.columns)) < len(dataframe.columns):
            raise ValueError("{} must have unique column names".format(df1_or_df2))

        if self.on_index:
            if dataframe.index.duplicated().sum() > 0:
                self._any_dupes = True
        else:
            if len(dataframe.drop_duplicates(subset=self.join_columns)) < len(dataframe):
                self._any_dupes = True

    def _compare(self, ignore_spaces, ignore_case):
        """Actually run the comparison.  This tries to run df1.equals(df2)
        first so that if they're truly equal we can tell.

        This method will log out information about what is different between
        the two dataframes, and will also return a boolean.
        """
        LOG.debug("Checking equality")
        if self.df1.equals(self.df2):
            LOG.info("df1 Pandas.DataFrame.equals df2")
        else:
            LOG.info("df1 does not Pandas.DataFrame.equals df2")
        LOG.info("Number of columns in common: {0}".format(len(self.intersect_columns)))
        LOG.debug("Checking column overlap")
        for col in self.df1_unq_columns:
            LOG.info("Column in df1 and not in df2: {0}".format(col))
        LOG.info("Number of columns in df1 and not in df2: {0}".format(len(self.df1_unq_columns)))
        for col in self.df2_unq_columns:
            LOG.info("Column in df2 and not in df1: {}".format(col))
        LOG.info("Number of columns in df2 and not in df1: {}".format(len(self.df2_unq_columns)))
        LOG.debug("Merging dataframes")
        self._dataframe_merge(ignore_spaces)
        self._intersect_compare(ignore_spaces, ignore_case)
        if self.matches():
            LOG.info("df1 matches df2")
        else:
            LOG.info("df1 does not match df2")

    @property
    def df1_unq_columns(self):
        """Get columns that are unique to df1 - works for Spark"""
        return set(self.df1.columns) - set(self.df2.columns)

    @property
    def df2_unq_columns(self):
        """Get columns that are unique to df2 - works for Spark"""
        return set(self.df2.columns) - set(self.df1.columns)

    @property
    def intersect_columns(self):
        """Get columns that are shared between the two dataframes - works for Spark."""
        return set(self.df1.columns) & set(self.df2.columns)

    def _dataframe_merge(self, ignore_spaces):
        """Merge df1 to df2 on the join columns, to get df1 - df2, df2 - df1
        and df1 & df2

        If ``on_index`` is True, this will join on index values, otherwise it
        will join on the ``join_columns``.
        """

        LOG.debug("Outer joining")
        if self._any_dupes:
            LOG.debug("Duplicate rows found, deduping by order of remaining fields")
            # Bring index into a column
            if self.on_index:
                index_column = utils.temp_column_name(self.df1, self.df2)
                self.df1[index_column] = self.df1.index
                self.df2[index_column] = self.df2.index
                temp_join_columns = [index_column]
            else:
                temp_join_columns = list(self.join_columns)

            # Create order column for uniqueness of match
            order_column = utils.temp_column_name(self.df1, self.df2)
            self.df1[order_column] = utils.generate_id_within_group(self.df1, temp_join_columns)
            self.df2[order_column] = utils.generate_id_within_group(self.df2, temp_join_columns)
            temp_join_columns.append(order_column)

            params = {"on": temp_join_columns}
        elif self.on_index:
            params = {"left_index": True, "right_index": True}
        else:
            params = {"on": self.join_columns}

        outer_join = self.df1.merge(
            self.df2, how="outer", suffixes=("_df1", "_df2"), indicator=True, **params
        )

        # Clean up temp columns for duplicate row matching
        if self._any_dupes:
            if self.on_index:
                outer_join.index = outer_join[index_column]
                outer_join.drop(index_column, axis=1, inplace=True)
                self.df1.drop(index_column, axis=1, inplace=True)
                self.df2.drop(index_column, axis=1, inplace=True)
            outer_join.drop(order_column, axis=1, inplace=True)
            self.df1.drop(order_column, axis=1, inplace=True)
            self.df2.drop(order_column, axis=1, inplace=True)

        df1_cols = utils.get_merged_columns(self.df1, outer_join, "_df1")
        df2_cols = utils.get_merged_columns(self.df2, outer_join, "_df2")

        LOG.debug("Selecting df1 unique rows")
        self.df1_unq_rows = outer_join[outer_join["_merge"] == "left_only"][df1_cols].copy()
        self.df1_unq_rows.columns = self.df1.columns

        LOG.debug("Selecting df2 unique rows")
        self.df2_unq_rows = outer_join[outer_join["_merge"] == "right_only"][df2_cols].copy()
        self.df2_unq_rows.columns = self.df2.columns
        LOG.info("Number of rows in df1 and not in df2: {}".format(len(self.df1_unq_rows)))
        LOG.info("Number of rows in df2 and not in df1: {}".format(len(self.df2_unq_rows)))

        LOG.debug("Selecting intersecting rows")
        self.intersect_rows = outer_join[outer_join["_merge"] == "both"].copy()
        LOG.info(
            "Number of rows in df1 and df2 (not necessarily equal): {}".format(
                self.intersect_rows_count
            )
        )

    def _intersect_compare(self, ignore_spaces, ignore_case):
        """Run the comparison on the intersect dataframe

        This loops through all columns that are shared between df1 and df2, and creates a column
        column_match which is True for matches, False otherwise.  Also sets the self.column_stats
        dataframe.
        """
        LOG.debug("Comparing intersection")
        row_cnt = self.intersect_rows_count
        column_stats_temp = []
        for column in self.intersect_columns:
            if column in self.join_columns:
                match_cnt = row_cnt
                col_match = ""
                max_diff = 0
                null_diff = 0
            else:
                col_1 = column + "_df1"
                col_2 = column + "_df2"
                col_match = column + "_match"
                self.intersect_rows[col_match] = utils.columns_equal(
                    self.intersect_rows[col_1],
                    self.intersect_rows[col_2],
                    self.rel_tol,
                    self.abs_tol,
                    ignore_spaces,
                    ignore_case,
                )
                match_cnt = self.intersect_rows[col_match].sum()
                max_diff = utils.calculate_max_diff(
                    self.intersect_rows[col_1], self.intersect_rows[col_2]
                )
                null_diff = (
                    (self.intersect_rows[col_1].isnull()) ^ (self.intersect_rows[col_2].isnull())
                ).sum()

            if row_cnt > 0:
                match_rate = float(match_cnt) / row_cnt
            else:
                match_rate = 0
            LOG.info(
                "{0}: {1} / {2} ({3:.2%}) match".format(column, match_cnt, row_cnt, match_rate)
            )

            column_stats_temp.append(
                [
                    column,
                    col_match,
                    match_cnt,
                    row_cnt - match_cnt,
                    str(self.df1[column].dtype),
                    str(self.df2[column].dtype),
                    all((self.df1[column].dtype == self.df2[column].dtype, row_cnt == match_cnt)),
                    max_diff,
                    null_diff,
                    -1,  # TODO: implement this
                ]
            )
        self.column_stats = pd.DataFrame(
            column_stats_temp,
            columns=[
                "column",
                "match_column",
                "match_cnt",
                "unequal_cnt",
                "dtype1",
                "dtype2",
                "all_match",
                "max_diff",
                "null_diff",
                "known_diff_cnt",
            ],
        )

    def all_columns_match(self):
        """Whether the columns all match in the dataframes"""
        return self.df1_unq_columns == self.df2_unq_columns == set()

    def all_rows_overlap(self):
        """Whether the rows are all present in both dataframes

        Returns
        -------
        bool
            True if all rows in df1 are in df2 and vice versa (based on
            existence for join option)
        """
        return len(self.df1_unq_rows) == len(self.df2_unq_rows) == 0

    @property
    def matching_rows_count(self):
        """Count the number of rows match (on overlapping fields)

        Returns
        -------
        int
            Number of matching rows
        """
        match_columns = [
            col + "_match" for col in self.intersect_columns if col not in self.join_columns
        ]
        return self.intersect_rows[match_columns].all(axis=1).sum()

    def df_row_count(self, df1_or_df2):
        """Get the count of rows in a dataframe.  Is overwritten in Spark."""
        return getattr(self, df1_or_df2).shape[0]

    def df_column_cnt(self, df1_or_df2):
        """Get number of columns in a dataframe.  Should work for Pandas and Spark."""
        return len(getattr(self, df1_or_df2).columns)

    @property
    def intersect_rows_count(self):
        return self.intersect_rows.shape[0]

    def intersect_rows_match(self):
        """Check whether the intersect rows all match"""
        return self.matching_rows_count == self.intersect_rows_count

    def matches(self, ignore_extra_columns=False):
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

    def subset(self):
        """Return True if dataframe 2 is a subset of dataframe 1.

        Dataframe 2 is considered a subset if all of its columns are in
        dataframe 1, and all of its rows match rows in dataframe 1 for the
        shared columns.
        """
        if not self.df2_unq_columns == set():
            return False
        elif not len(self.df2_unq_rows) == 0:
            return False
        elif not self.intersect_rows_match():
            return False
        else:
            return True

    def sample_mismatch(self, column, sample_count=10, for_display=False):
        """Returns a sample sub-dataframe which contains the identifying columns, and df1 and df2
        versions of the column.

        Parameters
        ----------
        column : str
            The raw column name (i.e. without ``_df1`` appended)
        sample_count : int, optional
            The number of sample records to return.  Defaults to 10.
        for_display : bool, optional
            Whether this is just going to be used for display (overwrite the column names)

        Returns
        -------
        Pandas.DataFrame
            A sample of the intersection dataframe, containing only the "pertinent" columns, for
            rows that don't match on the provided column.
        """
        col_match = self.intersect_rows[column + "_match"]
        match_cnt = col_match.sum()
        sample_count = min(sample_count, self.intersect_rows_count - match_cnt)
        sample = self.intersect_rows[~col_match].sample(sample_count)
        return_cols = self.join_columns + [column + "_df1", column + "_df2"]
        to_return = sample[return_cols]
        if for_display:
            to_return.columns = self.join_columns + [
                column + " (" + self.df1_name + ")",
                column + " (" + self.df2_name + ")",
            ]
        return to_return

    def sample_unique_rows(self, df1_or_df2, sample_count=10):
        """Returns a sample sub-dataframe of rows that are only in one dataframe.  This is replaced
        in the Spark sub-class.

        Parameters
        ----------
        df1_or_df2 : {'df1', 'df2'}
            Which dataframe you're picking from
        sample_count : int
            How many samples to take

        Returns
        -------
        pd.DataFrame
            A sample of unique rows from the dataframe you specified.  The columns are trimmed at
            10 columns to make them more print-friendly
        """
        df_unq_rows = getattr(self, df1_or_df2 + "_unq_rows")
        return df_unq_rows.sample(min(sample_count, df_unq_rows.shape[0]))[df_unq_rows.columns[:10]]

    def cnt_df_unq_rows(self, df1_or_df2):
        return getattr(self, df1_or_df2 + "_unq_rows").shape[0]

    def _df_to_string(self, dataframe):
        """Function to return a string representation of a dataframe.  Changes between Pandas and
        Spark."""
        return dataframe.to_string()

    def _pre_report(self):
        """Pre-processing for the report step - nothing for Pandas"""
        pass

    def report(self, sample_count=10, file=sys.stdout):
        """Creates a string representation of a report, and prints it to stdout (or a file).  This
        method just gathers a bunch of other methods together into one report so that Pandas and
        Spark can implement each one separately.

        Parameters
        ----------
        sample_count : int, optional
            The number of sample records to print.  Defaults to 10.
        file : ``file``, optional
            A filehandle to write the report to. By default, this is sys.stdout, printing the report
            to stdout. You can also redirect this to an output file, as in the example.

        Examples
        --------
        >>> with open('my_report.txt', 'w') as report_file:
        ...     comparison.report(file=report_file)
        """
        self._pre_report()  # OK for Spark I think
        self._report_header(file)  # OK for Spark I think
        self._report_column_summary(file)  # OK for Spark I think
        self._report_row_summary(file)
        self._report_column_comparison(file)  # OK for Spark I think
        self._report_column_comparison_samples(sample_count, file)  # OK for Spark I think
        self._report_sample_rows("df1", sample_count, file)
        self._report_sample_rows("df2", sample_count, file)

    def _report_header(self, target):
        """Prints the report header, which is largely summary stats"""
        print(utils.render("header.txt"), file=target)
        df_header = pd.DataFrame(
            {
                "DataFrame": [self.df1_name, self.df2_name],
                "Columns": [self.df_column_cnt("df1"), self.df_column_cnt("df2")],
                "Rows": [self.df_row_count("df1"), self.df_row_count("df2")],
            }
        )
        print(df_header[["DataFrame", "Columns", "Rows"]].to_string() + "\n", file=target)

    def _report_column_summary(self, target):
        """Prints the column summary"""
        print(
            utils.render(
                "column_summary.txt",
                cnt_intersect_columns=len(self.intersect_columns),
                cnt_df1_unq_columns=len(self.df1_unq_columns),
                cnt_df2_unq_columns=len(self.df2_unq_columns),
                df1_name=self.df1_name,
                df2_name=self.df2_name,
            )
            + "\n",
            file=target,
        )

    def _report_row_summary(self, target):
        """Prints the row summary to the report"""
        print(
            utils.render(
                "row_summary.txt",
                match_on="index" if self.on_index else ", ".join(self.join_columns),
                abs_tol=self.abs_tol,
                rel_tol=self.rel_tol,
                cnt_intersect_rows=self.intersect_rows_count,
                cnt_df1_unq_rows=self.cnt_df_unq_rows("df1"),
                cnt_df2_unq_rows=self.cnt_df_unq_rows("df2"),
                cnt_unequal_rows=self.intersect_rows_count - self.matching_rows_count,
                cnt_matching_rows=self.matching_rows_count,
                df1_name=self.df1_name,
                df2_name=self.df2_name,
                any_dupes="Yes" if self._any_dupes else "No",
            )
            + "\n",
            file=target,
        )

    def _report_column_comparison(self, target):
        """High-level column comparison, printed to target."""
        print(
            utils.render(
                "column_comparison.txt",
                col_cnt_uneq_vals=(self.column_stats.unequal_cnt > 0).sum(),
                col_cnt_eq_vals=(self.column_stats.unequal_cnt == 0).sum(),
                col_cnt_known_eq_vals=(self.column_stats.known_diff_cnt > 0).sum(),
                cnt_uneq_vals=self.column_stats.unequal_cnt.sum(),
            )
            + "\n",
            file=target,
        )

    def _report_column_comparison_samples(self, sample_count, target):
        """Column comparison which prints some samples

        Parameters
        ----------
        sample_count : int
            The count of samples to print out
        target : file
            The file object to print out to
        """
        fields_to_print = {
            "column": "Column",
            "dtype1": "{} dtype".format(self.df1_name),
            "dtype2": "{} dtype".format(self.df2_name),
            "unequal_cnt": "# Unequal",
            "max_diff": "Max Diff",
            "null_diff": "# Null Diff",
        }
        match_stats = (
            self.column_stats[~self.column_stats.all_match][fields_to_print.keys()]
            .rename(fields_to_print, axis=1)
            .sort_values("Column")
        )
        match_sample = [
            self.sample_mismatch(col, sample_count, for_display=True)
            for col in self.column_stats[~self.column_stats.all_match].column
        ]

        if not self.column_stats.all_match.all():  # Only print if there are non-matchers
            print(utils.render("unequal_columns.txt", match_stats=match_stats), file=target)
            print(
                utils.render(
                    "unequal_rows.txt",
                    match_sample="\n\n".join(self._df_to_string(sample) for sample in match_sample),
                )
                + "\n",
                file=target,
            )

    def _report_sample_rows(self, df1_or_df2, sample_count, target):
        """Prints a sample of rows in one of the dataframes that aren't in the other to the report

        Parameters
        ----------
        df1_or_df2 : {'df1', 'df2'}
            Which dataframe you're looking at
        sample_count : int
            The count of samples to print out
        target : file
            The file to print out to
        """
        df_name = getattr(self, df1_or_df2 + "_name")
        if self.cnt_df_unq_rows(df1_or_df2) > 0:
            print(
                utils.render(
                    "unique_rows.txt",
                    df_name=df_name,
                    df_name_dashes="-" * len(df_name),
                    sample_rows=self._df_to_string(
                        self.sample_unique_rows(df1_or_df2, sample_count)
                    ),
                )
                + "\n",
                file=target,
            )
