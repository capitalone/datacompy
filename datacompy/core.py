#
# Copyright 2020 Capital One Services, LLC
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


import numpy as np
import pandas as pd

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
        Flag to strip whitespace (including newlines) from string columns (including any join
        columns)
    ignore_case : bool, optional
        Flag to ignore the case of string columns
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
        cast_column_names_lower=True,
    ):

        self.cast_column_names_lower = cast_column_names_lower
        if on_index and join_columns is not None:
            raise Exception("Only provide on_index or join_columns")
        elif on_index:
            self.on_index = True
            self.join_columns = []
        elif isinstance(join_columns, (str, int, float)):
            self.join_columns = [
                str(join_columns).lower() if self.cast_column_names_lower else str(join_columns)
            ]
            self.on_index = False
        else:
            self.join_columns = [
                str(col).lower() if self.cast_column_names_lower else str(col)
                for col in join_columns
            ]
            self.on_index = False

        self._any_dupes = False
        self.df1 = df1
        self.df2 = df2
        self.df1_name = df1_name
        self.df2_name = df2_name
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.df1_unq_rows = self.df2_unq_rows = self.intersect_rows = None
        self.column_stats = []
        self._compare(ignore_spaces, ignore_case)

    @property
    def df1(self):
        return self._df1

    @df1.setter
    def df1(self, df1):
        """Check that it is a dataframe and has the join columns"""
        self._df1 = df1
        self._validate_dataframe("df1", cast_column_names_lower=self.cast_column_names_lower)

    @property
    def df2(self):
        return self._df2

    @df2.setter
    def df2(self, df2):
        """Check that it is a dataframe and has the join columns"""
        self._df2 = df2
        self._validate_dataframe("df2", cast_column_names_lower=self.cast_column_names_lower)

    def _validate_dataframe(self, index, cast_column_names_lower=True):
        """Check that it is a dataframe and has the join columns

        Parameters
        ----------
        index : str
            The "index" of the dataframe - df1 or df2.
        cast_column_names_lower: bool, optional
            Boolean indicator that controls of column names will be cast into lower case
        """
        dataframe = getattr(self, index)
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("{} must be a pandas DataFrame".format(index))

        if cast_column_names_lower:
            dataframe.columns = [str(col).lower() for col in dataframe.columns]
        else:
            dataframe.columns = [str(col) for col in dataframe.columns]
        # Check if join_columns are present in the dataframe
        if not set(self.join_columns).issubset(set(dataframe.columns)):
            raise ValueError("{} must have all columns from join_columns".format(index))

        if len(set(dataframe.columns)) < len(dataframe.columns):
            raise ValueError("{} must have unique column names".format(index))

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
        LOG.info("Number of columns in common: {}".format(len(self.intersect_columns())))
        LOG.debug("Checking column overlap")
        for col in self.df1_unq_columns():
            LOG.info("Column in df1 and not in df2: {}".format(col))
        LOG.info("Number of columns in df1 and not in df2: {}".format(len(self.df1_unq_columns())))
        for col in self.df2_unq_columns():
            LOG.info("Column in df2 and not in df1: {}".format(col))
        LOG.info("Number of columns in df2 and not in df1: {}".format(len(self.df2_unq_columns())))
        LOG.debug("Merging dataframes")
        self._dataframe_merge(ignore_spaces)
        self._intersect_compare(ignore_spaces, ignore_case)
        if self.matches():
            LOG.info("df1 matches df2")
        else:
            LOG.info("df1 does not match df2")

    def df1_unq_columns(self):
        """Get columns that are unique to df1"""
        return set(self.df1.columns) - set(self.df2.columns)

    def df2_unq_columns(self):
        """Get columns that are unique to df2"""
        return set(self.df2.columns) - set(self.df1.columns)

    def intersect_columns(self):
        """Get columns that are shared between the two dataframes"""
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
                index_column = temp_column_name(self.df1, self.df2)
                self.df1[index_column] = self.df1.index
                self.df2[index_column] = self.df2.index
                temp_join_columns = [index_column]
            else:
                temp_join_columns = list(self.join_columns)

            # Create order column for uniqueness of match
            order_column = temp_column_name(self.df1, self.df2)
            self.df1[order_column] = generate_id_within_group(self.df1, temp_join_columns)
            self.df2[order_column] = generate_id_within_group(self.df2, temp_join_columns)
            temp_join_columns.append(order_column)

            params = {"on": temp_join_columns}
        elif self.on_index:
            params = {"left_index": True, "right_index": True}
        else:
            params = {"on": self.join_columns}

        if ignore_spaces:
            for column in self.join_columns:
                if self.df1[column].dtype.kind == "O":
                    self.df1[column] = self.df1[column].str.strip()
                if self.df2[column].dtype.kind == "O":
                    self.df2[column] = self.df2[column].str.strip()

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

        df1_cols = get_merged_columns(self.df1, outer_join, "_df1")
        df2_cols = get_merged_columns(self.df2, outer_join, "_df2")

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
                len(self.intersect_rows)
            )
        )

    def _intersect_compare(self, ignore_spaces, ignore_case):
        """Run the comparison on the intersect dataframe

        This loops through all columns that are shared between df1 and df2, and
        creates a column column_match which is True for matches, False
        otherwise.
        """
        LOG.debug("Comparing intersection")
        row_cnt = len(self.intersect_rows)
        for column in self.intersect_columns():
            if column in self.join_columns:
                match_cnt = row_cnt
                col_match = ""
                max_diff = 0
                null_diff = 0
            else:
                col_1 = column + "_df1"
                col_2 = column + "_df2"
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
                null_diff = (
                    (self.intersect_rows[col_1].isnull()) ^ (self.intersect_rows[col_2].isnull())
                ).sum()

            if row_cnt > 0:
                match_rate = float(match_cnt) / row_cnt
            else:
                match_rate = 0
            LOG.info("{}: {} / {} ({:.2%}) match".format(column, match_cnt, row_cnt, match_rate))

            self.column_stats.append(
                {
                    "column": column,
                    "match_column": col_match,
                    "match_cnt": match_cnt,
                    "unequal_cnt": row_cnt - match_cnt,
                    "dtype1": str(self.df1[column].dtype),
                    "dtype2": str(self.df2[column].dtype),
                    "all_match": all(
                        (self.df1[column].dtype == self.df2[column].dtype, row_cnt == match_cnt)
                    ),
                    "max_diff": max_diff,
                    "null_diff": null_diff,
                }
            )

    def all_columns_match(self):
        """Whether the columns all match in the dataframes"""
        return self.df1_unq_columns() == self.df2_unq_columns() == set()

    def all_rows_overlap(self):
        """Whether the rows are all present in both dataframes

        Returns
        -------
        bool
            True if all rows in df1 are in df2 and vice versa (based on
            existence for join option)
        """
        return len(self.df1_unq_rows) == len(self.df2_unq_rows) == 0

    def count_matching_rows(self):
        """Count the number of rows match (on overlapping fields)

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

    def intersect_rows_match(self):
        """Check whether the intersect rows all match"""
        actual_length = self.intersect_rows.shape[0]
        return self.count_matching_rows() == actual_length

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
        if not self.df2_unq_columns() == set():
            return False
        elif not len(self.df2_unq_rows) == 0:
            return False
        elif not self.intersect_rows_match():
            return False
        else:
            return True

    def sample_mismatch(self, column, sample_count=10, for_display=False):
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
        Pandas.DataFrame
            A sample of the intersection dataframe, containing only the
            "pertinent" columns, for rows that don't match on the provided
            column.
        """
        row_cnt = self.intersect_rows.shape[0]
        col_match = self.intersect_rows[column + "_match"]
        match_cnt = col_match.sum()
        sample_count = min(sample_count, row_cnt - match_cnt)
        sample = self.intersect_rows[~col_match].sample(sample_count)
        return_cols = self.join_columns + [column + "_df1", column + "_df2"]
        to_return = sample[return_cols]
        if for_display:
            to_return.columns = self.join_columns + [
                column + " (" + self.df1_name + ")",
                column + " (" + self.df2_name + ")",
            ]
        return to_return

    def all_mismatch(self):
        """All rows with any columns that have a mismatch. Returns all df1 and df2 versions of the columns and join
        columns.

        Returns
        -------
        Pandas.DataFrame
            All rows of the intersection dataframe, containing any columns, that don't match.
        """
        match_list = []
        return_list = []
        for col in self.intersect_rows.columns:
            if col.endswith("_match"):
                match_list.append(col)
                return_list.extend([col[:-6] + "_df1", col[:-6] + "_df2"])

        mm_bool = self.intersect_rows[match_list].all(axis="columns")
        return self.intersect_rows[~mm_bool][self.join_columns + return_list]

    def report(self, sample_count=10):
        """Returns a string representation of a report.  The representation can
        then be printed or saved to a file.

        Parameters
        ----------
        sample_count : int, optional
            The number of sample records to return.  Defaults to 10.

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
        if self.on_index:
            match_on = "index"
        else:
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
        cnt_intersect = self.intersect_rows.shape[0]
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
                        "{} dtype".format(self.df1_name): column["dtype1"],
                        "{} dtype".format(self.df2_name): column["dtype2"],
                        "# Unequal": column["unequal_cnt"],
                        "Max Diff": column["max_diff"],
                        "# Null Diff": column["null_diff"],
                    }
                )
                if column["unequal_cnt"] > 0:
                    match_sample.append(
                        self.sample_mismatch(column["column"], sample_count, for_display=True)
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
                    "{} dtype".format(self.df1_name),
                    "{} dtype".format(self.df2_name),
                    "# Unequal",
                    "Max Diff",
                    "# Null Diff",
                ]
            ].to_string()
            report += "\n\n"

            report += "Sample Rows with Unequal Values\n"
            report += "-------------------------------\n"
            report += "\n"
            for sample in match_sample:
                report += sample.to_string()
                report += "\n\n"

        if self.df1_unq_rows.shape[0] > 0:
            report += "Sample Rows Only in {} (First 10 Columns)\n".format(self.df1_name)
            report += "---------------------------------------{}\n".format("-" * len(self.df1_name))
            report += "\n"
            columns = self.df1_unq_rows.columns[:10]
            unq_count = min(sample_count, self.df1_unq_rows.shape[0])
            report += self.df1_unq_rows.sample(unq_count)[columns].to_string()
            report += "\n\n"

        if self.df2_unq_rows.shape[0] > 0:
            report += "Sample Rows Only in {} (First 10 Columns)\n".format(self.df2_name)
            report += "---------------------------------------{}\n".format("-" * len(self.df2_name))
            report += "\n"
            columns = self.df2_unq_rows.columns[:10]
            unq_count = min(sample_count, self.df2_unq_rows.shape[0])
            report += self.df2_unq_rows.sample(unq_count)[columns].to_string()
            report += "\n\n"

        return report


def render(filename, *fields):
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
    with open(os.path.join(this_dir, "templates", filename)) as file_open:
        return file_open.read().format(*fields)


def columns_equal(col_1, col_2, rel_tol=0, abs_tol=0, ignore_spaces=False, ignore_case=False):
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
    try:
        compare = pd.Series(np.isclose(col_1, col_2, rtol=rel_tol, atol=abs_tol, equal_nan=True))
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
        except (ValueError, TypeError):
            try:
                if ignore_spaces:
                    if col_1.dtype.kind == "O":
                        col_1 = col_1.str.strip()
                    if col_2.dtype.kind == "O":
                        col_2 = col_2.str.strip()

                if ignore_case:
                    if col_1.dtype.kind == "O":
                        col_1 = col_1.str.upper()
                    if col_2.dtype.kind == "O":
                        col_2 = col_2.str.upper()

                if {col_1.dtype.kind, col_2.dtype.kind} == {"M", "O"}:
                    compare = compare_string_and_date_columns(col_1, col_2)
                else:
                    compare = pd.Series((col_1 == col_2) | (col_1.isnull() & col_2.isnull()))
            except:
                # Blanket exception should just return all False
                compare = pd.Series(False, index=col_1.index)
    compare.index = col_1.index
    return compare


def compare_string_and_date_columns(col_1, col_2):
    """Compare a string column and date column, value-wise.  This tries to
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
    except:
        return pd.Series(False, index=col_1.index)


def get_merged_columns(original_df, merged_df, suffix):
    """Gets the columns from an original dataframe, in the new merged dataframe

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
        elif col + suffix in merged_df.columns:
            columns.append(col + suffix)
        else:
            raise ValueError("Column not found: %s", col)
    return columns


def temp_column_name(*dataframes):
    """Gets a temp column name that isn't included in columns of any dataframes

    Parameters
    ----------
    dataframes : list of Pandas.DataFrame
        The DataFrames to create a temporary column name for

    Returns
    -------
    str
        String column name that looks like '_temp_x' for some integer x
    """
    i = 0
    while True:
        temp_column = "_temp_{}".format(i)
        unique = True
        for dataframe in dataframes:
            if temp_column in dataframe.columns:
                i += 1
                unique = False
        if unique:
            return temp_column


def calculate_max_diff(col_1, col_2):
    """Get a maximum difference between two columns

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
        return (col_1.astype(float) - col_2.astype(float)).abs().max()
    except:
        return 0


def generate_id_within_group(dataframe, join_columns):
    """Generate an ID column that can be used to deduplicate identical rows.  The series generated
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
            raise ValueError("{} was found in your join columns".format(default_value))
        return (
            dataframe[join_columns]
            .astype(str)
            .fillna(default_value)
            .groupby(join_columns)
            .cumcount()
        )
    else:
        return dataframe[join_columns].groupby(join_columns).cumcount()
