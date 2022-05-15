#
# Copyright 2022 Capital One Services, LLC
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
Compare two Spark DataFrames

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""

import logging
import os
from enum import Enum

import pandas as pd
import pyspark.sql
from ordered_set import OrderedSet
from pyspark.sql.functions import (
    abs,
    array,
    array_contains,
    coalesce,
    col,
    isnan,
    isnull,
    lit,
    trim,
    upper,
    when,
)

from datacompy.base import BaseCompare

LOG = logging.getLogger(__name__)


try:
    from pyspark.sql import functions as F
except ImportError:
    pass  # Let non-Spark people at least enjoy the loveliness of the pandas datacompy functionality


class MatchType(Enum):
    MISMATCH, MATCH, KNOWN_DIFFERENCE = range(3)


# Used for checking equality with decimal(X, Y) types. Otherwise treated as the string "decimal".
def decimal_comparator():
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


def _is_comparable(type1, type2):
    """Checks if two Spark data types can be safely compared.

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
    return type1 == type2 or (
        type1 in NUMERIC_SPARK_TYPES and type2 in NUMERIC_SPARK_TYPES
    )


class SparkCompare(BaseCompare):
    """Comparison class to be used to compare whether two Spark dataframes as equal.

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
    join_columns : list or str
        Column(s) to join dataframes on.  If a string is passed in, that one
        column will be used.
    abs_tol : float, optional
        Absolute tolerance between two values.
    rel_tol : float, optional
        Relative tolerance between two values.
    df1_name : str, optional
        A string name for the first dataframe. This allows the reporting to
        print out an actual name instead of "df1", and allows human users to
        more easily track the dataframes.
    df2_name : str, optional
        A string name for the second dataframe. This allows the reporting to
        print out an actual name instead of "df2", and allows human users to
        more easily track the dataframes.
    ignore_spaces : bool, optional
        Flag to strip whitespace (including newlines) from string columns (including any join
        columns)
    ignore_case : bool, optional
        Flag to ignore the case of string columns

    Attributes
    ----------
    spark_session : pyspark.sql.SparkSession
        A ``SparkSession`` to be used to execute Spark commands in the comparison.
    df1_unq_rows : pyspark.sql.DataFrame
        All records that are only in df1 (based on a join on join_columns)
    df2_unq_rows : pyspark.sql.DataFrame
        All records that are only in df2 (based on a join on join_columns)
    """

    def __init__(
        self,
        spark_session,
        df1,
        df2,
        join_columns=None,
        abs_tol=0,
        rel_tol=0,
        df1_name="df1",
        df2_name="df2",
        ignore_spaces=False,
        ignore_case=False,
    ):

        if not isinstance(df1, pyspark.sql.DataFrame):
            raise TypeError("df1 must be a Spark DataFrame")
        if not isinstance(df2, pyspark.sql.DataFrame):
            raise TypeError("df2 must be a Spark DataFrame")

        self.spark = spark_session
        self._any_dupes = False

        if join_columns is None:
            raise Exception("Please provide join_columns")
        elif isinstance(join_columns, (str, int, float)):
            self.join_columns = [str(join_columns).lower()]
        else:
            self.join_columns = [str(col).lower() for col in join_columns]

        self.df1 = df1.toDF(*[str(c).lower() for c in df1.columns])
        self.df2 = df2.toDF(*[str(c).lower() for c in df2.columns])

        self.df1_name = df1_name
        self.df2_name = df2_name
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.df1_unq_rows = self.df2_unq_rows = self.intersect_rows = None
        self.column_stats = []
        self._compare(ignore_spaces, ignore_case)

    @BaseCompare.df1.setter
    def df1(self, df1):
        """Check that it is a dataframe and has the join columns"""
        self._df1 = df1
        self._validate_dataframe("df1")

    @BaseCompare.df2.setter
    def df2(self, df2):
        """Check that it is a dataframe and has the join columns"""
        self._df2 = df2
        self._validate_dataframe("df2")

    def _validate_dataframe(self, index):
        """Check that a dataframe has the join columns

        Parameters
        ----------
        index : str
            The "index" of the dataframe - df1 or df2.

        Returns
        -------
        None
        """
        dataframe = getattr(self, index)

        # Check if join_columns are present in the dataframe
        if not set(self.join_columns).issubset(set(dataframe.columns)):
            raise ValueError("{} must have all columns from join_columns".format(index))

        if len(set(dataframe.columns)) < len(dataframe.columns):
            raise ValueError("{} must have unique column names".format(index))

        if (
            dataframe.drop_duplicates(subset=self.join_columns).count()
            < dataframe.count()
        ):
            self._any_dupes = True

    def _compare(self, ignore_spaces, ignore_case):
        """Actually run the comparison.

        This method will log out information about what is different between
        the two dataframes, and will also return a boolean.

        Unlike Pandas which tries to run df1.equals(df2) first so that if they're truly equal we
        can tell. Spark does not have the same functionality.
        """
        LOG.warning(
            "Number of columns in common: {}".format(len(self.intersect_columns()))
        )
        LOG.debug("Checking column overlap")
        for col in self.df1_unq_columns():
            LOG.warning("Column in df1 and not in df2: {}".format(col))
        LOG.warning(
            "Number of columns in df1 and not in df2: {}".format(
                len(self.df1_unq_columns())
            )
        )
        for col in self.df2_unq_columns():
            LOG.warning("Column in df2 and not in df1: {}".format(col))
        LOG.warning(
            "Number of columns in df2 and not in df1: {}".format(
                len(self.df2_unq_columns())
            )
        )

        if self._any_dupes:
            LOG.warning("Dropping duplicates")
            self.df1 = self.df1.dropDuplicates(self.join_columns)
            self.df2 = self.df2.dropDuplicates(self.join_columns)

        LOG.debug("Merging dataframes")
        self._dataframe_merge(ignore_spaces)
        self._intersect_compare(ignore_spaces, ignore_case)
        if self.matches():
            LOG.warning("df1 matches df2")
        else:
            LOG.warning("df1 does not match df2")

    def df1_unq_columns(self):
        """Get columns that are unique to df1"""
        return OrderedSet(self.df1.columns) - OrderedSet(self.df2.columns)

    def df2_unq_columns(self):
        """Get columns that are unique to df2"""
        return OrderedSet(self.df2.columns) - OrderedSet(self.df1.columns)

    def intersect_columns(self):
        """Get columns that are shared between the two dataframes"""
        return OrderedSet(self.df1.columns) & OrderedSet(self.df2.columns)

    def non_join_columns(self):
        """Get columns that are shared between the two dataframes excluding join_columns"""
        return (
            OrderedSet(self.df1.columns) | OrderedSet(self.df2.columns)
        ) - OrderedSet(self.join_columns)

    def _generate_merge_indicator(self):
        self.df1_renamed = self.df1_renamed.join(
            self.df1_renamed.select(self.join_columns)
            .join(
                self.df2_renamed.select(self.join_columns),
                on=self.join_columns,
                how="left_anti",
            )
            .withColumn("_merge_left", lit("left_only")),
            on=self.join_columns,
            how="left",
        ).fillna("both", "_merge_left")

        self.df2_renamed = self.df2_renamed.join(
            self.df2_renamed.select(self.join_columns)
            .join(
                self.df1_renamed.select(self.join_columns),
                on=self.join_columns,
                how="left_anti",
            )
            .withColumn("_merge_right", lit("right_only")),
            on=self.join_columns,
            how="left",
        ).fillna("both", "_merge_right")

    def _dataframe_merge(self, ignore_spaces):
        """Merge df1 to df2 on the join columns, to get df1 - df2, df2 - df1
        and df1 & df2
        """
        LOG.debug("Outer joining")
        params = {"on": self.join_columns}

        if ignore_spaces:
            for column in self.join_columns:
                df1_dtype, _ = get_column_dtypes(self.df1, column, column)
                df2_dtype, _ = get_column_dtypes(self.df2, column, column)
                if df1_dtype == "string":
                    self.df1 = self.df1.withColumn(column, trim(col(column)))
                if df2_dtype == "string":
                    self.df2 = self.df2.withColumn(column, trim(col(column)))

        # add suffixes using non_join_columns
        self.df1_renamed = self.df1
        self.df2_renamed = self.df2

        for c in self.non_join_columns():
            self.df1_renamed = self.df1_renamed.withColumnRenamed(
                c, "{}{}".format(c, "_df1")
            )
            self.df2_renamed = self.df2_renamed.withColumnRenamed(
                c, "{}{}".format(c, "_df2")
            )

        self._generate_merge_indicator()

        # Null safe logic for the outer join
        ############
        conditions = []
        for c in self.join_columns:
            self.df1_renamed = self.df1_renamed.withColumnRenamed(
                c, "{}{}".format(c, "_df1")
            )
            self.df2_renamed = self.df2_renamed.withColumnRenamed(
                c, "{}{}".format(c, "_df2")
            )
            conditions.append(
                self.df1_renamed[c + "_df1"].eqNullSafe(self.df2_renamed[c + "_df2"])
            )
        outer_join = self.df1_renamed.join(self.df2_renamed, conditions, how="outer")

        # merge indicator corner case with nulls
        outer_join = outer_join.withColumn(
            "_merge_left",
            when(
                (outer_join["_merge_left"] == "both")
                & (isnull(outer_join["_merge_right"])),
                "left_only",
            ).otherwise(outer_join["_merge_left"]),
        )
        outer_join = outer_join.withColumn(
            "_merge_right",
            when(
                (outer_join["_merge_right"] == "both")
                & (isnull(outer_join["_merge_left"])),
                "right_only",
            ).otherwise(outer_join["_merge_right"]),
        )

        # cleanup _merge_left and _merge_right into _merge
        outer_join = outer_join.withColumn(
            "_merge", coalesce(outer_join["_merge_left"], outer_join["_merge_right"])
        )

        # collapse join_columns back into one set of columns from _df1 / _df2
        for c in self.join_columns:
            outer_join = outer_join.withColumn(
                c,
                when(
                    outer_join["_merge"].isin(["left_only", "both"]),
                    outer_join[c + "_df1"],
                ).otherwise(outer_join[c + "_df2"]),
            )
            # outer_join = outer_join.withColumn(col, when(outer_join["_merge"] == "right_only", ))
            outer_join = outer_join.drop(c + "_df1", c + "_df2")

        df1_cols = get_merged_columns(self.df1.drop("_merge_left"), outer_join, "_df1")
        df2_cols = get_merged_columns(self.df2.drop("_merge_right"), outer_join, "_df2")

        LOG.debug("Selecting df1 unique rows")
        self.df1_unq_rows = outer_join.select(df1_cols).where("_merge = 'left_only'")

        LOG.debug("Selecting df2 unique rows")
        self.df2_unq_rows = outer_join.select(df2_cols).where("_merge = 'right_only'")
        LOG.warning(
            "Number of rows in df1 and not in df2: {}".format(self.df1_unq_rows.count())
        )
        LOG.warning(
            "Number of rows in df2 and not in df1: {}".format(self.df2_unq_rows.count())
        )

        LOG.debug("Selecting intersecting rows")
        self.intersect_rows = outer_join.where("_merge = 'both'")
        LOG.warning(
            "Number of rows in df1 and df2 (not necessarily equal): {}".format(
                self.intersect_rows.count()
            )
        )

    def _intersect_compare(self, ignore_spaces, ignore_case):
        """Run the comparison on the intersect dataframe

        This loops through all columns that are shared between df1 and df2, and
        creates a column column_match which is True for matches, False
        otherwise.
        """
        LOG.debug("Comparing intersection")
        row_cnt = self.intersect_rows.count()
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

                self.intersect_rows = columns_equal(
                    self.intersect_rows,
                    col_1,
                    col_2,
                    col_match,
                    self.rel_tol,
                    self.abs_tol,
                    ignore_spaces,
                    ignore_case,
                )
                match_cnt = (
                    self.intersect_rows.select(col_match)
                    .where(col(col_match) == True)
                    .count()
                )
                max_diff = calculate_max_diff(self.intersect_rows, col_1, col_2)
                null_diff = calculate_null_diff(self.intersect_rows, col_1, col_2)

            if row_cnt > 0:
                match_rate = float(match_cnt) / row_cnt
            else:
                match_rate = 0
            LOG.warning(
                "{}: {} / {} ({:.2%}) match".format(
                    column, match_cnt, row_cnt, match_rate
                )
            )

            col1_dtype, _ = get_column_dtypes(self.df1, column, column)
            col2_dtype, _ = get_column_dtypes(self.df2, column, column)

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
        return self.df1_unq_rows.count() == self.df2_unq_rows.count() == 0

    def count_matching_rows(self):
        """Count the number of rows match (on overlapping fields)

        Returns
        -------
        int
            Number of matching rows
        """
        conditions = []
        for column in self.intersect_columns():
            if column not in self.join_columns:
                match_columns = column + "_match"
                conditions.append("{} == True".format(match_columns))
        if len(conditions) > 0:
            match_columns_count = self.intersect_rows.filter(
                " and ".join(conditions)
            ).count()
        else:
            match_columns_count = 0
        return match_columns_count

    def intersect_rows_match(self):
        """Check whether the intersect rows all match"""
        actual_length = self.intersect_rows.count()
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
        elif not self.df2_unq_rows.count() == 0:
            return False
        elif not self.intersect_rows_match():
            return False
        else:
            return True

    def sample_mismatch(self, column, sample_count=10):
        """Returns a sample sub-dataframe which contains the identifying
        columns, and df1 and df2 versions of the column.

        Parameters
        ----------
        column : str
            The raw column name (i.e. without ``_df1`` appended)
        sample_count : int, optional
            The number of sample records to return.  Defaults to 10.

        Returns
        -------
        pyspark.sql.DataFrame
            A sample of the intersection dataframe, containing only the
            "pertinent" columns, for rows that don't match on the provided
            column.
        """
        row_cnt = self.intersect_rows.count()
        col_match = self.intersect_rows.select(column + "_match")
        match_cnt = col_match.where(col(column + "_match") == True).count()
        sample_count = min(sample_count, row_cnt - match_cnt)
        sample = (
            self.intersect_rows.where(col(column + "_match") == False)
            .drop(column + "_match")
            .limit(sample_count)
        )
        return_cols = self.join_columns + [column + "_df1", column + "_df2"]
        to_return = sample.select(return_cols)
        return to_return

    def all_mismatch(self):
        """All rows with any columns that have a mismatch. Returns all df1 and df2 versions of the
        columns and join columns.

        Returns
        -------
        pyspark.sql.DataFrame
            All rows of the intersection dataframe, containing any columns, that don't match.
        """
        match_list = []
        return_list = []
        for col in self.intersect_rows.columns:
            if col.endswith("_match"):
                match_list.append(col)
                return_list.extend([col[:-6] + "_df1", col[:-6] + "_df2"])

        mm_rows = self.intersect_rows.withColumn(
            "match_array", array(match_list)
        ).where(array_contains("match_array", False))
        return mm_rows.select(self.join_columns + return_list)

    def report(self, sample_count=10, column_count=10):
        """Returns a string representation of a report.  The representation can
        then be printed or saved to a file.

        Parameters
        ----------
        sample_count : int, optional
            The number of sample records to return.  Defaults to 10.

        column_count : int, optional
            The number of columns to display in the sample records output.  Defaults to 10.

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
        cnt_intersect = self.intersect_rows.count()
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
                        self.sample_mismatch(column["column"], sample_count)
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

            if sample_count > 0:
                report += "Sample Rows with Unequal Values\n"
                report += "-------------------------------\n"
                report += "\n"
                for sample in match_sample:
                    report += sample._jdf.showString(20, 20, False)
                    report += "\n\n"

        if min(sample_count, self.df1_unq_rows.count()) > 0:
            report += "Sample Rows Only in {} (First {} Columns)\n".format(
                self.df1_name, column_count
            )
            report += "---------------------------------------{}\n".format(
                "-" * len(self.df1_name)
            )
            report += "\n"
            columns = self.df1_unq_rows.columns[:column_count]
            unq_count = min(sample_count, self.df1_unq_rows.count())
            report += (
                self.df1_unq_rows.limit(unq_count)
                .select(columns)
                ._jdf.showString(20, 20, False)
            )
            report += "\n\n"

        if min(sample_count, self.df2_unq_rows.count()) > 0:
            report += "Sample Rows Only in {} (First {} Columns)\n".format(
                self.df2_name, column_count
            )
            report += "---------------------------------------{}\n".format(
                "-" * len(self.df2_name)
            )
            report += "\n"
            columns = self.df2_unq_rows.columns[:column_count]
            unq_count = min(sample_count, self.df2_unq_rows.count())
            report += (
                self.df2_unq_rows.limit(unq_count)
                .select(columns)
                ._jdf.showString(20, 20, False)
            )
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


def columns_equal(
    dataframe,
    col_1,
    col_2,
    col_match,
    rel_tol=0,
    abs_tol=0,
    ignore_spaces=False,
    ignore_case=False,
):
    """Compares two columns from a dataframe, returning the DataFrame with a True/False column.

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
    col_match : str
        The matching column denoting if the compare was a match or not
    rel_tol : float, optional
        Relative tolerance
    abs_tol : float, optional
        Absolute tolerance
    ignore_spaces : bool, optional
        Flag to strip whitespace (including newlines) from string columns
    ignore_case : bool, optional
        Flag to ignore the case of string columns

    Notes
    -----
    For finite values, ``isclose`` uses the following equation to test whether two floating point
    values are equivalent. To align with Pandas we will use the same for Spark.

    absolute(a - b) <= (atol + rtol * absolute(b))

    https://numpy.org/doc/stable/reference/generated/numpy.isclose.html#numpy-isclose

    Returns
    -------
    pyspark.sql.DataFrame
        A column of boolean values are added.  True == the values match, False == the
        values don't match.
    """
    base_dtype, compare_dtype = get_column_dtypes(dataframe, col_1, col_2)
    if _is_comparable(base_dtype, compare_dtype):
        if (base_dtype in NUMERIC_SPARK_TYPES) and (
            compare_dtype in NUMERIC_SPARK_TYPES
        ):  # numeric tolerance comparison
            dataframe = dataframe.withColumn(
                col_match,
                when(
                    (col(col_1).eqNullSafe(col(col_2)))
                    | (
                        abs(col(col_1) - col(col_2))
                        <= lit(abs_tol) + (lit(rel_tol) * abs(col_2))
                    ),
                    # corner case of col1 != NaN and col2 == Nan returns True incorrectly
                    when(
                        (isnan(col(col_1)) == False) & (isnan(col(col_2)) == True),
                        lit(False),
                    ).otherwise(lit(True)),
                ).otherwise(lit(False)),
            )
        else:  # non-numeric comparison
            if ignore_case and not ignore_spaces:
                when_clause = upper(col(col_1)).eqNullSafe(upper(col(col_2)))
            elif not ignore_case and ignore_spaces:
                when_clause = trim(col(col_1)).eqNullSafe(trim(col(col_2)))
            elif ignore_case and ignore_spaces:
                when_clause = upper(trim(col(col_1))).eqNullSafe(
                    upper(trim(col(col_2)))
                )
            else:
                when_clause = col(col_1).eqNullSafe(col(col_2))

            dataframe = dataframe.withColumn(
                col_match,
                when(when_clause, lit(True)).otherwise(lit(False)),
            )
    else:
        LOG.debug(
            "Skipping {}({}) and {}({}), columns are not comparable".format(
                col_1, base_dtype, col_2, compare_dtype
            )
        )
    return dataframe


def get_merged_columns(original_df, merged_df, suffix):
    """Gets the columns from an original dataframe, in the new merged dataframe

    Parameters
    ----------
    original_df : pyspark.sql.DataFrame
        The original, pre-merge dataframe
    merged_df : pyspark.sql.DataFrame
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
    dataframes : list of pyspark.sql.DataFrame
        The DataFrames to create a temporary column name for

    Returns
    -------
    str
        String column name that looks like '_temp_x' for some integer x
    """
    column_names = []
    for dataframe in dataframes:
        column_names.extend(dataframe.columns)
    i = 0
    while True:
        temp_column = "_temp_{}".format(i)
        unique = True
        for col in set(column_names):
            if temp_column in col:
                i += 1
                unique = False
        if unique:
            return temp_column


def calculate_max_diff(dataframe, col_1, col_2):
    """Get a maximum difference between two columns

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
    Numeric
        Numeric field, or zero if NaN, Null, or None.
    """
    diff = dataframe.select(
        (col(col_1).astype("float") - col(col_2).astype("float")).alias("diff")
    )
    abs_diff = diff.select(abs(col("diff")).alias("abs_diff"))
    max_diff = (
        abs_diff.where(isnan(col("abs_diff")) == False)
        .agg({"abs_diff": "max"})
        .collect()[0][0]
    )

    if pd.isna(max_diff) or pd.isnull(max_diff) or max_diff is None:
        return 0
    else:
        return max_diff


def calculate_null_diff(dataframe, col_1, col_2):
    """Get the null differences between two columns

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
    Numeric
        Numeric field, or zero if NaN, Null, or None.
    """
    nulls_df = dataframe.withColumn(
        "col_1_null", when(col(col_1).isNull() == True, lit(True)).otherwise(lit(False))
    )
    nulls_df = nulls_df.withColumn(
        "col_2_null", when(col(col_2).isNull() == True, lit(True)).otherwise(lit(False))
    ).select(["col_1_null", "col_2_null"])

    # (not a and b) or (a and not b)
    null_diff = nulls_df.where(
        ((col("col_1_null") == False) & (col("col_2_null") == True))
        | ((col("col_1_null") == True) & (col("col_2_null") == False))
    ).count()

    if pd.isna(null_diff) or pd.isnull(null_diff) or null_diff is None:
        return 0
    else:
        return null_diff


def get_column_dtypes(dataframe, col_1, col_2):
    """Get the dtypes of two columns

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
    base_dtype = [d[1] for d in dataframe.dtypes if d[0] == col_1][0]
    compare_dtype = [d[1] for d in dataframe.dtypes if d[0] == col_2][0]
    return base_dtype, compare_dtype
