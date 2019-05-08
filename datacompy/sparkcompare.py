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

from __future__ import print_function

import contextlib
import io
import sys
from enum import Enum

import pandas as pd

from datacompy import Compare

try:
    import pyspark
    from pyspark.sql import functions as F
except ImportError:
    pass  # Let non-Spark people at least enjoy the loveliness of the pandas datacompy functionality


class MatchType(Enum):
    MISMATCH, MATCH, KNOWN_DIFFERENCE, NULL_DIFFERENCE = range(4)


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

    return type1 == type2 or (type1 in NUMERIC_SPARK_TYPES and type2 in NUMERIC_SPARK_TYPES)


class SparkCompare(Compare):
    """Comparison class used to compare two Spark Dataframes.

    Extends the ``Compare`` functionality to the wide world of Spark and
    out-of-memory data.

    Parameters
    ----------
    spark_session : ``pyspark.sql.SparkSession``
        A ``SparkSession`` to be used to execute Spark commands in the
        comparison.
    df1 : ``pyspark.sql.DataFrame``
        The dataframe to serve as a basis for comparison. While you will
        ultimately get the same results comparing A to B as you will comparing
        B to A, by convention ``df1`` should be the canonical, gold
        standard reference dataframe in the comparison.
    df2 : ``pyspark.sql.DataFrame``
        The dataframe to be compared against ``df1``.
    join_columns : list
        A list of columns comprising the join key(s) of the two dataframes.
        If the column names are the same in the two dataframes, the names of
        the columns can be given as strings. If the names differ, the
        ``join_columns`` list should include tuples of the form
        (df1_column_name, df2_column_name).
    column_mapping : list[tuple], optional
        If columns to be compared have different names in the df1 and df2
        dataframes, a list should be provided in ``columns_mapping`` consisting
        of tuples of the form (df1_column_name, df2_column_name) for each
        set of differently-named columns to be compared against each other.
    cache_intermediates : bool, optional
        Whether or not ``SparkCompare`` will cache intermediate dataframes
        (such as the deduplicated version of dataframes, or the joined
        comparison). This will take a large amount of cache, proportional to
        the size of your dataframes, but will significantly speed up
        performance, as multiple steps will not have to recompute
        transformations. False by default.
    known_differences : list[dict], optional
        A list of dictionaries that define transformations to apply to the
        df2 dataframe to match values when there are known differences
        between df1 and df2. The dictionaries should contain:

            * name: A name that describes the transformation
            * types: The types that the transformation should be applied to.
                This prevents certain transformations from being applied to
                types that don't make sense and would cause exceptions.
            * transformation: A Spark SQL statement to apply to the column
                in the df2 dataset. The string "{input}" will be replaced
                by the variable in question.
    abs_tol : float, optional
        Absolute tolerance between two values.
    rel_tol : float, optional
        Relative tolerance between two values.
    show_all_columns : bool, optional
        If true, all columns will be shown in the report including columns
        with a 100% match rate.

    Returns
    -------
    SparkCompare
        Instance of a ``SparkCompare`` object, ready to do some comparin'.
        Note that if ``cache_intermediates=True``, this instance will already
        have done some work deduping the input dataframes. If
        ``cache_intermediates=False``, the instantiation of this object is lazy.
    """

    def __init__(
        self,
        spark_session,
        df1,
        df2,
        join_columns,
        column_mapping=None,
        cache_intermediates=False,
        known_differences=None,
        rel_tol=0,
        abs_tol=0,
        show_all_columns=False,
        df1_name="df1",
        df2_name="df2",
    ):
        self._any_dupes = False  # Later reset based on dedupin'
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.df1_name = df1_name
        self.df2_name = df2_name
        self.on_index = False
        if self.rel_tol < 0 or self.abs_tol < 0:
            raise ValueError("Please enter positive valued tolerances")
        self.show_all_columns = show_all_columns

        self._original_df1 = df1
        self._original_df2 = df2
        self.cache_intermediates = cache_intermediates

        self.join_column_tuples = self._tuplizer(join_columns)
        self.join_columns = [name[0] for name in self.join_column_tuples]

        self._known_differences = known_differences

        if column_mapping:
            for mapping in column_mapping:
                df2 = df2.withColumnRenamed(mapping[1], mapping[0])
            self.column_mapping = dict(column_mapping)
        else:
            self.column_mapping = {}

        for mapping in self.join_column_tuples:
            if mapping[1] != mapping[0]:
                df2 = df2.withColumnRenamed(mapping[1], mapping[0])

        self.spark = spark_session
        self.df1_unq_rows = self.df2_unq_rows = None
        self._df1_row_count = self._df2_row_count = self._intersect_rows_count = None
        self._joined_dataframe = None
        self._all_matched_rows = None
        self._all_rows_mismatched = None
        self.column_stats = None
        self.columns_match_dict = {}

        # drop the duplicates before actual comparison made.
        self.df1 = df1.dropDuplicates(self.join_columns)
        self.df2 = df2.dropDuplicates(self.join_columns)
        self.df1_dtypes = dict(self.df1.dtypes)
        self.df2_dtypes = dict(self.df2.dtypes)

        if cache_intermediates:
            self.df1.cache()
            self.df2.cache()

        self._merge_dataframes()

    def _validate_dataframe(self, df1_or_df2):
        """Check that it is a dataframe and has the join columns

        Parameters
        ----------
        df1_or_df2 : str
            The "index" of the dataframe - df1 or df2.
        """
        dataframe = getattr(self, df1_or_df2)
        original_dataframe = getattr(self, "_original_" + df1_or_df2)
        if not isinstance(dataframe, pyspark.sql.dataframe.DataFrame):
            raise TypeError("{} must be a PySpark DataFrame".format(df1_or_df2))

        # Check if join_columns are present in the dataframe
        if not set(self.join_columns).issubset(set(dataframe.columns)):
            raise ValueError("{} must have all columns from join_columns".format(df1_or_df2))

        if len(set(dataframe.columns)) < len(dataframe.columns):
            raise ValueError("{} must have unique column names".format(df1_or_df2))

        if dataframe.count() < original_dataframe.count():
            self._any_dupes = True

    def _tuplizer(self, input_list):
        """Return a list of tuples for mapping join columns together"""
        return [(val, val) if isinstance(val, str) else val for val in input_list]

    @property
    def columns_compared(self):
        """set([str]): Get columns to be compared in both dataframes (all columns in both excluding
        the join key(s)"""
        return self.intersect_columns - set(self.join_columns)

    def df_row_count(self, df1_or_df2):
        """int: Get the count of rows in a dataframe"""
        attr = "_" + df1_or_df2 + "_row_count"
        if getattr(self, attr) is None:
            setattr(self, attr, getattr(self, df1_or_df2).count())
        return getattr(self, attr)

    @property
    def intersect_rows_count(self):
        """int: Get the count of rows in common between df1 and df2 dataframes, regardless of
        whether they match or not"""
        if self._intersect_rows_count is None:
            intersect_rows = self._get_or_create_joined_dataframe()
            self._intersect_rows_count = intersect_rows.count()

        return self._intersect_rows_count

    @property
    def matching_rows_count(self):
        # match_dataframe contains columns from both dataframes with flag to indicate if columns matched
        match_dataframe = self._get_or_create_joined_dataframe().select(*self.columns_compared)
        match_dataframe.createOrReplaceTempView("matched_df")

        where_cond = " AND ".join(
            ["A." + name + "=" + str(MatchType.MATCH.value) for name in self.columns_compared]
        )
        match_query = r"SELECT count(*) AS row_count FROM matched_df A WHERE {}".format(where_cond)
        return self.spark.sql(match_query).head()[0]

    def _get_unq_df1_rows(self):
        """Get the rows only from df1 data frame"""
        return self.df1.select(self.join_columns).subtract(self.df2.select(self.join_columns))

    def _get_unq_df2_rows(self):
        """Get the rows only from df2 data frame"""
        return self.df2.select(self.join_columns).subtract(self.df1.select(self.join_columns))

    def df_unq_rows(self, df1_or_df2):
        """pyspark.sql.DataFrame: Returns rows only in the specified dataframe"""
        attr = df1_or_df2 + "_unq_rows"
        if not getattr(self, attr, None):
            rows = getattr(self, "_get_unq_" + df1_or_df2 + "_rows")()
            rows.createOrReplaceTempView("unique_rows")
            getattr(self, df1_or_df2).createOrReplaceTempView("whole_table")
            join_condition = " AND ".join(
                ["A." + name + "=B." + name for name in self.join_columns]
            )
            sql_query = "select A.* from whole_table as A, unique_rows as B where " + join_condition
            setattr(self, attr, self.spark.sql(sql_query))

            if self.cache_intermediates:
                getattr(self, attr).cache().count()

        return getattr(self, attr)

    def cnt_df_unq_rows(self, df1_or_df2):
        return self.df_unq_rows(df1_or_df2).count()

    def _generate_select_statement(self, match_data=True):
        """This function is to generate the select statement to be used later in the query.  For
        intersect columns this returns column which takes the values of the matchtype enum,
        column_df1 and column_df2.

        Parameters
        ----------
        match_data : bool

        Returns
        -------
        str
            The SQL that's before the FROM statement.
        """

        sorted_list = sorted(
            list(self.df1_unq_columns | self.df2_unq_columns | self.intersect_columns)
        )
        select_statement = []

        for column_name in sorted_list:
            if column_name in self.columns_compared:
                if match_data:
                    select_statement.append(self._create_case_statement(name=column_name))
                else:
                    select_statement.append(self._create_select_statement(name=column_name))
            elif column_name in self.df1_unq_columns | set(self.join_columns):
                select_statement.append("A." + column_name)
            elif column_name in self.df2_unq_columns:
                if match_data:
                    select_statement.append("B." + column_name)
                else:
                    select_statement.append("A." + column_name)

        return " , ".join(select_statement)

    def _merge_dataframes(self):
        """Merges the two dataframes and creates self._all_matched_rows and self._all_rows_mismatched."""
        full_joined_dataframe = self._get_or_create_joined_dataframe()
        full_joined_dataframe.createOrReplaceTempView("full_matched_table")

        select_statement = self._generate_select_statement(match_data=False)
        select_query = "SELECT {} FROM full_matched_table A".format(select_statement)

        self._all_matched_rows = self.spark.sql(select_query).orderBy(self.join_columns)
        self._all_matched_rows.createOrReplaceTempView("matched_table")

        where_cond = " OR ".join(["A." + name + "_match= False" for name in self.columns_compared])
        mismatch_query = """SELECT * FROM matched_table A WHERE {}""".format(where_cond)
        self._all_rows_mismatched = self.spark.sql(mismatch_query).orderBy(self.join_columns)
        _ = [self.df_unq_rows("df1"), self.df_unq_rows("df2")]

    def _get_or_create_joined_dataframe(self):
        if self._joined_dataframe is None:
            join_condition = " AND ".join(
                ["A." + name + "=B." + name for name in self.join_columns]
            )
            select_statement = self._generate_select_statement(match_data=True)

            self.df1.createOrReplaceTempView("df1_table")
            self.df2.createOrReplaceTempView("df2_table")

            join_query = r"""
                   SELECT {}
                   FROM df1_table A
                   JOIN df2_table B
                   ON {}""".format(
                select_statement, join_condition
            )

            self._joined_dataframe = self.spark.sql(join_query)
            if self.cache_intermediates:
                self._joined_dataframe.cache()

        return self._joined_dataframe

    def _populate_columns_match_dict(self):
        """
        side effects:
            columns_match_dict assigned to { column -> match_type_counts }
                where:
                    column (string): Name of a column that exists in both the df1 and comparison
                    columns
                    match_type_counts (list of int with size = len(MatchType)): The number of each
                    match type seen for this column (in order of the MatchType enum values)

        returns: None
        """

        match_dataframe = self._get_or_create_joined_dataframe().select(*self.columns_compared)

        def helper(c):
            # Create a predicate for each match type, comparing column values to the match type value
            predicates = [F.col(c) == k.value for k in MatchType]
            # Create a tuple(number of match types found for each match type in this column)
            return F.struct([F.lit(F.sum(pred.cast("integer"))) for pred in predicates]).alias(c)

        # For each column, create a single tuple. This tuple's values correspond to the number of times
        # each match type appears in that column
        match_data = match_dataframe.agg(*[helper(col) for col in self.columns_compared]).collect()[
            0
        ]

        for c in self.columns_compared:
            self.columns_match_dict[c] = match_data[c]
        column_stats_temp = [
            [
                column,
                value[MatchType.MATCH.value],
                (value[MatchType.MISMATCH.value] + value[MatchType.NULL_DIFFERENCE.value]),
                self.df1_dtypes[column],
                self.df2_dtypes[column],
                (value[MatchType.MISMATCH.value] + value[MatchType.NULL_DIFFERENCE.value]) == 0,
                -1,  # TODO: implement this
                value[MatchType.NULL_DIFFERENCE.value],
                value[MatchType.KNOWN_DIFFERENCE.value],
            ]
            for column, value in self.columns_match_dict.items()
        ]
        self.column_stats = pd.DataFrame(
            column_stats_temp,
            columns=[
                "column",
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

    def _create_select_statement(self, name):
        if self._known_differences:
            match_type_comparison = ""
            for k in MatchType:
                match_type_comparison += " WHEN (A.{name}={match_value}) THEN '{match_name}'".format(
                    name=name, match_value=str(k.value), match_name=k.name
                )
            return (
                "A.{name}_df1, "
                "A.{name}_df2, "
                "CASE WHEN A.{name} IN ({mismatch}, {null_difference}) THEN False ELSE True END AS {name}_match, "
                "CASE {match_type_comparison} ELSE 'UNDEFINED' END AS {name}_match_type "
            ).format(
                name=name,
                mismatch=MatchType.MISMATCH.value,
                null_difference=MatchType.NULL_DIFFERENCE.value,
                match_type_comparison=match_type_comparison,
            )
        else:
            return (
                "A.{name}_df1, "
                "A.{name}_df2, "
                "CASE WHEN A.{name} IN ({match_failure}, {null_difference}) THEN False ELSE True END AS {name}_match "
            ).format(
                name=name,
                match_failure=MatchType.MISMATCH.value,
                null_difference=MatchType.NULL_DIFFERENCE.value,
            )

    def _create_case_statement(self, name):
        equal_comparisons = ["(A.{name} IS NULL AND B.{name} IS NULL)"]
        known_diff_comparisons = ["(FALSE)"]

        if (
            self.df1_dtypes[name] in NUMERIC_SPARK_TYPES
            and self.df2_dtypes[name] in NUMERIC_SPARK_TYPES
        ):
            # numeric tolerance comparison
            equal_comparisons.append(
                "((A.{name}=B.{name}) OR ((abs(A.{name}-B.{name}))<=("
                + str(self.abs_tol)
                + "+("
                + str(self.rel_tol)
                + "*abs(A.{name})))))"
            )
        elif self.df1_dtypes[name] == self.df2_dtypes[name]:  # non-numeric comparison
            equal_comparisons.append("((A.{name}=B.{name}))")

        if self._known_differences:
            new_input = "B.{name}"
            for kd in self._known_differences:
                if self.df2_dtypes[name] in kd["types"]:
                    if "flags" in kd and "nullcheck" in kd["flags"]:
                        known_diff_comparisons.append(
                            "(("
                            + kd["transformation"].format(new_input, input=new_input)
                            + ") is null AND A.{name} is null)"
                        )
                    else:
                        known_diff_comparisons.append(
                            "(("
                            + kd["transformation"].format(new_input, input=new_input)
                            + ") = A.{name})"
                        )

        # Null differences can be OR since AND case has been taken care of above
        null_differences = "(A.{name} IS NULL OR B.{name} IS NULL)".format(name=name)

        case_string = (
            "( CASE WHEN ("
            + " OR ".join(equal_comparisons)
            + ") THEN {match_success} WHEN ("
            + " OR ".join(known_diff_comparisons)
            + ") THEN {match_known_difference} WHEN "
            + "{null_differences} THEN {match_null_difference} ELSE {match_failure} END) "
            + "AS {name}, A.{name} AS {name}_df1, B.{name} AS {name}_df2"
        )

        return case_string.format(
            name=name,
            null_differences=null_differences,
            match_success=MatchType.MATCH.value,
            match_known_difference=MatchType.KNOWN_DIFFERENCE.value,
            match_null_difference=MatchType.NULL_DIFFERENCE.value,
            match_failure=MatchType.MISMATCH.value,
        )

    def _df1_to_df2_name(self, df1_name):
        """Translates a column name in the df1 dataframe to its counterpart in the
        df2 dataframe, if they are different."""

        if df1_name in self.column_mapping:
            return self.column_mapping[df1_name]
        else:
            for name in self.join_column_tuples:
                if df1_name == name[0]:
                    return name[1]
            return df1_name

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
        if for_display:
            as_df1 = " AS `" + column + " (" + self.df1_name + ")`"
            as_df2 = " AS `" + column + " (" + self.df2_name + ")`"
        else:
            as_df1 = as_df2 = ""
        filter_columns = list(self.join_columns) + [column + "_df1", column + "_df2"] + [column]
        match_dataframe = self._get_or_create_joined_dataframe().select(*filter_columns)
        match_dataframe.createOrReplaceTempView("matched_df")
        select_sql = (
            "SELECT {join_columns}, {column}_df1{as_df1}, {column}_df2{as_df2} "
            " FROM matched_df WHERE {column} = {MISMATCH}"
        )
        self.spark.sql(
            select_sql.format(
                join_columns=",".join(self.join_columns),
                column=column,
                as_df1=as_df1,
                as_df2=as_df2,
                MISMATCH=MatchType.MISMATCH.value,
            )
        ).createOrReplaceTempView("limited_columns")

        sample_sql = "SELECT * FROM limited_columns TABLESAMPLE({sample_count} ROWS)"
        return self.spark.sql(sample_sql.format(sample_count=sample_count))

    def sample_unique_rows(self, df1_or_df2, sample_count=10):
        """Returns a sample sub-dataframe of rows that are only in one dataframe.

        Parameters
        ----------
        df1_or_df2 : {'df1', 'df2'}
            Which dataframe you're picking from
        sample_count : int
            How many samples to take

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            A sample of unique rows from the dataframe you specified.  The columns are trimmed at
            10 columns to make them more print-friendly
        """
        self.df_unq_rows(df1_or_df2).createOrReplaceTempView("unique_rows")
        columns = getattr(self, df1_or_df2).columns[:10]
        sample_sql = "SELECT {columns} FROM unique_rows TABLESAMPLE({sample_count} ROWS)"
        return self.spark.sql(
            sample_sql.format(columns=",".join(columns), sample_count=sample_count)
        )

    def _df_to_string(self, dataframe):
        """Function to return a string representation of a dataframe.  Changes between Pandas and
        Spark.  I can't find a way to convert a Spark dataframe to string, so have to capture
        stdout."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dataframe.show()

        return output.getvalue()

    def _pre_report(self):
        """Processing that needs to happen before the report can happen"""
        self._populate_columns_match_dict()
