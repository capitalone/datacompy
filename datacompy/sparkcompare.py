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

import sys
from enum import Enum
from itertools import chain

import six

from datacompy import Compare

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
    match_rates : bool, optional
        If true, match rates by column will be shown in the column summary.

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
        match_rates=False,
        df1_name="df1",
        df2_name="df2",
    ):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        if self.rel_tol < 0 or self.abs_tol < 0:
            raise ValueError("Please enter positive valued tolerances")
        self.show_all_columns = show_all_columns
        self.match_rates = match_rates

        self._original_df1 = df1
        self._original_df2 = df2
        self.cache_intermediates = cache_intermediates

        self.join_columns = self._tuplizer(join_columns)
        self._join_column_names = [name[0] for name in self.join_columns]

        self._known_differences = known_differences

        if column_mapping:
            for mapping in column_mapping:
                df2 = df2.withColumnRenamed(mapping[1], mapping[0])
            self.column_mapping = dict(column_mapping)
        else:
            self.column_mapping = {}

        for mapping in self.join_columns:
            if mapping[1] != mapping[0]:
                df2 = df2.withColumnRenamed(mapping[1], mapping[0])

        self.spark = spark_session
        self.df1_unq_rows = self.df2_unq_rows = None
        self._df1_row_count = self._df2_row_count = self._intersect_rows_count = None
        self._joined_dataframe = None
        self._rows_only_df1 = None
        self._rows_only_df2 = None
        self._all_matched_rows = None
        self._all_rows_mismatched = None
        self.column_stats = None
        self.columns_match_dict = {}

        # drop the duplicates before actual comparison made.
        self.df1 = df1.dropDuplicates(self._join_column_names)
        self.df2 = df2.dropDuplicates(self._join_column_names)

        if cache_intermediates:
            self.df1.cache()
            self.df2.cache()

        self._merge_dataframes()

    def _tuplizer(self, input_list):
        join_columns = []
        for val in input_list:
            if isinstance(val, six.string_types):
                join_columns.append((val, val))
            else:
                join_columns.append(val)

        return join_columns

    @property
    def columns_in_both(self):
        """set[str]: Get columns in both dataframes"""
        return set(self.df1.columns) & set(self.df2.columns)

    @property
    def columns_compared(self):
        """list[str]: Get columns to be compared in both dataframes (all
        columns in both excluding the join key(s)"""
        return [
            column for column in list(self.columns_in_both) if column not in self._join_column_names
        ]

    @property
    def columns_only_df1(self):
        """set[str]: Get columns that are unique to the df1 dataframe"""
        return set(self.df1.columns) - set(self.df2.columns)

    @property
    def columns_only_df2(self):
        """set[str]: Get columns that are unique to the df2 dataframe"""
        return set(self.df2.columns) - set(self.df1.columns)

    def df_row_count(self, index)):
        """int: Get the count of rows in a dataframe"""
        attr = '_' + index + '_row_count'
        if getattr(self, attr) is None:
            setattr(self, attr, getattr(self, index).count())
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
        raise NotImplementedError("Not yet!")

    def _get_unq_df1_rows(self):
        """Get the rows only from df1 data frame"""
        return self.df1.select(self._join_column_names).subtract(
            self.df2.select(self._join_column_names)
        )

    def _get_df2_rows(self):
        """Get the rows only from df2 data frame"""
        return self.df2.select(self._join_column_names).subtract(
            self.df1.select(self._join_column_names)
        )

    def _print_columns_summary(self, myfile):
        print(
            "Number of columns in common with matching schemas: {}".format(
                len(self._columns_with_matching_schema())
            ),
            file=myfile,
        )
        print(
            "Number of columns in common with schema differences: {}".format(
                len(self._columns_with_schemadiff())
            ),
            file=myfile,
        )
        print(
            "Number of columns in df1 but not df2: {}".format(len(self.columns_only_df1)),
            file=myfile,
        )
        print(
            "Number of columns in df2 but not df1: {}".format(len(self.columns_only_df2)),
            file=myfile,
        )

    def _print_only_columns(self, df1_or_df2, myfile):
        """Prints the columns and data types only in either the df1 or df2 datasets"""

        if df1_or_df2 == "df1":
            columns = self.columns_only_df1
            df = self.df1
        elif df1_or_df2 == "df2":
            columns = self.columns_only_df2
            df = self.df2
        else:
            raise ValueError('df1_or_df2 must be df1 or df2, but was "{}"'.format(df1_or_df2))

        if (
            not columns
        ):  # If there are no columns only in this dataframe, don't display this section
            return

        max_length = max([len(col) for col in columns] + [11])
        format_pattern = "{{:{max}s}}".format(max=max_length)

        print("\n****** Columns In {} Only ******".format(df1_or_df2), file=myfile)
        print((format_pattern + "  Dtype").format("Column Name"), file=myfile)
        print("-" * max_length + "  -------------", file=myfile)

        for column in columns:
            col_type = df.select(column).dtypes[0][1]
            print((format_pattern + "  {:13s}").format(column, col_type), file=myfile)

    def _columns_with_matching_schema(self):
        """ This function will identify the columns which has matching schema"""
        col_schema_match = {}
        df1_columns_dict = dict(self.df1.dtypes)
        df2_columns_dict = dict(self.df2.dtypes)

        for df1_row, df1_type in df1_columns_dict.items():
            if df1_row in df2_columns_dict:
                if df1_type in df2_columns_dict.get(df1_row):
                    col_schema_match[df1_row] = df2_columns_dict.get(df1_row)

        return col_schema_match

    def _columns_with_schemadiff(self):
        """ This function will identify the columns which has different schema"""
        col_schema_diff = {}
        df1_columns_dict = dict(self.df1.dtypes)
        df2_columns_dict = dict(self.df2.dtypes)

        for df1_row, df1_type in df1_columns_dict.items():
            if df1_row in df2_columns_dict:
                if df1_type not in df2_columns_dict.get(df1_row):
                    col_schema_diff[df1_row] = dict(
                        df1_type=df1_type, df2_type=df2_columns_dict.get(df1_row)
                    )
        return col_schema_diff

    @property
    def rows_both_mismatch(self):
        """pyspark.sql.DataFrame: Returns all rows in both dataframes that have mismatches"""
        if self._all_rows_mismatched is None:
            self._merge_dataframes()

        return self._all_rows_mismatched

    @property
    def rows_both_all(self):
        """pyspark.sql.DataFrame: Returns all rows in both dataframes"""
        if self._all_matched_rows is None:
            self._merge_dataframes()

        return self._all_matched_rows

    @property
    def rows_only_df1(self):
        """pyspark.sql.DataFrame: Returns rows only in the df1 dataframe"""
        if not self._rows_only_df1:
            df1_rows = self._get_unq_df1_rows()
            df1_rows.createOrReplaceTempView("df1Rows")
            self.df1.createOrReplaceTempView("df1Table")
            join_condition = " AND ".join(
                ["A." + name + "=B." + name for name in self._join_column_names]
            )
            sql_query = "select A.* from df1Table as A, df1Rows as B where {}".format(
                join_condition
            )
            self._rows_only_df1 = self.spark.sql(sql_query)

            if self.cache_intermediates:
                self._rows_only_df1.cache().count()

        return self._rows_only_df1

    @property
    def rows_only_df2(self):
        """pyspark.sql.DataFrame: Returns rows only in the df2 dataframe"""
        if not self._rows_only_df2:
            df2_rows = self._get_df2_rows()
            df2_rows.createOrReplaceTempView("df2Rows")
            self.df2.createOrReplaceTempView("df2Table")
            where_condition = " AND ".join(
                ["A." + name + "=B." + name for name in self._join_column_names]
            )
            sql_query = "select A.* from df2Table as A, df2Rows as B where {}".format(
                where_condition
            )
            self._rows_only_df2 = self.spark.sql(sql_query)

            if self.cache_intermediates:
                self._rows_only_df2.cache().count()

        return self._rows_only_df2

    def _generate_select_statement(self, match_data=True):
        """This function is to generate the select statement to be used later in the query."""
        df1_only = list(set(self.df1.columns) - set(self.df2.columns))
        df2_only = list(set(self.df2.columns) - set(self.df1.columns))
        sorted_list = sorted(list(chain(df1_only, df2_only, self.columns_in_both)))
        select_statement = ""

        for column_name in sorted_list:
            if column_name in self.columns_compared:
                if match_data:
                    select_statement = select_statement + ",".join(
                        [self._create_case_statement(name=column_name)]
                    )
                else:
                    select_statement = select_statement + ",".join(
                        [self._create_select_statement(name=column_name)]
                    )
            elif column_name in df1_only:
                select_statement = select_statement + ",".join(["A." + column_name])

            elif column_name in df2_only:
                if match_data:
                    select_statement = select_statement + ",".join(["B." + column_name])
                else:
                    select_statement = select_statement + ",".join(["A." + column_name])
            elif column_name in self._join_column_names:
                select_statement = select_statement + ",".join(["A." + column_name])

            if column_name != sorted_list[-1]:
                select_statement = select_statement + " , "

        return select_statement

    def _merge_dataframes(self):
        """Merges the two dataframes and creates self._all_matched_rows and self._all_rows_mismatched."""
        full_joined_dataframe = self._get_or_create_joined_dataframe()
        full_joined_dataframe.createOrReplaceTempView("full_matched_table")

        select_statement = self._generate_select_statement(False)
        select_query = """SELECT {} FROM full_matched_table A""".format(select_statement)

        self._all_matched_rows = self.spark.sql(select_query).orderBy(self._join_column_names)
        self._all_matched_rows.createOrReplaceTempView("matched_table")

        where_cond = " OR ".join(["A." + name + "_match= False" for name in self.columns_compared])
        mismatch_query = """SELECT * FROM matched_table A WHERE {}""".format(where_cond)
        self._all_rows_mismatched = self.spark.sql(mismatch_query).orderBy(self._join_column_names)

    def _get_or_create_joined_dataframe(self):
        if self._joined_dataframe is None:
            join_condition = " AND ".join(
                ["A." + name + "=B." + name for name in self._join_column_names]
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

    def _print_num_of_rows_with_column_equality(self, myfile):
        # match_dataframe contains columns from both dataframes with flag to indicate if columns matched
        match_dataframe = self._get_or_create_joined_dataframe().select(*self.columns_compared)
        match_dataframe.createOrReplaceTempView("matched_df")

        where_cond = " AND ".join(
            ["A." + name + "=" + str(MatchType.MATCH.value) for name in self.columns_compared]
        )
        match_query = r"""SELECT count(*) AS row_count FROM matched_df A WHERE {}""".format(
            where_cond
        )
        all_rows_matched = self.spark.sql(match_query)
        matched_rows = all_rows_matched.head()[0]

        print("\n****** Row Comparison ******", file=myfile)
        print(
            "Number of rows with some columns unequal: {}".format(
                self.intersect_rows_count - matched_rows
            ),
            file=myfile,
        )
        print("Number of rows with all columns equal: {}".format(matched_rows), file=myfile)

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

    def _create_select_statement(self, name):
        if self._known_differences:
            match_type_comparison = ""
            for k in MatchType:
                match_type_comparison += " WHEN (A.{name}={match_value}) THEN '{match_name}'".format(
                    name=name, match_value=str(k.value), match_name=k.name
                )
            return "A.{name}_df1, A.{name}_df2, (CASE WHEN (A.{name}={match_failure}) THEN False ELSE True END) AS {name}_match, (CASE {match_type_comparison} ELSE 'UNDEFINED' END) AS {name}_match_type ".format(
                name=name,
                match_failure=MatchType.MISMATCH.value,
                match_type_comparison=match_type_comparison,
            )
        else:
            return "A.{name}_df1, A.{name}_df2, CASE WHEN (A.{name}={match_failure})  THEN False ELSE True END AS {name}_match ".format(
                name=name, match_failure=MatchType.MISMATCH.value
            )

    def _create_case_statement(self, name):
        equal_comparisons = ["(A.{name} IS NULL AND B.{name} IS NULL)"]
        known_diff_comparisons = ["(FALSE)"]

        df1_dtype = [d[1] for d in self.df1.dtypes if d[0] == name][0]
        df2_dtype = [d[1] for d in self.df2.dtypes if d[0] == name][0]

        if _is_comparable(df1_dtype, df2_dtype):
            if (df1_dtype in NUMERIC_SPARK_TYPES) and (
                df2_dtype in NUMERIC_SPARK_TYPES
            ):  # numeric tolerance comparison
                equal_comparisons.append(
                    "((A.{name}=B.{name}) OR ((abs(A.{name}-B.{name}))<=("
                    + str(self.abs_tol)
                    + "+("
                    + str(self.rel_tol)
                    + "*abs(A.{name})))))"
                )
            else:  # non-numeric comparison
                equal_comparisons.append("((A.{name}=B.{name}))")

        if self._known_differences:
            new_input = "B.{name}"
            for kd in self._known_differences:
                if df2_dtype in kd["types"]:
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

        case_string = (
            "( CASE WHEN ("
            + " OR ".join(equal_comparisons)
            + ") THEN {match_success} WHEN ("
            + " OR ".join(known_diff_comparisons)
            + ") THEN {match_known_difference} ELSE {match_failure} END) "
            + "AS {name}, A.{name} AS {name}_df1, B.{name} AS {name}_df2"
        )

        return case_string.format(
            name=name,
            match_success=MatchType.MATCH.value,
            match_known_difference=MatchType.KNOWN_DIFFERENCE.value,
            match_failure=MatchType.MISMATCH.value,
        )

    def _print_row_summary(self, myfile):
        df1_cnt = self.df1.count()
        df2_cnt = self.df2.count()
        df1_with_dup_cnt = self._original_df1.count()
        df2_with_dup_cnt = self._original_df2.count()

        print("\n****** Row Summary ******", file=myfile)
        print("Number of rows in common: {}".format(self.intersect_rows_count), file=myfile)
        print(
            "Number of rows in df1 but not df2: {}".format(df1_cnt - self.intersect_rows_count),
            file=myfile,
        )
        print(
            "Number of rows in df2 but not df1: {}".format(df2_cnt - self.intersect_rows_count),
            file=myfile,
        )
        print(
            "Number of duplicate rows found in df1: {}".format(df1_with_dup_cnt - df1_cnt),
            file=myfile,
        )
        print(
            "Number of duplicate rows found in df2: {}".format(df2_with_dup_cnt - df2_cnt),
            file=myfile,
        )

    def _print_schema_diff_details(self, myfile):
        schema_diff_dict = self._columns_with_schemadiff()

        if not schema_diff_dict:  # If there are no differences, don't print the section
            return

        # For columns with mismatches, what are the longest df1 and df2 column name lengths (with minimums)?
        df1_name_max = max([len(key) for key in schema_diff_dict] + [16])
        df2_name_max = max([len(self._df1_to_df2_name(key)) for key in schema_diff_dict] + [19])

        format_pattern = "{{:{df1}s}}  {{:{df2}s}}".format(df1=df1_name_max, df2=df2_name_max)

        print("\n****** Schema Differences ******", file=myfile)
        print(
            (format_pattern + "  df1 Dtype      df2 Dtype").format(
                "df1 Column Name", "df2 Column Name"
            ),
            file=myfile,
        )
        print(
            "-" * df1_name_max + "  " + "-" * df2_name_max + "  -------------  -------------",
            file=myfile,
        )

        for df1_column, types in schema_diff_dict.items():
            df2_column = self._df1_to_df2_name(df1_column)

            print(
                (format_pattern + "  {:13s}  {:13s}").format(
                    df1_column, df2_column, types["df1_type"], types["df2_type"]
                ),
                file=myfile,
            )

    def _df1_to_df2_name(self, df1_name):
        """Translates a column name in the df1 dataframe to its counterpart in the
        df2 dataframe, if they are different."""

        if df1_name in self.column_mapping:
            return self.column_mapping[df1_name]
        else:
            for name in self.join_columns:
                if df1_name == name[0]:
                    return name[1]
            return df1_name

    def _print_row_matches_by_column(self, myfile):
        self._populate_columns_match_dict()
        columns_with_mismatches = {
            key: self.columns_match_dict[key]
            for key in self.columns_match_dict
            if self.columns_match_dict[key][MatchType.MISMATCH.value]
        }
        columns_fully_matching = {
            key: self.columns_match_dict[key]
            for key in self.columns_match_dict
            if sum(self.columns_match_dict[key])
            == self.columns_match_dict[key][MatchType.MATCH.value]
        }
        columns_with_any_diffs = {
            key: self.columns_match_dict[key]
            for key in self.columns_match_dict
            if sum(self.columns_match_dict[key])
            != self.columns_match_dict[key][MatchType.MATCH.value]
        }
        df1_types = {x[0]: x[1] for x in self.df1.dtypes}
        df2_types = {x[0]: x[1] for x in self.df2.dtypes}

        print("\n****** Column Comparison ******", file=myfile)

        if self._known_differences:
            print(
                "Number of columns compared with unexpected differences in some values: {}".format(
                    len(columns_with_mismatches)
                ),
                file=myfile,
            )
            print(
                "Number of columns compared with all values equal but known differences found: {}".format(
                    len(self.columns_compared)
                    - len(columns_with_mismatches)
                    - len(columns_fully_matching)
                ),
                file=myfile,
            )
            print(
                "Number of columns compared with all values completely equal: {}".format(
                    len(columns_fully_matching)
                ),
                file=myfile,
            )
        else:
            print(
                "Number of columns compared with some values unequal: {}".format(
                    len(columns_with_mismatches)
                ),
                file=myfile,
            )
            print(
                "Number of columns compared with all values equal: {}".format(
                    len(columns_fully_matching)
                ),
                file=myfile,
            )

        # If all columns matched, don't print columns with unequal values
        if (not self.show_all_columns) and (
            len(columns_fully_matching) == len(self.columns_compared)
        ):
            return

        # if show_all_columns is set, set column name length maximum to max of ALL columns(with minimum)
        if self.show_all_columns:
            df1_name_max = max([len(key) for key in self.columns_match_dict] + [16])
            df2_name_max = max(
                [len(self._df1_to_df2_name(key)) for key in self.columns_match_dict] + [19]
            )

        # For columns with any differences, what are the longest df1 and df2 column name lengths (with minimums)?
        else:
            df1_name_max = max([len(key) for key in columns_with_any_diffs] + [16])
            df2_name_max = max(
                [len(self._df1_to_df2_name(key)) for key in columns_with_any_diffs] + [19]
            )

        """ list of (header, condition, width, align)
                where
                    header (String) : output header for a column
                    condition (Bool): true if this header should be displayed
                    width (Int)     : width of the column
                    align (Bool)    : true if right-aligned
        """
        headers_columns_unequal = [
            ("df1 Column Name", True, df1_name_max, False),
            ("df2 Column Name", True, df2_name_max, False),
            ("df1 Dtype   ", True, 13, False),
            ("df2 Dtype", True, 13, False),
            ("# Matches", True, 9, True),
            ("# Known Diffs", self._known_differences is not None, 13, True),
            ("# Mismatches", True, 12, True),
        ]
        if self.match_rates:
            headers_columns_unequal.append(("Match Rate %", True, 12, True))
        headers_columns_unequal_valid = [h for h in headers_columns_unequal if h[1]]
        padding = 2  # spaces to add to left and right of each column

        if self.show_all_columns:
            print("\n****** Columns with Equal/Unequal Values ******", file=myfile)
        else:
            print("\n****** Columns with Unequal Values ******", file=myfile)

        format_pattern = (" " * padding).join(
            [
                ("{:" + (">" if h[3] else "") + str(h[2]) + "}")
                for h in headers_columns_unequal_valid
            ]
        )
        print(format_pattern.format(*[h[0] for h in headers_columns_unequal_valid]), file=myfile)
        print(
            format_pattern.format(*["-" * len(h[0]) for h in headers_columns_unequal_valid]),
            file=myfile,
        )

        for column_name, column_values in sorted(
            self.columns_match_dict.items(), key=lambda i: i[0]
        ):
            num_matches = column_values[MatchType.MATCH.value]
            num_known_diffs = (
                None
                if self._known_differences is None
                else column_values[MatchType.KNOWN_DIFFERENCE.value]
            )
            num_mismatches = column_values[MatchType.MISMATCH.value]
            df2_column = self._df1_to_df2_name(column_name)

            if num_mismatches or num_known_diffs or self.show_all_columns:
                output_row = [
                    column_name,
                    df2_column,
                    df1_types.get(column_name),
                    df2_types.get(column_name),
                    str(num_matches),
                    str(num_mismatches),
                ]
                if self.match_rates:
                    match_rate = 100 * (
                        1
                        - (column_values[MatchType.MISMATCH.value] + 0.0) / self.intersect_rows_count
                        + 0.0
                    )
                    output_row.append("{:02.5f}".format(match_rate))
                if num_known_diffs is not None:
                    output_row.insert(len(output_row) - 1, str(num_known_diffs))
                print(format_pattern.format(*output_row), file=myfile)

    # noinspection PyUnresolvedReferences
    def report(self, file=sys.stdout):
        """Creates a comparison report and prints it to the file specified
        (stdout by default).

        Parameters
        ----------
        file : ``file``, optional
            A filehandle to write the report to. By default, this is
            sys.stdout, printing the report to stdout. You can also redirect
            this to an output file, as in the example.

        Examples
        --------
        >>> with open('my_report.txt', 'w') as report_file:
        ...     comparison.report(file=report_file)
        """
        self._report_header(file)
        self._report_column_summary(file)
        self._report_row_summary(file) # One thing to change
        # self._report_column_comparison(file)
        # self._report_column_comparison_samples(sample_count, file)
        # self._report_sample_rows("df1", sample_count, file) # Use spark.createDataFrame(df.rdd.takeSample(False, 3)).toPandas()?
        # self._report_sample_rows("df2", sample_count, file)

        self._print_columns_summary(file)
        self._print_schema_diff_details(file)
        self._print_only_columns("df1", file)
        self._print_only_columns("df2", file)
        self._print_row_summary(file)
        self._print_num_of_rows_with_column_equality(file)
        self._print_row_matches_by_column(file)
