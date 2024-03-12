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

import sys
from enum import Enum
from itertools import chain
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple, Union

try:
    import pyspark
    from pyspark.sql import functions as F
except ImportError:
    pass  # Let non-Spark people at least enjoy the loveliness of the pandas datacompy functionality


class MatchType(Enum):
    MISMATCH, MATCH, KNOWN_DIFFERENCE = range(3)


# Used for checking equality with decimal(X, Y) types. Otherwise treated as the string "decimal".
def decimal_comparator() -> str:
    class DecimalComparator(str):
        def __eq__(self, other: str) -> bool:  # type: ignore[override]
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


def _is_comparable(type1: str, type2: str) -> bool:
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


class SparkCompare:
    """Comparison class used to compare two Spark Dataframes.

    Extends the ``Compare`` functionality to the wide world of Spark and
    out-of-memory data.

    Parameters
    ----------
    spark_session : ``pyspark.sql.SparkSession``
        A ``SparkSession`` to be used to execute Spark commands in the
        comparison.
    base_df : ``pyspark.sql.DataFrame``
        The dataframe to serve as a basis for comparison. While you will
        ultimately get the same results comparing A to B as you will comparing
        B to A, by convention ``base_df`` should be the canonical, gold
        standard reference dataframe in the comparison.
    compare_df : ``pyspark.sql.DataFrame``
        The dataframe to be compared against ``base_df``.
    join_columns : list
        A list of columns comprising the join key(s) of the two dataframes.
        If the column names are the same in the two dataframes, the names of
        the columns can be given as strings. If the names differ, the
        ``join_columns`` list should include tuples of the form
        (base_column_name, compare_column_name).
    column_mapping : list[tuple], optional
        If columns to be compared have different names in the base and compare
        dataframes, a list should be provided in ``columns_mapping`` consisting
        of tuples of the form (base_column_name, compare_column_name) for each
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
        compare dataframe to match values when there are known differences
        between base and compare. The dictionaries should contain:

            * name: A name that describes the transformation
            * types: The types that the transformation should be applied to.
                This prevents certain transformations from being applied to
                types that don't make sense and would cause exceptions.
            * transformation: A Spark SQL statement to apply to the column
                in the compare dataset. The string "{input}" will be replaced
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
        spark_session: "pyspark.sql.SparkSession",
        base_df: "pyspark.sql.DataFrame",
        compare_df: "pyspark.sql.DataFrame",
        join_columns: List[Union[str, Tuple[str, str]]],
        column_mapping: Optional[List[Tuple[str, str]]] = None,
        cache_intermediates: bool = False,
        known_differences: Optional[List[Dict[str, Any]]] = None,
        rel_tol: float = 0,
        abs_tol: float = 0,
        show_all_columns: bool = False,
        match_rates: bool = False,
    ):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        if self.rel_tol < 0 or self.abs_tol < 0:
            raise ValueError("Please enter positive valued tolerances")
        self.show_all_columns = show_all_columns
        self.match_rates = match_rates

        self._original_base_df = base_df
        self._original_compare_df = compare_df
        self.cache_intermediates = cache_intermediates

        self.join_columns = self._tuplizer(input_list=join_columns)
        self._join_column_names = [name[0] for name in self.join_columns]

        self._known_differences = known_differences

        if column_mapping:
            for mapping in column_mapping:
                compare_df = compare_df.withColumnRenamed(mapping[1], mapping[0])
            self.column_mapping = dict(column_mapping)
        else:
            self.column_mapping = {}

        for mapping in self.join_columns:
            if mapping[1] != mapping[0]:
                compare_df = compare_df.withColumnRenamed(mapping[1], mapping[0])

        self.spark = spark_session
        self.base_unq_rows = self.compare_unq_rows = None
        self._base_row_count: Optional[int] = None
        self._compare_row_count: Optional[int] = None
        self._common_row_count: Optional[int] = None
        self._joined_dataframe: Optional["pyspark.sql.DataFrame"] = None
        self._rows_only_base: Optional["pyspark.sql.DataFrame"] = None
        self._rows_only_compare: Optional["pyspark.sql.DataFrame"] = None
        self._all_matched_rows: Optional["pyspark.sql.DataFrame"] = None
        self._all_rows_mismatched: Optional["pyspark.sql.DataFrame"] = None
        self.columns_match_dict: Dict[str, Any] = {}

        # drop the duplicates before actual comparison made.
        self.base_df = base_df.dropDuplicates(self._join_column_names)
        self.compare_df = compare_df.dropDuplicates(self._join_column_names)

        if cache_intermediates:
            self.base_df.cache()
            self._base_row_count = self.base_df.count()
            self.compare_df.cache()
            self._compare_row_count = self.compare_df.count()

    def _tuplizer(
        self, input_list: List[Union[str, Tuple[str, str]]]
    ) -> List[Tuple[str, str]]:
        join_columns: List[Tuple[str, str]] = []
        for val in input_list:
            if isinstance(val, str):
                join_columns.append((val, val))
            else:
                join_columns.append(val)

        return join_columns

    @property
    def columns_in_both(self) -> Set[str]:
        """set[str]: Get columns in both dataframes"""
        return set(self.base_df.columns) & set(self.compare_df.columns)

    @property
    def columns_compared(self) -> List[str]:
        """list[str]: Get columns to be compared in both dataframes (all
        columns in both excluding the join key(s)"""
        return [
            column
            for column in list(self.columns_in_both)
            if column not in self._join_column_names
        ]

    @property
    def columns_only_base(self) -> Set[str]:
        """set[str]: Get columns that are unique to the base dataframe"""
        return set(self.base_df.columns) - set(self.compare_df.columns)

    @property
    def columns_only_compare(self) -> Set[str]:
        """set[str]: Get columns that are unique to the compare dataframe"""
        return set(self.compare_df.columns) - set(self.base_df.columns)

    @property
    def base_row_count(self) -> int:
        """int: Get the count of rows in the de-duped base dataframe"""
        if self._base_row_count is None:
            self._base_row_count = self.base_df.count()

        return self._base_row_count

    @property
    def compare_row_count(self) -> int:
        """int: Get the count of rows in the de-duped compare dataframe"""
        if self._compare_row_count is None:
            self._compare_row_count = self.compare_df.count()

        return self._compare_row_count

    @property
    def common_row_count(self) -> int:
        """int: Get the count of rows in common between base and compare dataframes"""
        if self._common_row_count is None:
            common_rows = self._get_or_create_joined_dataframe()
            self._common_row_count = common_rows.count()

        return self._common_row_count

    def _get_unq_base_rows(self) -> "pyspark.sql.DataFrame":
        """Get the rows only from base data frame"""
        return self.base_df.select(self._join_column_names).subtract(
            self.compare_df.select(self._join_column_names)
        )

    def _get_compare_rows(self) -> "pyspark.sql.DataFrame":
        """Get the rows only from compare data frame"""
        return self.compare_df.select(self._join_column_names).subtract(
            self.base_df.select(self._join_column_names)
        )

    def _print_columns_summary(self, myfile: TextIO) -> None:
        """Prints the column summary details"""
        print("\n****** Column Summary ******", file=myfile)
        print(
            f"Number of columns in common with matching schemas: {len(self._columns_with_matching_schema())}",
            file=myfile,
        )
        print(
            f"Number of columns in common with schema differences: {len(self._columns_with_schemadiff())}",
            file=myfile,
        )
        print(
            f"Number of columns in base but not compare: {len(self.columns_only_base)}",
            file=myfile,
        )
        print(
            f"Number of columns in compare but not base: {len(self.columns_only_compare)}",
            file=myfile,
        )

    def _print_only_columns(self, base_or_compare: str, myfile: TextIO) -> None:
        """Prints the columns and data types only in either the base or compare datasets"""

        if base_or_compare.upper() == "BASE":
            columns = self.columns_only_base
            df = self.base_df
        elif base_or_compare.upper() == "COMPARE":
            columns = self.columns_only_compare
            df = self.compare_df
        else:
            raise ValueError(
                f'base_or_compare must be BASE or COMPARE, but was "{base_or_compare}"'
            )

        # If there are no columns only in this dataframe, don't display this section
        if not columns:
            return

        max_length = max([len(col) for col in columns] + [11])
        format_pattern = f"{{:{max_length}s}}"

        print(f"\n****** Columns In {base_or_compare.title()} Only ******", file=myfile)
        print((format_pattern + "  Dtype").format("Column Name"), file=myfile)
        print("-" * max_length + "  -------------", file=myfile)

        for column in columns:
            col_type = df.select(column).dtypes[0][1]
            print((format_pattern + "  {:13s}").format(column, col_type), file=myfile)

    def _columns_with_matching_schema(self) -> Dict[str, str]:
        """This function will identify the columns which has matching schema"""
        col_schema_match = {}
        base_columns_dict = dict(self.base_df.dtypes)
        compare_columns_dict = dict(self.compare_df.dtypes)

        for base_row, base_type in base_columns_dict.items():
            if base_row in compare_columns_dict:
                compare_column_type = compare_columns_dict.get(base_row)
                if compare_column_type is not None and base_type in compare_column_type:
                    col_schema_match[base_row] = compare_column_type

        return col_schema_match

    def _columns_with_schemadiff(self) -> Dict[str, Dict[str, str]]:
        """This function will identify the columns which has different schema"""
        col_schema_diff = {}
        base_columns_dict = dict(self.base_df.dtypes)
        compare_columns_dict = dict(self.compare_df.dtypes)

        for base_row, base_type in base_columns_dict.items():
            if base_row in compare_columns_dict:
                compare_column_type = compare_columns_dict.get(base_row)
                if (
                    compare_column_type is not None
                    and base_type not in compare_column_type
                ):
                    col_schema_diff[base_row] = dict(
                        base_type=base_type,
                        compare_type=compare_column_type,
                    )
        return col_schema_diff

    @property
    def rows_both_mismatch(self) -> Optional["pyspark.sql.DataFrame"]:
        """pyspark.sql.DataFrame: Returns all rows in both dataframes that have mismatches"""
        if self._all_rows_mismatched is None:
            self._merge_dataframes()

        return self._all_rows_mismatched

    @property
    def rows_both_all(self) -> Optional["pyspark.sql.DataFrame"]:
        """pyspark.sql.DataFrame: Returns all rows in both dataframes"""
        if self._all_matched_rows is None:
            self._merge_dataframes()

        return self._all_matched_rows

    @property
    def rows_only_base(self) -> "pyspark.sql.DataFrame":
        """pyspark.sql.DataFrame: Returns rows only in the base dataframe"""
        if not self._rows_only_base:
            base_rows = self._get_unq_base_rows()
            base_rows.createOrReplaceTempView("baseRows")
            self.base_df.createOrReplaceTempView("baseTable")
            join_condition = " AND ".join(
                ["A." + name + "<=>B." + name for name in self._join_column_names]
            )
            sql_query = "select A.* from baseTable as A, baseRows as B where {}".format(
                join_condition
            )
            self._rows_only_base = self.spark.sql(sql_query)

            if self.cache_intermediates:
                self._rows_only_base.cache().count()

        return self._rows_only_base

    @property
    def rows_only_compare(self) -> Optional["pyspark.sql.DataFrame"]:
        """pyspark.sql.DataFrame: Returns rows only in the compare dataframe"""
        if not self._rows_only_compare:
            compare_rows = self._get_compare_rows()
            compare_rows.createOrReplaceTempView("compareRows")
            self.compare_df.createOrReplaceTempView("compareTable")
            where_condition = " AND ".join(
                ["A." + name + "<=>B." + name for name in self._join_column_names]
            )
            sql_query = (
                "select A.* from compareTable as A, compareRows as B where {}".format(
                    where_condition
                )
            )
            self._rows_only_compare = self.spark.sql(sql_query)

            if self.cache_intermediates:
                self._rows_only_compare.cache().count()

        return self._rows_only_compare

    def _generate_select_statement(self, match_data: bool = True) -> str:
        """This function is to generate the select statement to be used later in the query."""
        base_only = list(set(self.base_df.columns) - set(self.compare_df.columns))
        compare_only = list(set(self.compare_df.columns) - set(self.base_df.columns))
        sorted_list = sorted(list(chain(base_only, compare_only, self.columns_in_both)))
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
            elif column_name in base_only:
                select_statement = select_statement + ",".join(["A." + column_name])

            elif column_name in compare_only:
                if match_data:
                    select_statement = select_statement + ",".join(["B." + column_name])
                else:
                    select_statement = select_statement + ",".join(["A." + column_name])
            elif column_name in self._join_column_names:
                select_statement = select_statement + ",".join(["A." + column_name])

            if column_name != sorted_list[-1]:
                select_statement = select_statement + " , "

        return select_statement

    def _merge_dataframes(self) -> None:
        """Merges the two dataframes and creates self._all_matched_rows and self._all_rows_mismatched."""
        full_joined_dataframe = self._get_or_create_joined_dataframe()
        full_joined_dataframe.createOrReplaceTempView("full_matched_table")

        select_statement = self._generate_select_statement(False)
        select_query = """SELECT {} FROM full_matched_table A""".format(
            select_statement
        )
        self._all_matched_rows = self.spark.sql(select_query).orderBy(
            self._join_column_names  # type: ignore[arg-type]
        )
        self._all_matched_rows.createOrReplaceTempView("matched_table")

        where_cond = " OR ".join(
            ["A." + name + "_match= False" for name in self.columns_compared]
        )
        mismatch_query = """SELECT * FROM matched_table A WHERE {}""".format(where_cond)
        self._all_rows_mismatched = self.spark.sql(mismatch_query).orderBy(
            self._join_column_names  # type: ignore[arg-type]
        )

    def _get_or_create_joined_dataframe(self) -> "pyspark.sql.DataFrame":
        if self._joined_dataframe is None:
            join_condition = " AND ".join(
                ["A." + name + "<=>B." + name for name in self._join_column_names]
            )
            select_statement = self._generate_select_statement(match_data=True)

            self.base_df.createOrReplaceTempView("base_table")
            self.compare_df.createOrReplaceTempView("compare_table")

            join_query = r"""
                   SELECT {}
                   FROM base_table A
                   JOIN compare_table B
                   ON {}""".format(
                select_statement, join_condition
            )

            self._joined_dataframe = self.spark.sql(join_query)
            if self.cache_intermediates:
                self._joined_dataframe.cache()
                self._common_row_count = self._joined_dataframe.count()

        return self._joined_dataframe

    def _print_num_of_rows_with_column_equality(self, myfile: TextIO) -> None:
        # match_dataframe contains columns from both dataframes with flag to indicate if columns matched
        match_dataframe = self._get_or_create_joined_dataframe().select(
            *self.columns_compared
        )
        match_dataframe.createOrReplaceTempView("matched_df")

        where_cond = " AND ".join(
            [
                "A." + name + "=" + str(MatchType.MATCH.value)
                for name in self.columns_compared
            ]
        )
        match_query = (
            r"""SELECT count(*) AS row_count FROM matched_df A WHERE {}""".format(
                where_cond
            )
        )
        all_rows_matched = self.spark.sql(match_query)
        all_rows_matched_head = all_rows_matched.head()
        matched_rows = (
            all_rows_matched_head[0] if all_rows_matched_head is not None else 0
        )

        print("\n****** Row Comparison ******", file=myfile)
        print(
            f"Number of rows with some columns unequal: {self.common_row_count - matched_rows}",
            file=myfile,
        )
        print(f"Number of rows with all columns equal: {matched_rows}", file=myfile)

    def _populate_columns_match_dict(self) -> None:
        """
        side effects:
            columns_match_dict assigned to { column -> match_type_counts }
                where:
                    column (string): Name of a column that exists in both the base and comparison columns
                    match_type_counts (list of int with size = len(MatchType)): The number of each match type seen for this column (in order of the MatchType enum values)

        returns: None
        """

        match_dataframe = self._get_or_create_joined_dataframe().select(
            *self.columns_compared
        )

        def helper(c: str) -> "pyspark.sql.Column":
            # Create a predicate for each match type, comparing column values to the match type value
            predicates = [F.col(c) == k.value for k in MatchType]
            # Create a tuple(number of match types found for each match type in this column)
            return F.struct(
                [F.lit(F.sum(pred.cast("integer"))) for pred in predicates]
            ).alias(c)

        # For each column, create a single tuple. This tuple's values correspond to the number of times
        # each match type appears in that column
        match_data_agg = match_dataframe.agg(
            *[helper(col) for col in self.columns_compared]
        ).collect()
        match_data = match_data_agg[0]

        for c in self.columns_compared:
            self.columns_match_dict[c] = match_data[c]

    def _create_select_statement(self, name: str) -> str:
        if self._known_differences:
            match_type_comparison = ""
            for k in MatchType:
                match_type_comparison += (
                    " WHEN (A.{name}={match_value}) THEN '{match_name}'".format(
                        name=name, match_value=str(k.value), match_name=k.name
                    )
                )
            return "A.{name}_base, A.{name}_compare, (CASE WHEN (A.{name}={match_failure}) THEN False ELSE True END) AS {name}_match, (CASE {match_type_comparison} ELSE 'UNDEFINED' END) AS {name}_match_type ".format(
                name=name,
                match_failure=MatchType.MISMATCH.value,
                match_type_comparison=match_type_comparison,
            )
        else:
            return "A.{name}_base, A.{name}_compare, CASE WHEN (A.{name}={match_failure})  THEN False ELSE True END AS {name}_match ".format(
                name=name, match_failure=MatchType.MISMATCH.value
            )

    def _create_case_statement(self, name: str) -> str:
        equal_comparisons = ["(A.{name} IS NULL AND B.{name} IS NULL)"]
        known_diff_comparisons = ["(FALSE)"]

        base_dtype = [d[1] for d in self.base_df.dtypes if d[0] == name][0]
        compare_dtype = [d[1] for d in self.compare_df.dtypes if d[0] == name][0]

        if _is_comparable(base_dtype, compare_dtype):
            if (base_dtype in NUMERIC_SPARK_TYPES) and (
                compare_dtype in NUMERIC_SPARK_TYPES
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
                if compare_dtype in kd["types"]:
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
            + "AS {name}, A.{name} AS {name}_base, B.{name} AS {name}_compare"
        )

        return case_string.format(
            name=name,
            match_success=MatchType.MATCH.value,
            match_known_difference=MatchType.KNOWN_DIFFERENCE.value,
            match_failure=MatchType.MISMATCH.value,
        )

    def _print_row_summary(self, myfile: TextIO) -> None:
        base_df_cnt = self.base_df.count()
        compare_df_cnt = self.compare_df.count()
        base_df_with_dup_cnt = self._original_base_df.count()
        compare_df_with_dup_cnt = self._original_compare_df.count()

        print("\n****** Row Summary ******", file=myfile)
        print(f"Number of rows in common: {self.common_row_count}", file=myfile)
        print(
            f"Number of rows in base but not compare: {base_df_cnt - self.common_row_count}",
            file=myfile,
        )
        print(
            f"Number of rows in compare but not base: {compare_df_cnt - self.common_row_count}",
            file=myfile,
        )
        print(
            f"Number of duplicate rows found in base: {base_df_with_dup_cnt - base_df_cnt}",
            file=myfile,
        )
        print(
            f"Number of duplicate rows found in compare: {compare_df_with_dup_cnt - compare_df_cnt}",
            file=myfile,
        )

    def _print_schema_diff_details(self, myfile: TextIO) -> None:
        schema_diff_dict = self._columns_with_schemadiff()

        if not schema_diff_dict:  # If there are no differences, don't print the section
            return

        # For columns with mismatches, what are the longest base and compare column name lengths (with minimums)?
        base_name_max = max([len(key) for key in schema_diff_dict] + [16])
        compare_name_max = max(
            [len(self._base_to_compare_name(key)) for key in schema_diff_dict] + [19]
        )

        format_pattern = "{{:{base}s}}  {{:{compare}s}}".format(
            base=base_name_max, compare=compare_name_max
        )

        print("\n****** Schema Differences ******", file=myfile)
        print(
            (format_pattern + "  Base Dtype     Compare Dtype").format(
                "Base Column Name", "Compare Column Name"
            ),
            file=myfile,
        )
        print(
            "-" * base_name_max
            + "  "
            + "-" * compare_name_max
            + "  -------------  -------------",
            file=myfile,
        )

        for base_column, types in schema_diff_dict.items():
            compare_column = self._base_to_compare_name(base_column)

            print(
                (format_pattern + "  {:13s}  {:13s}").format(
                    base_column,
                    compare_column,
                    types["base_type"],
                    types["compare_type"],
                ),
                file=myfile,
            )

    def _base_to_compare_name(self, base_name: str) -> str:
        """Translates a column name in the base dataframe to its counterpart in the
        compare dataframe, if they are different."""

        if base_name in self.column_mapping:
            return self.column_mapping[base_name]
        else:
            for name in self.join_columns:
                if base_name == name[0]:
                    return name[1]
            return base_name

    def _print_row_matches_by_column(self, myfile: TextIO) -> None:
        self._populate_columns_match_dict()
        columns_with_mismatches = {
            key: self.columns_match_dict[key]
            for key in self.columns_match_dict
            if self.columns_match_dict[key][MatchType.MISMATCH.value]
        }

        # corner case: when all columns match but no rows match
        # issue: #276
        try:
            columns_fully_matching = {
                key: self.columns_match_dict[key]
                for key in self.columns_match_dict
                if sum(self.columns_match_dict[key])
                == self.columns_match_dict[key][MatchType.MATCH.value]
            }
        except TypeError:
            columns_fully_matching = {}

        try:
            columns_with_any_diffs = {
                key: self.columns_match_dict[key]
                for key in self.columns_match_dict
                if sum(self.columns_match_dict[key])
                != self.columns_match_dict[key][MatchType.MATCH.value]
            }
        except TypeError:
            columns_with_any_diffs = {}
        #

        base_types = {x[0]: x[1] for x in self.base_df.dtypes}
        compare_types = {x[0]: x[1] for x in self.compare_df.dtypes}

        print("\n****** Column Comparison ******", file=myfile)

        if self._known_differences:
            print(
                f"Number of columns compared with unexpected differences in some values: {len(columns_with_mismatches)}",
                file=myfile,
            )
            print(
                f"Number of columns compared with all values equal but known differences found: {len(self.columns_compared) - len(columns_with_mismatches) - len(columns_fully_matching)}",
                file=myfile,
            )
            print(
                f"Number of columns compared with all values completely equal: {len(columns_fully_matching)}",
                file=myfile,
            )
        else:
            print(
                f"Number of columns compared with some values unequal: {len(columns_with_mismatches)}",
                file=myfile,
            )
            print(
                f"Number of columns compared with all values equal: {len(columns_fully_matching)}",
                file=myfile,
            )

        # If all columns matched, don't print columns with unequal values
        if (not self.show_all_columns) and (
            len(columns_fully_matching) == len(self.columns_compared)
        ):
            return

        # if show_all_columns is set, set column name length maximum to max of ALL columns(with minimum)
        if self.show_all_columns:
            base_name_max = max([len(key) for key in self.columns_match_dict] + [16])
            compare_name_max = max(
                [
                    len(self._base_to_compare_name(key))
                    for key in self.columns_match_dict
                ]
                + [19]
            )

        # For columns with any differences, what are the longest base and compare column name lengths (with minimums)?
        else:
            base_name_max = max([len(key) for key in columns_with_any_diffs] + [16])
            compare_name_max = max(
                [len(self._base_to_compare_name(key)) for key in columns_with_any_diffs]
                + [19]
            )

        """ list of (header, condition, width, align)
                where
                    header (String) : output header for a column
                    condition (Bool): true if this header should be displayed
                    width (Int)     : width of the column
                    align (Bool)    : true if right-aligned
        """
        headers_columns_unequal = [
            ("Base Column Name", True, base_name_max, False),
            ("Compare Column Name", True, compare_name_max, False),
            ("Base Dtype   ", True, 13, False),
            ("Compare Dtype", True, 13, False),
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
        print(
            format_pattern.format(*[h[0] for h in headers_columns_unequal_valid]),
            file=myfile,
        )
        print(
            format_pattern.format(
                *["-" * len(h[0]) for h in headers_columns_unequal_valid]
            ),
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
            compare_column = self._base_to_compare_name(column_name)

            if num_mismatches or num_known_diffs or self.show_all_columns:
                output_row = [
                    column_name,
                    compare_column,
                    base_types.get(column_name),
                    compare_types.get(column_name),
                    str(num_matches),
                    str(num_mismatches),
                ]
                if self.match_rates:
                    match_rate = 100 * (
                        1
                        - (column_values[MatchType.MISMATCH.value] + 0.0)
                        / self.common_row_count
                        + 0.0
                    )
                    output_row.append("{:02.5f}".format(match_rate))
                if num_known_diffs is not None:
                    output_row.insert(len(output_row) - 1, str(num_known_diffs))
                print(format_pattern.format(*output_row), file=myfile)

    # noinspection PyUnresolvedReferences
    def report(self, file: TextIO = sys.stdout) -> None:
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

        self._print_columns_summary(file)
        self._print_schema_diff_details(file)
        self._print_only_columns("BASE", file)
        self._print_only_columns("COMPARE", file)
        self._print_row_summary(file)
        self._merge_dataframes()
        self._print_num_of_rows_with_column_equality(file)
        self._print_row_matches_by_column(file)
