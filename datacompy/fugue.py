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

"""Compare two DataFrames that are supported by Fugue."""

import logging
import pickle
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Tuple, cast

import pandas as pd
from ordered_set import OrderedSet

from datacompy.base import df_to_str, render, save_html_report
from datacompy.core import Compare

LOG = logging.getLogger(__name__)
HASH_COL = "__datacompy__hash__"


try:
    import fugue.api as fa
    import pyarrow as pa
    from fugue import AnyDataFrame
    from triad import Schema
except ImportError:
    LOG.warning(
        "Please note that you are missing the optional dependency: fugue. "
        "If you need to use this functionality it must be installed."
    )


def unq_columns(df1: "AnyDataFrame", df2: "AnyDataFrame") -> OrderedSet[str]:
    """Get columns that are unique to df1.

    Parameters
    ----------
    df1 : ``AnyDataFrame``
        First dataframe to check

    df2 : ``AnyDataFrame``
        Second dataframe to check

    Returns
    -------
    OrderedSet
        Set of columns that are unique to df1
    """
    col1 = fa.get_column_names(df1)
    col2 = fa.get_column_names(df2)
    return cast(OrderedSet[str], OrderedSet(col1) - OrderedSet(col2))


def intersect_columns(df1: "AnyDataFrame", df2: "AnyDataFrame") -> OrderedSet[str]:
    """Get columns that are shared between the two dataframes.

    Parameters
    ----------
    df1 : ``AnyDataFrame``
        First dataframe to check

    df2 : ``AnyDataFrame``
        Second dataframe to check

    Returns
    -------
    OrderedSet
        Set of that are shared between the two dataframes
    """
    col1 = fa.get_column_names(df1)
    col2 = fa.get_column_names(df2)
    return OrderedSet(col1) & OrderedSet(col2)


def all_columns_match(df1: "AnyDataFrame", df2: "AnyDataFrame") -> bool:
    """Whether the columns all match in the dataframes.

    Parameters
    ----------
    df1 : ``AnyDataFrame``
        First dataframe to check

    df2 : ``AnyDataFrame``
        Second dataframe to check

    Returns
    -------
    bool
        Boolean indicating whether the columns all match in the dataframes
    """
    return unq_columns(df1, df2) == unq_columns(df2, df1) == set()


def is_match(
    df1: "AnyDataFrame",
    df2: "AnyDataFrame",
    join_columns: str | List[str],
    abs_tol: float = 0,
    rel_tol: float = 0,
    df1_name: str = "df1",
    df2_name: str = "df2",
    ignore_spaces: bool = False,
    ignore_case: bool = False,
    cast_column_names_lower: bool = True,
    parallelism: int | None = None,
    strict_schema: bool = False,
) -> bool:
    """Check whether two dataframes match.

    Both df1 and df2 should be dataframes containing all of the join_columns,
    with unique column names. Differences between values are compared to
    abs_tol + rel_tol * abs(df2['value']).

    Parameters
    ----------
    df1 : ``AnyDataFrame``
        First dataframe to check
    df2 : ``AnyDataFrame``
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
    parallelism: int, optional
        An integer representing the amount of parallelism. Entering a value for this
        will force to use of Fugue over just vanilla Pandas
    strict_schema: bool, optional
        The schema must match exactly if set to ``True``. This includes the names and types. Allows for a fast fail.

    Returns
    -------
    bool
        Returns boolean as to if the DataFrames match.
    """
    if (
        isinstance(df1, pd.DataFrame)
        and isinstance(df2, pd.DataFrame)
        and parallelism is None  # user did not specify parallelism
        and fa.get_current_parallelism() == 1  # currently on a local execution engine
    ):
        comp = Compare(
            df1=df1,
            df2=df2,
            join_columns=join_columns,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            df1_name=df1_name,
            df2_name=df2_name,
            ignore_spaces=ignore_spaces,
            ignore_case=ignore_case,
            cast_column_names_lower=cast_column_names_lower,
        )
        return comp.matches()

    try:
        matches = _distributed_compare(
            df1=df1,
            df2=df2,
            join_columns=join_columns,
            return_obj_func=lambda comp: comp.matches(),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            df1_name=df1_name,
            df2_name=df2_name,
            ignore_spaces=ignore_spaces,
            ignore_case=ignore_case,
            cast_column_names_lower=cast_column_names_lower,
            parallelism=parallelism,
            strict_schema=strict_schema,
        )
    except _StrictSchemaError:
        return False

    return all(matches)


def all_rows_overlap(
    df1: "AnyDataFrame",
    df2: "AnyDataFrame",
    join_columns: str | List[str],
    abs_tol: float = 0,
    rel_tol: float = 0,
    df1_name: str = "df1",
    df2_name: str = "df2",
    ignore_spaces: bool = False,
    ignore_case: bool = False,
    cast_column_names_lower: bool = True,
    parallelism: int | None = None,
    strict_schema: bool = False,
) -> bool:
    """Check if the rows are all present in both dataframes.

    Parameters
    ----------
    df1 : ``AnyDataFrame``
        First dataframe to check
    df2 : ``AnyDataFrame``
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
    parallelism: int, optional
        An integer representing the amount of parallelism. Entering a value for this
        will force to use of Fugue over just vanilla Pandas
    strict_schema: bool, optional
        The schema must match exactly if set to ``True``. This includes the names and types. Allows for a fast fail.

    Returns
    -------
    bool
        True if all rows in df1 are in df2 and vice versa (based on
        existence for join option)
    """
    if (
        isinstance(df1, pd.DataFrame)
        and isinstance(df2, pd.DataFrame)
        and parallelism is None  # user did not specify parallelism
        and fa.get_current_parallelism() == 1  # currently on a local execution engine
    ):
        comp = Compare(
            df1=df1,
            df2=df2,
            join_columns=join_columns,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            df1_name=df1_name,
            df2_name=df2_name,
            ignore_spaces=ignore_spaces,
            ignore_case=ignore_case,
            cast_column_names_lower=cast_column_names_lower,
        )
        return comp.all_rows_overlap()

    try:
        overlap = _distributed_compare(
            df1=df1,
            df2=df2,
            join_columns=join_columns,
            return_obj_func=lambda comp: comp.all_rows_overlap(),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            df1_name=df1_name,
            df2_name=df2_name,
            ignore_spaces=ignore_spaces,
            ignore_case=ignore_case,
            cast_column_names_lower=cast_column_names_lower,
            parallelism=parallelism,
            strict_schema=strict_schema,
        )
    except _StrictSchemaError:
        return False

    return all(overlap)


def count_matching_rows(
    df1: "AnyDataFrame",
    df2: "AnyDataFrame",
    join_columns: str | List[str],
    abs_tol: float = 0,
    rel_tol: float = 0,
    df1_name: str = "df1",
    df2_name: str = "df2",
    ignore_spaces: bool = False,
    ignore_case: bool = False,
    cast_column_names_lower: bool = True,
    parallelism: int | None = None,
    strict_schema: bool = False,
) -> int:
    """Count the number of rows match (on overlapping fields).

    Parameters
    ----------
    df1 : ``AnyDataFrame``
        First dataframe to check
    df2 : ``AnyDataFrame``
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
    parallelism: int, optional
        An integer representing the amount of parallelism. Entering a value for this
        will force to use of Fugue over just vanilla Pandas
    strict_schema: bool, optional
        The schema must match exactly if set to ``True``. This includes the names and types. Allows for a fast fail.

    Returns
    -------
    int
        Number of matching rows
    """
    if (
        isinstance(df1, pd.DataFrame)
        and isinstance(df2, pd.DataFrame)
        and parallelism is None  # user did not specify parallelism
        and fa.get_current_parallelism() == 1  # currently on a local execution engine
    ):
        comp = Compare(
            df1=df1,
            df2=df2,
            join_columns=join_columns,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            df1_name=df1_name,
            df2_name=df2_name,
            ignore_spaces=ignore_spaces,
            ignore_case=ignore_case,
            cast_column_names_lower=cast_column_names_lower,
        )
        return comp.count_matching_rows()

    try:
        count_matching_rows = _distributed_compare(
            df1=df1,
            df2=df2,
            join_columns=join_columns,
            return_obj_func=lambda comp: comp.count_matching_rows(),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            df1_name=df1_name,
            df2_name=df2_name,
            ignore_spaces=ignore_spaces,
            ignore_case=ignore_case,
            cast_column_names_lower=cast_column_names_lower,
            parallelism=parallelism,
            strict_schema=strict_schema,
        )
    except _StrictSchemaError:
        return False

    return sum(count_matching_rows)


def report(
    df1: "AnyDataFrame",
    df2: "AnyDataFrame",
    join_columns: str | List[str],
    abs_tol: float = 0,
    rel_tol: float = 0,
    df1_name: str = "df1",
    df2_name: str = "df2",
    ignore_spaces: bool = False,
    ignore_case: bool = False,
    cast_column_names_lower: bool = True,
    sample_count: int = 10,
    column_count: int = 10,
    html_file: str | None = None,
    parallelism: int | None = None,
) -> str:
    """Return a string representation of a report.

    The representation can
    then be printed or saved to a file.

    Both df1 and df2 should be dataframes containing all of the join_columns,
    with unique column names. Differences between values are compared to
    abs_tol + rel_tol * abs(df2['value']).

    Parameters
    ----------
    df1 : ``AnyDataFrame``
        First dataframe to check
    df2 : ``AnyDataFrame``
        Second dataframe to check
    join_columns : list or str
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
    parallelism: int, optional
        An integer representing the amount of parallelism. Entering a value for this
        will force to use of Fugue over just vanilla Pandas
    strict_schema: bool, optional
        The schema must match exactly if set to ``True``. This includes the names and types. Allows for a fast fail.
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
    if isinstance(join_columns, str):
        join_columns = [join_columns]

    if (
        isinstance(df1, pd.DataFrame)
        and isinstance(df2, pd.DataFrame)
        and parallelism is None  # user did not specify parallelism
        and fa.get_current_parallelism() == 1  # currently on a local execution engine
    ):
        comp = Compare(
            df1=df1,
            df2=df2,
            join_columns=join_columns,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            df1_name=df1_name,
            df2_name=df2_name,
            ignore_spaces=ignore_spaces,
            ignore_case=ignore_case,
            cast_column_names_lower=cast_column_names_lower,
        )
        return comp.report(
            sample_count=sample_count, column_count=column_count, html_file=html_file
        )

    res = _distributed_compare(
        df1=df1,
        df2=df2,
        join_columns=join_columns,
        return_obj_func=lambda c: _get_compare_result(
            c, sample_count=sample_count, column_count=column_count
        ),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        df1_name=df1_name,
        df2_name=df2_name,
        ignore_spaces=ignore_spaces,
        ignore_case=ignore_case,
        cast_column_names_lower=cast_column_names_lower,
        parallelism=parallelism,
        strict_schema=False,
    )

    first = res[0]

    def shape0(col: str) -> int:
        return sum(x[col][0] for x in res)

    def shape1(col: str) -> Any:
        return first[col][1]

    def _sum(col: str) -> int:
        return sum(x[col] for x in res)

    def _any(col: str) -> int:
        return any(x[col] for x in res)

    # Fugue implementation doesn't support joining on index
    on_index = False

    # Prepare template data
    column_stats: List[Dict[str, Any]]
    match_sample: List[pd.DataFrame]
    column_stats, match_sample = _aggregate_stats(res, sample_count=sample_count)
    any_mismatch = len(match_sample) > 0

    # Prepare column comparison stats
    match_stats = []
    for column in column_stats:
        if not column["all_match"]:
            match_stats.append(
                {
                    "column": column["column"],
                    "dtype1": str(column["dtype1"])
                    .replace("dtype('", "")
                    .replace("')", ""),
                    "dtype2": str(column["dtype2"])
                    .replace("dtype('", "")
                    .replace("')", ""),
                    "unequal_cnt": column["unequal_cnt"],
                    "max_diff": column["max_diff"],
                    "null_diff": column["null_diff"],
                }
            )

    # Get unique rows samples
    df1_unq_rows_samples = [
        r["df1_unq_rows_sample"] for r in res if r["df1_unq_rows_sample"] is not None
    ]
    df2_unq_rows_samples = [
        r["df2_unq_rows_sample"] for r in res if r["df2_unq_rows_sample"] is not None
    ]

    df1_unq_sample = None
    if df1_unq_rows_samples:
        df1_unq_sample = _sample(
            pd.concat(df1_unq_rows_samples), sample_count=sample_count
        )

    df2_unq_sample = None
    if df2_unq_rows_samples:
        df2_unq_sample = _sample(
            pd.concat(df2_unq_rows_samples), sample_count=sample_count
        )

    # Prepare template data
    template_data = {
        "column_summary": {
            "common_columns": len(first["intersect_columns"]),
            "df1_unique": len(first["df1_unq_columns"]),
            "df2_unique": len(first["df2_unq_columns"]),
            "df1_name": df1_name,
            "df2_name": df2_name,
        },
        "row_summary": {
            "match_columns": ", ".join(join_columns),
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
            "common_rows": shape0("intersect_rows_shape"),
            "df1_unique": shape0("df1_unq_rows_shape"),
            "df2_unique": shape0("df2_unq_rows_shape"),
            "unequal_rows": shape0("intersect_rows_shape")
            - _sum("count_matching_rows"),
            "equal_rows": _sum("count_matching_rows"),
            "df1_name": df1_name,
            "df2_name": df2_name,
            "has_duplicates": "Yes" if _any("_any_dupes") else "No",
        },
        "column_comparison": {
            "unequal_columns": len(
                [col for col in column_stats if col["unequal_cnt"] > 0]
            ),
            "equal_columns": len(
                [col for col in column_stats if col["unequal_cnt"] == 0]
            ),
            "unequal_values": sum(col["unequal_cnt"] for col in column_stats),
        },
        "df1_name": df1_name,
        "df2_name": df2_name,
        "df1_shape": (shape0("df1_shape"), shape1("df1_shape")),  # (rows, columns)
        "df2_shape": (shape0("df2_shape"), shape1("df2_shape")),  # (rows, columns)
        "column_stats": match_stats,
        "sample_count": sample_count,
        "column_count": column_count,
        "df_to_str": lambda x: df_to_str(
            x, sample_count=sample_count, on_index=on_index
        ),
        "mismatch_stats": {
            "has_mismatches": any_mismatch,
            "has_samples": len(match_sample) > 0 and sample_count > 0,
            "samples": [s.to_string() for s in match_sample]
            if sample_count > 0
            else [],
            "df1_name": df1_name,
            "df2_name": df2_name,
            "stats": [
                {
                    "column": col["column"],
                    "dtype1": col["dtype1"],
                    "dtype2": col["dtype2"],
                    "unequal_cnt": col["unequal_cnt"],
                    "max_diff": col["max_diff"],
                    "null_diff": col["null_diff"],
                    "rel_tol": rel_tol,
                    "abs_tol": abs_tol,
                }
                for col in column_stats
                if not col["all_match"]
            ],
        },
        "df1_unique_rows": {
            "has_rows": df1_unq_sample is not None and len(df1_unq_sample) > 0,
            "rows": df_to_str(df1_unq_sample, sample_count=sample_count),
            "columns": list(df1_unq_rows_samples[0].columns)
            if df1_unq_rows_samples
            else [],
        },
        "df2_unique_rows": {
            "has_rows": df2_unq_sample is not None and len(df2_unq_sample) > 0,
            "rows": df_to_str(df2_unq_sample, sample_count=sample_count),
            "columns": list(df2_unq_rows_samples[0].columns)
            if df2_unq_rows_samples
            else [],
        },
    }

    # Render the report
    rpt = render("report_template.j2", **template_data)

    # Handle HTML output if requested
    if html_file:
        save_html_report(rpt, html_file)

    return rpt


def _distributed_compare(
    df1: "AnyDataFrame",
    df2: "AnyDataFrame",
    join_columns: str | List[str],
    return_obj_func: Callable[[Compare], Any],
    abs_tol: float = 0,
    rel_tol: float = 0,
    df1_name: str = "df1",
    df2_name: str = "df2",
    ignore_spaces: bool = False,
    ignore_case: bool = False,
    cast_column_names_lower: bool = True,
    parallelism: int | None = None,
    strict_schema: bool = False,
) -> List[Any]:
    """Compare the data distributively using the core Compare class.

    Both df1 and df2 should be dataframes containing all of the join_columns,
    with unique column names. Differences between values are compared to
    abs_tol + rel_tol * abs(df2['value']).

    Parameters
    ----------
    df1 : ``AnyDataFrame``
        First dataframe to check
    df2 : ``AnyDataFrame``
        Second dataframe to check
    join_columns : list or str
        Column(s) to join dataframes on.  If a string is passed in, that one
        column will be used.
    return_obj_func : Callable[[Compare], Any]
        A function that takes in a Compare object and returns a picklable value.
        It determines what is returned from the distributed compare.
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
    parallelism: int, optional
        An integer representing the amount of parallelism. Entering a value for this
        will force to use of Fugue over just vanilla Pandas
    strict_schema: bool, optional
        The schema must match exactly if set to ``True``. This includes the names and types. Allows for a fast fail.

    Returns
    -------
    List[Any]
        Returns the list of objects returned from the return_obj_func
    """
    tdf1 = fa.as_fugue_df(df1)
    tdf2 = fa.as_fugue_df(df2)

    if isinstance(join_columns, str):
        hash_cols = [join_columns]
    else:
        hash_cols = join_columns

    if cast_column_names_lower:
        tdf1 = tdf1.rename(
            {col: col.lower() for col in tdf1.schema.names if col != col.lower()}
        )
        tdf2 = tdf2.rename(
            {col: col.lower() for col in tdf2.schema.names if col != col.lower()}
        )
        hash_cols = [col.lower() for col in hash_cols]

    if strict_schema and tdf1.schema != tdf2.schema:
        raise _StrictSchemaError()

    # check that hash columns exist
    assert hash_cols in tdf1.schema, f"{hash_cols} not found in {tdf1.schema}"
    assert hash_cols in tdf2.schema, f"{hash_cols} not found in {tdf2.schema}"

    df1_schema = tdf1.schema
    df2_schema = tdf2.schema
    str_cols = {f.name for f in tdf1.schema.fields if pa.types.is_string(f.type)}
    bucket = (
        parallelism if parallelism is not None else fa.get_current_parallelism() * 2
    )

    def _serialize(dfs: Iterable[pd.DataFrame], left: bool) -> Iterable[Dict[str, Any]]:
        for df in dfs:
            df = df.convert_dtypes()
            cols = {}
            for name in df.columns:
                col = df[name]
                if name in str_cols:
                    if ignore_spaces:
                        col = col.str.strip()
                    if ignore_case:
                        col = col.str.lower()
                cols[name] = col
            data = pd.DataFrame(cols)
            gp = pd.util.hash_pandas_object(df[hash_cols], index=False).mod(bucket)
            for k, sub in data.groupby(gp, as_index=False, group_keys=False):
                yield {"key": k, "left": left, "data": pickle.dumps(sub)}

    ser = fa.union(
        fa.transform(
            tdf1,
            _serialize,
            schema="key:int,left:bool,data:binary",
            params={"left": True},
        ),
        fa.transform(
            tdf2,
            _serialize,
            schema="key:int,left:bool,data:binary",
            params={"left": False},
        ),
        distinct=False,
    )

    def _deserialize(
        df: List[Dict[str, Any]], left: bool, schema: Schema
    ) -> pd.DataFrame:
        arr = [pickle.loads(r["data"]) for r in df if r["left"] == left]
        if len(arr) > 0:
            return cast(
                pd.DataFrame,
                pd.concat(arr).sort_values(schema.names).reset_index(drop=True),
            )
        # The following is how to construct an empty pandas dataframe with
        # the correct schema, it avoids pandas schema inference which is wrong.
        # This is not needed when upgrading to Fugue >= 0.8.7
        sample_row: List[Any] = []
        for field in schema.fields:
            if pa.types.is_string(field.type):
                sample_row.append("x")
            elif pa.types.is_integer(field.type):
                sample_row.append(1)
            elif pa.types.is_floating(field.type):
                sample_row.append(1.1)
            elif pa.types.is_boolean(field.type):
                sample_row.append(True)
            elif pa.types.is_timestamp(field.type):
                sample_row.append(pd.NaT)
            else:
                sample_row.append(None)
        return (
            pd.DataFrame([sample_row], columns=schema.names)
            .astype(schema.pandas_dtype)
            .convert_dtypes()
            .head(0)
        )

    def _comp(df: List[Dict[str, Any]]) -> List[List[Any]]:
        df1 = _deserialize(df, True, df1_schema)
        df2 = _deserialize(df, False, df2_schema)
        comp = Compare(
            df1=df1,
            df2=df2,
            join_columns=join_columns,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            df1_name=df1_name,
            df2_name=df2_name,
            cast_column_names_lower=False,
        )
        return [[pickle.dumps(return_obj_func(comp))]]

    objs = fa.as_array(
        fa.transform(
            ser, _comp, schema="obj:binary", partition={"by": "key", "num": bucket}
        )
    )
    return [pickle.loads(row[0]) for row in objs]


def _get_compare_result(
    compare: Compare, sample_count: int, column_count: int
) -> Dict[str, Any]:
    mismatch_samples: Dict[str, pd.DataFrame] = {}
    for column in compare.column_stats:
        if not column["all_match"] and column["unequal_cnt"] > 0:
            mismatch_samples[column["column"]] = compare.sample_mismatch(
                column["column"], sample_count, for_display=True
            )

    df1_unq_rows_sample: Any = None
    if min(sample_count, compare.df1_unq_rows.shape[0]) > 0:
        columns = compare.df1_unq_rows.columns[:column_count]
        unq_count = min(sample_count, compare.df1_unq_rows.shape[0])
        df1_unq_rows_sample = _sample(compare.df1_unq_rows, sample_count=unq_count)[
            columns
        ]

    df2_unq_rows_sample: Any = None
    if min(sample_count, compare.df2_unq_rows.shape[0]) > 0:
        columns = compare.df2_unq_rows.columns[:column_count]
        unq_count = min(sample_count, compare.df2_unq_rows.shape[0])
        df2_unq_rows_sample = _sample(compare.df2_unq_rows, sample_count=unq_count)[
            columns
        ]

    return {
        "match": compare.matches(),
        "count_matching_rows": compare.count_matching_rows(),
        "intersect_columns": compare.intersect_columns(),
        "df1_shape": compare.df1.shape,
        "df2_shape": compare.df2.shape,
        "intersect_rows_shape": compare.intersect_rows.shape,
        "df1_unq_rows_shape": compare.df1_unq_rows.shape,
        "df1_unq_columns": compare.df1_unq_columns(),
        "df2_unq_rows_shape": compare.df2_unq_rows.shape,
        "df2_unq_columns": compare.df2_unq_columns(),
        "abs_tol": compare.abs_tol,
        "rel_tol": compare.rel_tol,
        "df1_name": compare.df1_name,
        "df2_name": compare.df2_name,
        "column_stats": compare.column_stats,
        "mismatch_samples": mismatch_samples,
        "df1_unq_rows_sample": df1_unq_rows_sample,
        "df2_unq_rows_sample": df2_unq_rows_sample,
        "_any_dupes": compare._any_dupes,
    }


def _aggregate_stats(
    compares: List[Any], sample_count: int
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    samples = defaultdict(list)
    stats = []
    for compare in compares:
        stats.extend(compare["column_stats"])
        for k, v in compare["mismatch_samples"].items():
            samples[k].append(v)

    df = pd.DataFrame(stats)
    df = (
        df.groupby("column", as_index=False, group_keys=True)
        .agg(
            {
                "match_column": "first",
                "match_cnt": "sum",
                "unequal_cnt": "sum",
                "dtype1": "first",
                "dtype2": "first",
                "all_match": "all",
                "max_diff": "max",
                "null_diff": "sum",
            }
        )
        .reset_index(drop=False)
    )
    return cast(
        Tuple[List[Dict[str, Any]], List[pd.DataFrame]],
        (
            df.to_dict(orient="records"),
            [
                _sample(pd.concat(v), sample_count=sample_count)
                for v in samples.values()
            ],
        ),
    )


def _sample(df: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    if len(df) <= sample_count:
        return df.reset_index(drop=True)
    return df.sample(n=sample_count, random_state=0).reset_index(drop=True)


class _StrictSchemaError(Exception):
    """Exception raised when strict schema is enabled and the schemas do not match."""

    pass
