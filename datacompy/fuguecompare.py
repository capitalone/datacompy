#
# Copyright 2023 Capital One Services, LLC
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
Compare two DataFrames that are supported by Fugue
"""

import logging
import pickle
from typing import Any, Dict, Iterable, List, Optional, Union

import fugue.api as fa
import pandas as pd
import pyarrow as pa
from fugue import AnyDataFrame

from .core import Compare

LOG = logging.getLogger(__name__)
HASH_COL = "__datacompy__hash__"


def is_match(
    df1: AnyDataFrame,
    df2: AnyDataFrame,
    join_columns: Union[str, List[str]],
    abs_tol: float = 0,
    rel_tol: float = 0,
    df1_name: str = "df1",
    df2_name: str = "df2",
    ignore_spaces: bool = False,
    ignore_case: bool = False,
    cast_column_names_lower: bool = True,
    parallelism: Optional[int] = None,
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

    if strict_schema:
        if tdf1.schema != tdf2.schema:
            return False

    # check that hash columns exist
    if hash_cols not in tdf1.schema or hash_cols not in tdf2.schema:
        return False

    all_cols = tdf1.schema.names
    str_cols = set(f.name for f in tdf1.schema.fields if pa.types.is_string(f.type))
    bucket = (
        parallelism if parallelism is not None else fa.get_current_parallelism() * 2
    )

    def _serialize(dfs: Iterable[pd.DataFrame], left: bool) -> Iterable[Dict[str, Any]]:
        for df in dfs:
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
            params=dict(left=True),
        ),
        fa.transform(
            tdf2,
            _serialize,
            schema="key:int,left:bool,data:binary",
            params=dict(left=False),
        ),
        distinct=False,
    )

    def _comp(df: List[Dict[str, Any]]) -> List[List[Any]]:
        df1 = (
            pd.concat([pickle.loads(r["data"]) for r in df if r["left"]])
            .sort_values(all_cols)
            .reset_index(drop=True)
        )
        df2 = (
            pd.concat([pickle.loads(r["data"]) for r in df if not r["left"]])
            .sort_values(all_cols)
            .reset_index(drop=True)
        )
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
        return [[comp.matches()]]

    matches = fa.as_pandas(
        fa.transform(
            ser, _comp, schema="match:bool", partition=dict(by="key", num=bucket)
        )
    )
    return matches.match.all()
