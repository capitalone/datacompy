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
Compare two Polars DataFrames

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""
import logging
from typing import Any, Dict, List, Optional, Union

import duckdb
import fugue.api as fa
import pandas as pd
import pyarrow as pa
from fugue import AnyDataFrame
from ordered_set import OrderedSet
from triad.utils.schema import quote_name

LOG = logging.getLogger(__name__)

_SIDE_FLAG = "_datacompy_side"
_ROW_DIFF_FLAG = "_datacompy_row_diff"
_TOTAL_COUNT_COL = "_datacompy_total_count"


class FugueSQLBuilder:
    def __init__(
        self,
        df1: AnyDataFrame,
        df2: AnyDataFrame,
        join_columns: Union[List[str], str],
        abs_tol: float = 0,
        rel_tol: float = 0,
        df1_name: str = "df1",
        df2_name: str = "df2",
        ignore_spaces: bool = False,
        ignore_case: bool = False,
        cast_column_names_lower: bool = True,
        dedup: bool = False,
    ) -> None:
        assert rel_tol == 0, "Relative tolerance is not supported"
        assert not ignore_case, "Ignore case is not supported"
        assert not ignore_spaces, "Ignore spaces is not supported"

        self.df1_name = df1_name
        self.df2_name = df2_name
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.ignore_spaces = ignore_spaces
        self.ignore_case = ignore_case
        self.dedup = dedup

        self.df1, self.df2 = fa.as_fugue_df(df1), fa.as_fugue_df(df2)
        self._schema1, self._schema2 = self.df1.schema, self.df2.schema
        self._join_cols = (
            [join_columns] if isinstance(join_columns, str) else join_columns
        )
        assert len(self._join_cols) > 0, "Join columns must be specified"
        df1_map_cols = self._schema1.names
        df2_map_cols = self._schema2.names
        if cast_column_names_lower:
            df1_map_cols = [c.lower() for c in df1_map_cols]
            df2_map_cols = [c.lower() for c in df2_map_cols]
            self._join_cols = [c.lower() for c in self._join_cols]
        self._map1 = {x: y for x, y in zip(df1_map_cols, self._schema1.names)}
        self._rmap1 = {y: x for x, y in zip(df1_map_cols, self._schema1.names)}
        self._map2 = {x: y for x, y in zip(df2_map_cols, self._schema2.names)}
        self._rmap2 = {y: x for x, y in zip(df2_map_cols, self._schema2.names)}
        df1_cols_set = OrderedSet(df1_map_cols)
        df2_cols_set = OrderedSet(df2_map_cols)
        self._df1_unq_cols = df1_cols_set - df2_cols_set
        self._df2_unq_cols = df2_cols_set - df1_cols_set
        self._intersect_cols = df1_cols_set & df2_cols_set
        self._value_cols = self._intersect_cols - OrderedSet(self._join_cols)
        assert len(self._value_cols) > 0, "No value columns to compare"
        assert all(
            x in self._intersect_cols for x in self._join_cols
        ), f"Join columns {self._join_cols} must be in both df1 amd df2"

    def gen_fsql(self, persist_diff: bool = True, sample_count: int = 10) -> str:
        steps: List[str] = []
        self._declare_a_b(steps)
        self._gen_select_diff(steps, persist=persist_diff and sample_count > 0)
        if sample_count > 0:
            self._gen_samples(steps, sample_count=sample_count)
        self._gen_cols_summary(steps)
        return "\n".join(steps)

    def run_duckdb(
        self, persist_diff: bool = False, sample_count: int = 10
    ) -> Dict[str, pd.DataFrame]:
        fsql = self.gen_fsql(persist_diff, sample_count)
        with duckdb.connect() as con:
            res = fa.fugue_sql_flow(fsql, df1=self.df1, df2=self.df2).run(con)
            return {k: fa.as_pandas(v) for k, v in res.items()}

    def _declare_a_b(self, steps: List[str]) -> None:
        scols = "\n\t,".join(
            _quote_name(self._map1[c]) + " AS " + _quote_name(c)
            for c in self._intersect_cols
        )
        steps.append(f"-- Get intersecting columns from {self.df1_name}")
        steps.append(f"a = SELECT {scols} FROM df1")
        if self.dedup:
            steps.append(f"a = TAKE 1 ROW PREPARTITION BY {self._jcols}")
        scols = "\n\t,".join(
            _quote_name(self._map2[c]) + " AS " + _quote_name(c)
            for c in self._intersect_cols
        )
        steps.append(f"-- Get intersecting columns from {self.df2_name}")
        steps.append(f"b = SELECT {scols} FROM df2")
        if self.dedup:
            steps.append(f"b = TAKE 1 ROW PREPARTITION BY {self._jcols}")

    def _gen_select_diff(self, steps: List[str], persist: bool) -> None:
        cols_expr: List[str] = []
        for col in self._join_cols:
            cols_expr.append(f"a.{_quote_name(col)}")
        for col in self._value_cols:
            cols_expr.append(self._gen_col_eq(col))
        _val_diff = "+".join(_quote_name(x) for x in self._value_cols)
        _fa = "a." + _SIDE_FLAG
        _fb = "b." + _SIDE_FLAG
        cols_expr.append(
            f"""
        CASE WHEN 
            {_fa} IS NULL AND {_fb} IS NULL THEN 0
            WHEN {_fb} IS NULL THEN 1
            WHEN {_fa} IS NULL THEN 2
            ELSE 3 END AS {_SIDE_FLAG}"""
        )
        cols = "\n\t,".join(cols_expr)
        _persist = "PERSIST" if persist else ""
        query = f"""
-- Get diff
SELECT {cols}
    FROM (SELECT *, 1 AS {_SIDE_FLAG} FROM df1) a
    FULL OUTER JOIN (SELECT *, 1 AS {_SIDE_FLAG} FROM df2) b
    ON {self._gen_join_on()}

diff =
    SELECT *, CASE WHEN {_val_diff}>0 THEN 1 ELSE 0 END AS {_ROW_DIFF_FLAG}
    {_persist}
        """
        steps.append(query)

    def _gen_samples(self, steps: List[str], sample_count: int) -> None:
        steps.append(
            f"""
-- Get samples
filter =
    SELECT * FROM (
        SELECT {self._jcols}, {_SIDE_FLAG} FROM diff
        WHERE {_SIDE_FLAG} = 1 LIMIT {sample_count})
    UNION ALL
    SELECT * FROM (
        SELECT {self._jcols}, {_SIDE_FLAG} FROM diff
        WHERE {_SIDE_FLAG} = 2 LIMIT {sample_count})
    UNION ALL
    SELECT * FROM (
        SELECT {self._jcols}, {_SIDE_FLAG} FROM diff
        WHERE {_SIDE_FLAG} = 3 AND {_ROW_DIFF_FLAG} > 0
        LIMIT {sample_count})
    PERSIST BROADCAST

df1_samples =
    SELECT df1.*, filter.{_SIDE_FLAG} FROM df1
        INNER JOIN filter ON {self._gen_join_on("df1", "filter")}
    YIELD LOCAL DATAFRAME
df2_samples =
    SELECT df2.*, filter.{_SIDE_FLAG} FROM df2
        INNER JOIN filter ON {self._gen_join_on("df2", "filter")}
    YIELD LOCAL DATAFRAME
"""
        )

    def _gen_cols_summary(self, steps: List[str]) -> None:
        cols_expr: List[str] = []
        for col in list(self._value_cols) + [_ROW_DIFF_FLAG]:
            cols_expr.append(f"SUM({_quote_name(col)}) AS {_quote_name(col)}")
        cols = "\n\t,".join(cols_expr)
        query = f"""
-- Get columns summary
cols_summary =
    SELECT {_SIDE_FLAG}, {cols},
        COUNT(*) AS {_TOTAL_COUNT_COL}
    FROM diff
    GROUP BY {_SIDE_FLAG}
    YIELD LOCAL DATAFRAME"""
        steps.append(query)

    @property
    def _jcols(self) -> str:
        return ", ".join(_quote_name(c) for c in self._join_cols)

    def _gen_join_on(self, name1: str = "a", name2: str = "b") -> str:
        return " AND ".join(
            [
                f"{name1}.{_quote_name(c)} = {name2}.{_quote_name(c)}"
                for c in self._join_cols
            ]
        )

    def _gen_col_eq(self, name: str) -> str:
        tp = self._schema1[self._rmap1[name]].type
        _f = _quote_name(name)
        _fa = "a." + _f
        _fb = "b." + _f
        _both_null = f"({_fa} IS NULL AND {_fb} IS NULL)"
        if pa.types.is_string(tp):
            c = f"{_fa} = {_fb}"
        elif pa.types.is_floating(tp) and self.abs_tol > 0:
            c = f"({_fa} - {_fb} <= {self.abs_tol} AND {_fa} - {_fb} >= -1 * {self.abs_tol})"
        else:
            c = f"{_fa} = {_fb}"
        return f"CASE WHEN {_both_null} OR {c} THEN 0 ELSE 1 END AS {_f}"


def _quote_name(name: str) -> str:
    return quote_name(name, "`")
