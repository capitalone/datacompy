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
Compare any two DataFrames using Fugue SQL
"""
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import duckdb
import fugue.api as fa
import pandas as pd
import pyarrow as pa
from fugue import AnyDataFrame, DataFrame
from ordered_set import OrderedSet
from triad import Schema
from triad.utils.schema import quote_name

from ._fsql_utils import infer_fugue_engine

LOG = logging.getLogger(__name__)

_SIDE_FLAG = "_datacompy_side"
_ROW_DIFF_FLAG = "_datacompy_row_diff"
_TOTAL_COUNT_COL = "_datacompy_total_count"
_DIFF_PREFIX = "_datacompy_diff_"
_MAX_DIFF_PREFIX = "_datacompy_max_diff_"
_NULL_DIFF_PREFIX = "_datacompy_null_diff_"


def lower_column_names(df: AnyDataFrame) -> AnyDataFrame:
    """
    Lowercase column names in a dataframe

    Args:

        df (AnyDataFrame): Input dataframe

    Returns:

        AnyDataFrame: Dataframe with lowercased column names
    """
    names = fa.get_column_names(df)
    rn = {x: x.lower() for x in names if x != x.lower()}
    if len(rn) > 0:
        return fa.rename(df, rn)  # type: ignore
    return df


def drop_duplicates(
    df: AnyDataFrame, columns: Union[str, List[str]], presort: str = ""
) -> AnyDataFrame:
    """
    Drop duplicates from a dataframe

    Args:

        df (AnyDataFrame): Input dataframe
        columns (Union[str, List[str]]): Columns to use for dedup
        presort (str, optional): Value presort expression. Defaults to "".

    Returns:

        AnyDataFrame: Dataframe with duplicates removed
    """
    cols = _to_cols(columns, allow_empty=False)
    return fa.take(df, 1, presort=presort, partition=cols)  # type: ignore


def compare_schemas(
    schema1: Schema,
    schema2: Schema,
    join_columns: Union[List[str], str],
    exact: bool = False,
) -> "SchemaCompareResult":
    join_cols = _to_cols(join_columns, allow_empty=False)
    assert join_cols in schema1, f"Columns {join_cols} not all in {schema1}"
    assert join_cols in schema2, f"Columns {join_cols} not all in {schema2}"
    df1_cols_set = OrderedSet(schema1.names)
    df2_cols_set = OrderedSet(schema2.names)
    left_only = list(df1_cols_set - df2_cols_set)
    right_only = list(df2_cols_set - df1_cols_set)
    intersect_cols = list(df1_cols_set & df2_cols_set)
    value_cols = list((df1_cols_set & df2_cols_set) - OrderedSet(join_cols))
    assert len(value_cols) > 0, "No value columns to compare"
    s1 = schema1.extract(intersect_cols)
    s2 = schema2.extract(intersect_cols)
    if s1 != s2:
        assert not exact, f"Intersecting columns have different types {s1} vs {s2}"
        for f1, f2 in zip(s1.fields, s2.fields):
            if f1.type != f2.type:
                if (
                    pa.types.is_floating(f1.type)
                    or pa.types.is_integer(f1.type)
                    or pa.types.is_decimal(f1.type)
                ) and (
                    pa.types.is_floating(f2.type)
                    or pa.types.is_integer(f2.type)
                    or pa.types.is_decimal(f2.type)
                ):
                    continue
                if (
                    hasattr(pa.types, "is_large_string")
                    and (
                        pa.types.is_string(f1.type) or pa.types.is_large_string(f1.type)
                    )
                    and (
                        pa.types.is_string(f2.type) or pa.types.is_large_string(f2.type)
                    )
                ):
                    continue
                if (
                    hasattr(pa.types, "is_large_binary")
                    and (
                        pa.types.is_binary(f1.type) or pa.types.is_large_binary(f1.type)
                    )
                    and (
                        pa.types.is_binary(f2.type) or pa.types.is_large_binary(f2.type)
                    )
                ):
                    continue
                if (
                    hasattr(pa.types, "is_large_list")
                    and (pa.types.is_list(f1.type) or pa.types.is_large_list(f1.type))
                    and (pa.types.is_list(f2.type) or pa.types.is_large_list(f2.type))
                ):
                    continue
                raise AssertionError(
                    "Intersecting columns have different and "
                    f"incompatible types {s1} vs {s2}"
                )

    return SchemaCompareResult(
        schema1=schema1,
        schema2=schema2,
        intersect_cols=intersect_cols,
        join_cols=join_cols,
        value_cols=value_cols,
        left_only=left_only,
        right_only=right_only,
    )


def compare(
    df1: Union[AnyDataFrame, str],
    df2: Union[AnyDataFrame, str],
    join_columns: Union[List[str], str],
    exact_type_match: bool = False,
    abs_tol: float = 0,
    rel_tol: float = 0,
    sample_count: int = 10,
    persist_diff: Optional[bool] = None,
    use_map: Optional[bool] = None,
    num_buckets: Optional[int] = None,
) -> "CompareResult":
    with infer_fugue_engine(df1, df2) as conf:
        _df1 = (
            fa.load(df1, as_fugue=True) if isinstance(df1, str) else fa.as_fugue_df(df1)
        )
        _df2 = (
            fa.load(df2, as_fugue=True) if isinstance(df2, str) else fa.as_fugue_df(df2)
        )
        schema_compare = compare_schemas(
            _df1.schema, _df2.schema, join_columns, exact=exact_type_match
        )
        sql_builder = _FugueSQLBuilder(schema_compare, abs_tol, rel_tol)
        _persist_diff = (
            persist_diff if persist_diff is not None else conf["persist_diff"]
        )
        _use_map = use_map if use_map is not None else conf["use_map"]
        _num_buckets = num_buckets if num_buckets is not None else conf["num_buckets"]
        sql = sql_builder.build(_persist_diff, sample_count)
        if _use_map:
            runner = _MapRunner(schema_compare, sql, _num_buckets)
            res = runner.run(_df1, _df2)
        else:
            raw = fa.fugue_sql_flow(sql, df1=_df1, df2=_df2).run()
            res = {k: fa.as_pandas(v) for k, v in raw.items()}
    return CompareResult(
        schema_compare=schema_compare,
        raw_diff_summary=res["diff_summary"],
        df1_samples=res.get("df1_samples"),
        df2_samples=res.get("df2_samples"),
    )


@dataclass
class SchemaCompareResult:
    schema1: Schema
    schema2: Schema
    intersect_cols: List[str]
    join_cols: List[str]
    value_cols: List[str]
    left_only: List[str]
    right_only: List[str]

    def are_equal(self, check_column_order: bool = True) -> bool:
        if len(self.left_only) > 0 or len(self.right_only) > 0:
            return False
        if check_column_order:
            return self.schema1.names == self.schema2.names
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "left_schema": str(self.schema1),
            "right_schema": str(self.schema2),
            "intersect_cols": ",".join(self.intersect_cols),
            "join_cols": ",".join(self.join_cols),
            "value_cols": ",".join(self.value_cols),
            "common_col_count": len(self.value_cols),
            "left_only_cols": ",".join(self.left_only),
            "right_only_cols": ",".join(self.right_only),
        }

    def is_floating(self, name: str) -> bool:
        tp = self.schema1[name].type
        tp2 = self.schema2[name].type
        return (
            pa.types.is_floating(tp)
            or pa.types.is_decimal(tp)
            or pa.types.is_floating(tp2)
            or pa.types.is_decimal(tp2)
        )

    def is_numeric(self, name: str) -> bool:
        tp = self.schema1[name].type
        tp2 = self.schema2[name].type
        return (
            pa.types.is_floating(tp)
            or pa.types.is_decimal(tp)
            or pa.types.is_floating(tp2)
            or pa.types.is_decimal(tp2)
            or pa.types.is_integer(tp)
            or pa.types.is_integer(tp2)
        )


@dataclass
class CompareResult:
    schema_compare: SchemaCompareResult
    raw_diff_summary: pd.DataFrame
    df1_samples: Optional[pd.DataFrame]
    df2_samples: Optional[pd.DataFrame]

    def are_equal(self, check_column_order: bool = True) -> bool:
        if not self.schema_compare.are_equal(check_column_order=check_column_order):
            return False
        diff = self.get_diff_summary()
        return diff["diff"].sum() == 0 and diff["null_diff"].sum() == 0

    def get_stats(self) -> Dict[str, Any]:
        schema_stats = self.schema_compare.get_stats()
        counts = self.get_row_counts()
        row_diff_count = self.get_common_rows_diff_count()
        row_equal_count = counts.get(3, 0) - row_diff_count
        rows_stats = {
            "left_only_row_count": counts.get(1, 0),
            "right_only_row_count": counts.get(2, 0),
            "common_row_count": counts.get(3, 0),
            "common_row_diff_count": row_diff_count,
            "common_row_equal_count": row_equal_count,
        }
        rows_stats.update(schema_stats)
        return rows_stats

    def get_row_counts(self) -> Dict[int, int]:
        return (
            self.raw_diff_summary.groupby(_SIDE_FLAG)[_TOTAL_COUNT_COL].sum().to_dict()
        )

    def get_common_rows_diff_count(self) -> int:
        sub = self.raw_diff_summary[self.raw_diff_summary[_SIDE_FLAG] == 3]
        if len(sub) == 0:
            return 0
        return int(sub[_ROW_DIFF_FLAG].sum())

    def get_diff_summary(self) -> pd.DataFrame:
        res: List[Dict[str, Any]] = []
        df = self.raw_diff_summary[self.raw_diff_summary[_SIDE_FLAG] == 3]
        for col in self.schema_compare.value_cols:
            res.append(
                {
                    "column": col,
                    "diff": int(df[_DIFF_PREFIX + col].sum()),
                    "null_diff": int(df[_NULL_DIFF_PREFIX + col].sum()),
                    "max_diff": float(df[_MAX_DIFF_PREFIX + col].max()),
                }
            )
        return pd.DataFrame(res)

    def get_unique_samples(self, side: int, sample_count: int = 10) -> pd.DataFrame:
        assert side in [1, 2], "side must be 1 or 2"
        return self._get_df(
            self.df1_samples if side == 1 else self.df2_samples,
            side,
            sample_count=sample_count,
        )

    def get_unequal_samples(
        self, sample_count: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.df1_samples is None or self.df2_samples is None:
            return pd.DataFrame(), pd.DataFrame()
        return self._get_df(
            self.df1_samples, 3, sample_count=sample_count
        ), self._get_df(self.df2_samples, 3, sample_count=sample_count)

    def _get_df(
        self, df: Optional[pd.DataFrame], side: int, sample_count: int
    ) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        return (
            df[df[_SIDE_FLAG] == side]
            .nsmallest(sample_count, self.schema_compare.join_cols)
            .reset_index(drop=True)
        )


class _FugueSQLBuilder:
    def __init__(
        self,
        schema_compare: SchemaCompareResult,
        abs_tol: float = 0,
        rel_tol: float = 0,
    ) -> None:
        assert rel_tol >= 0, "Relative tolerance must be non-negative"
        assert abs_tol >= 0, "Absolute tolerance must be non-negative"
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.schema_compare = schema_compare

    def build(self, persist_diff: bool = True, sample_count: int = 10) -> str:
        steps: List[str] = []
        self._declare_a_b(steps)
        self._gen_select_diff(steps, persist=persist_diff and sample_count > 0)
        if sample_count > 0:
            self._gen_samples(steps, sample_count=sample_count)
        self._gen_cols_summary(steps)
        return "\n".join(steps)

    def _declare_a_b(self, steps: List[str]) -> None:
        scols = ", ".join(_quote_name(c) for c in self.schema_compare.intersect_cols)
        steps.append(
            f"""
-- Get intersecting columns")
a = SELECT {scols} FROM df1
b = SELECT {scols} FROM df2"""
        )

    def _gen_select_diff(self, steps: List[str], persist: bool) -> None:
        cols_expr: List[str] = []
        for col in self.schema_compare.join_cols:
            # not using coalesce here
            ca = "a." + _quote_name(col)
            cb = "b." + _quote_name(col)
            cols_expr.append(f"CASE WHEN {ca} IS NULL THEN {cb} ELSE {ca} END AS {col}")
        for col in self.schema_compare.value_cols:
            cols_expr.extend(self._gen_col_eq(col))
        _val_diff = "+".join(
            _quote_name(_DIFF_PREFIX + x) for x in self.schema_compare.value_cols
        )
        _fa = "a." + _SIDE_FLAG
        _fb = "b." + _SIDE_FLAG
        cols_expr.append(
            f"""
        CASE WHEN {_fa} IS NULL AND {_fb} IS NULL THEN 0
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
        for col in self.schema_compare.value_cols:
            cn = _quote_name(_DIFF_PREFIX + col)
            cols_expr.append(f"SUM({cn}) AS {cn}")
            cn = _quote_name(_NULL_DIFF_PREFIX + col)
            cols_expr.append(f"SUM({cn}) AS {cn}")
            cn = _quote_name(_MAX_DIFF_PREFIX + col)
            cols_expr.append(f"MAX({cn}) AS {cn}")
        for col in [_ROW_DIFF_FLAG]:
            cols_expr.append(f"SUM({_quote_name(col)}) AS {_quote_name(col)}")
        cols = "\n\t,".join(cols_expr)
        query = f"""
-- Get diff summary
diff_summary =
    SELECT {cols},
        {_SIDE_FLAG},
        COUNT(*) AS {_TOTAL_COUNT_COL}
    FROM diff
    GROUP BY {_SIDE_FLAG}
    YIELD LOCAL DATAFRAME"""
        steps.append(query)

    @property
    def _jcols(self) -> str:
        return ", ".join(_quote_name(c) for c in self.schema_compare.join_cols)

    def _gen_join_on(self, name1: str = "a", name2: str = "b") -> str:
        return " AND ".join(
            [
                f"{name1}.{_quote_name(c)} = {name2}.{_quote_name(c)}"
                for c in self.schema_compare.join_cols
            ]
        )

    def _gen_col_eq(self, name: str) -> Iterable[str]:
        tp = self.schema_compare.schema1[name].type
        is_floating = self.schema_compare.is_floating(name)
        is_numeric = self.schema_compare.is_numeric(name)
        _f = _quote_name(name)
        _fa = "a." + _f
        _fb = "b." + _f
        _both_null = f"({_fa} IS NULL AND {_fb} IS NULL)"
        _one_null = (
            f"(({_fa} IS NULL AND {_fb} IS NOT NULL) "
            f"OR ({_fa} IS NOT NULL AND {_fb} IS NULL))"
        )
        _no_null = f"({_fa} IS NOT NULL AND {_fb} IS NOT NULL)"
        if pa.types.is_string(tp):
            c = f"{_fa} = {_fb}"
        elif (self.abs_tol > 0 or self.rel_tol > 0) and is_floating:
            # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
            # absolute(a - b) <= (atol + rtol * absolute(b))
            # c = f"ABS({_fa}-{_fb}) <= {self.abs_tol}+{_fb}*ABS({self.rel_tol})"
            a_b_sq = f"({_fa}-{_fb})*({_fa}-{_fb})"
            diff_pos = f"({self.abs_tol}+{_fb}*{self.rel_tol})"
            diff_neg = f"({self.abs_tol}-{_fb}*{self.rel_tol})"
            c = (
                f"CASE WHEN {_fb}>0 THEN {a_b_sq} < {diff_pos}*{diff_pos} "
                f"ELSE {a_b_sq} < {diff_neg}*{diff_neg} END"
            )
        else:
            c = f"{_fa} = {_fb}"
        diff_col = _quote_name(_DIFF_PREFIX + name)
        yield f"CASE WHEN {_both_null} OR {c} THEN 0 ELSE 1 END AS {diff_col}"
        null_diff_col = _quote_name(_NULL_DIFF_PREFIX + name)
        yield f"CASE WHEN {_one_null} THEN 1 ELSE 0 END AS {null_diff_col}"
        max_diff_col = _quote_name(_MAX_DIFF_PREFIX + name)
        if is_numeric:
            yield (
                f"""CASE WHEN {_no_null}
            THEN (
                CASE WHEN {_fa} > {_fb} THEN {_fa} - {_fb} ELSE {_fb} - {_fa} END
            ) ELSE 0 END AS {max_diff_col}"""
            )
        else:
            yield f"0 AS {max_diff_col}"


class _MapRunner:
    def __init__(
        self,
        schema_compare: SchemaCompareResult,
        sql: str,
        num_buckets: int,
    ):
        self.schema_compare = schema_compare
        self.sql = sql
        if num_buckets > 0:
            self.num_buckets = num_buckets
        else:
            self.num_buckets = fa.get_current_parallelism() * 2

    def run(self, df1: DataFrame, df2: DataFrame) -> Dict[str, pd.DataFrame]:
        ser = fa.union(
            fa.transform(
                df1,
                self._serialize,
                schema="key:int,left:bool,data:binary",
                params=dict(left=True),
            ),
            fa.transform(
                df2,
                self._serialize,
                schema="key:int,left:bool,data:binary",
                params=dict(left=False),
            ),
            distinct=False,
        )
        objs = fa.as_array(
            fa.transform(
                ser,
                self._comp,
                schema="obj:binary",
                partition=dict(by="key", num=self.num_buckets),
            )
        )
        dicts = defaultdict(list)
        for obj in objs:
            d = pickle.loads(obj[0])
            for k, v in d.items():
                dicts[k].append(v)
        return {k: pd.concat(v) for k, v in dicts.items()}

    def _serialize(
        self, dfs: Iterable[pa.Table], left: bool
    ) -> Iterable[Dict[str, Any]]:
        for df in dfs:
            keys = df.select(self.schema_compare.join_cols).to_pandas()
            gp = pd.util.hash_pandas_object(keys, index=False).mod(self.num_buckets)
            for k, idx in gp.index.groupby(gp).items():
                sub = df.take(pa.Array.from_pandas(idx))
                yield {"key": k, "left": left, "data": _serialize_pa_table(sub)}

    def _deserialize(self, df: List[Dict[str, Any]], left: bool) -> pa.Table:
        arr = [_deserialize_pa_table(r["data"]) for r in df if r["left"] == left]
        if len(arr) > 0:
            return pa.concat_tables(arr)
        if left:
            return self.schema_compare.schema1.create_empty_arrow_table()
        return self.schema_compare.schema2.create_empty_arrow_table()

    def _comp(self, df: List[Dict[str, Any]]) -> List[List[Any]]:
        df1 = self._deserialize(df, True)
        df2 = self._deserialize(df, False)
        with duckdb.connect() as con:
            with fa.engine_context(con):
                res = fa.fugue_sql_flow(self.sql, df1=df1, df2=df2).run(con)
                data = pickle.dumps({k: fa.as_pandas(v) for k, v in res.items()})
        return [[data]]


def _quote_name(name: str) -> str:
    return quote_name(name, "`")


def _to_cols(columns: Union[str, List[str]], allow_empty: bool) -> List[str]:
    cols = [columns] if isinstance(columns, str) else columns
    assert all(x.strip() != "" for x in cols), f"Empty column names found in {cols}"
    assert len(set(cols)) == len(cols), f"Duplicate columns found in {cols}"
    if not allow_empty:
        assert len(cols) > 0, "Columns must be specified"
    return cols


def _serialize_pa_table(tb: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, tb.schema) as writer:
        writer.write_table(tb)
    return sink.getvalue().to_pybytes()


def _deserialize_pa_table(buf: bytes) -> pa.Table:
    with pa.ipc.open_stream(buf) as reader:
        return reader.read_all()
