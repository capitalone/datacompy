import datetime

import duckdb
import fugue.api as fa
import fugue.test as ft
from fugue import DataFrame
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from triad import Schema
from typing import Any, List

from datacompy.fsql import (
    _deserialize_pa_table,
    _serialize_pa_table,
    _to_cols,
    compare,
    compare_schemas,
    drop_duplicates,
    lower_column_names,
)


def test_to_cols() -> None:
    assert _to_cols("ab", allow_empty=False) == ["ab"]
    assert _to_cols(["a"], allow_empty=False) == ["a"]
    assert _to_cols(["a", "b"], allow_empty=False) == ["a", "b"]
    with pytest.raises(AssertionError):
        _to_cols([], allow_empty=False)
    assert _to_cols([], allow_empty=True) == []
    with pytest.raises(AssertionError):
        _to_cols(["a", "b", "a"], allow_empty=False)
    with pytest.raises(AssertionError):
        _to_cols("", allow_empty=True)
    with pytest.raises(AssertionError):
        _to_cols([""], allow_empty=True)
    with pytest.raises(AssertionError):
        _to_cols(["a", ""], allow_empty=True)


def test_lower_column_names() -> None:
    df = pd.DataFrame({"a b": [1], "b": [4]})
    df = lower_column_names(df)
    assert df.columns.tolist() == ["a b", "b"]
    df = pd.DataFrame({"A b": [1], "b汉": [4]})
    df = lower_column_names(df)
    assert df.columns.tolist() == ["a b", "b汉"]


def test_drop_duplicates() -> None:
    df = pd.DataFrame(
        [
            [1, 2, "a"],
            [1, 3, "b"],
            [1, 2, "c"],
        ],
        columns=["ca", "cb", "cc"],
    )
    with pytest.raises(AssertionError):
        drop_duplicates(df, [])
    res = fa.as_pandas(drop_duplicates(df, ["ca"]))
    assert res.values.tolist() == [[1, 2, "a"]]
    res = drop_duplicates(df, ["ca", "cb"])
    assert res.sort_values(["ca", "cb"]).values.tolist() == [[1, 2, "a"], [1, 3, "b"]]
    res = drop_duplicates(df, ["ca", "cb"], presort="cc DESC")
    assert res.sort_values(["ca", "cb"]).values.tolist() == [[1, 2, "c"], [1, 3, "b"]]


def test_compare_schemas() -> None:
    s1 = Schema("a:int,b:str")
    s2 = Schema("b:str,a:int")
    comp = compare_schemas(s1, s2, "a")
    assert not comp.are_equal()
    assert comp.are_equal(check_column_order=False)
    assert comp.intersect_cols == ["a", "b"]
    assert comp.left_only == []
    assert comp.right_only == []
    assert comp.join_cols == ["a"]
    assert comp.value_cols == ["b"]
    assert not comp.is_floating("a")
    assert not comp.is_floating("b")
    assert comp.is_numeric("a")
    assert not comp.is_numeric("b")

    s1 = Schema("a:long,b:int,d:double")
    s2 = Schema("c:str,a:int16,b:double")
    comp = compare_schemas(s1, s2, "a")
    assert not comp.are_equal()
    assert not comp.are_equal(check_column_order=False)
    assert comp.intersect_cols == ["a", "b"]
    assert comp.left_only == ["d"]
    assert comp.right_only == ["c"]
    assert comp.join_cols == ["a"]
    assert comp.value_cols == ["b"]
    assert not comp.is_floating("a")
    assert comp.is_floating("b")  # one side is floating
    assert comp.is_numeric("a")
    assert comp.is_numeric("b")
    with pytest.raises(AssertionError):
        compare_schemas(s1, s2, "a", exact=True)

    s1 = Schema("a:int,b:str")
    s2 = Schema("a:int,b:int")
    with pytest.raises(AssertionError):
        # type mismatch
        compare_schemas(s1, s2, "a")
    with pytest.raises(AssertionError):
        # no join cols
        compare_schemas(s1, s2, "")
    with pytest.raises(AssertionError):
        # no join cols
        compare_schemas(s1, s2, [])

    s1 = Schema("a:int,b:str")
    s2 = Schema("a:int,c:int")
    with pytest.raises(AssertionError):
        # not all join cols in s2
        compare_schemas(s1, s2, ["a", "b"])
    with pytest.raises(AssertionError):
        # not all join cols in s1
        compare_schemas(s1, s2, ["a", "c"])

    s1 = Schema("a:int,b:str")
    s2 = Schema("a:int,b:str")
    with pytest.raises(AssertionError):
        # no value cols
        compare_schemas(s1, s2, ["a", "b"])


def test_pa_table_serder() -> None:
    src = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = pd.DataFrame(src)
    adf = pa.Table.from_pandas(df)
    s = _serialize_pa_table(adf)
    assert isinstance(s, bytes)
    actual = _deserialize_pa_table(s).to_pydict()
    assert actual == src


class CompareTests(ft.FugueTestSuite):
    def to_df(self, data: Any, schema: Any) -> DataFrame:
        return fa.as_fugue_df(data, schema=schema)

    def test_same_data(self) -> None:
        df1 = self.to_df(
            [
                [0, 1, 2],
                [1, 3, 4],
            ],
            "id:int,a:int,b:int",
        )
        df2 = df1
        res = compare(df1, df2, "id")
        assert res.get_row_counts()[3] == 2
        assert len(res.get_unique_samples(1)) == 0
        assert len(res.get_unique_samples(2)) == 0
        assert len(res.get_unequal_samples()[0]) == 0
        assert res.are_equal()

    def test_overlap(self) -> None:
        df1 = self.to_df(
            [
                [0, 1, 2, "ab"],  # left unique
                [-1, 1, 2, "ab"],  # left unique
                [1, 3, 4, "cd"],  # overlap
                [2, 5, 6, "ef"],  # overlap different
            ],
            "id:int,a:int,b:int,c:str",
        )
        df2 = self.to_df(
            [
                [3, 5, "xx", True],  # right unique
                [1, 4, "cd", False],  # overlap
                [2, 7, "ef", True],  # overlap different
            ],
            "id:int,b:double,c:str,d:bool",
        )
        res = compare(df1, df2, "id")
        assert res.get_row_counts() == {1: 2, 2: 1, 3: 2}
        assert len(res.get_unique_samples(1)) == 2
        assert len(res.get_unique_samples(2)) == 1
        assert len(res.get_unequal_samples()[0]) == 1
        assert len(res.get_unequal_samples()[1]) == 1
        diff = res.get_diff_summary().groupby("column")["diff"].sum().to_dict()
        assert diff == {"b": 1, "c": 0}

    def test_overlap_with_close_numbers(self) -> None:
        df1 = self.to_df(
            [
                [0, 1],
                [1, 3],
            ],
            "id:int,a:int",
        )
        df2 = self.to_df(
            [
                [0, 1.02],
                [1, 2.98],
            ],
            "id:int,a:double",
        )
        for rel_tol, abs_tol in [(0, 0), (0.005, 0.005)]:
            res = compare(df1, df2, "id", abs_tol=abs_tol, rel_tol=rel_tol)
            assert res.get_row_counts() == {3: 2}
            assert len(res.get_unequal_samples()[0]) == 2
            assert len(res.get_unequal_samples()[1]) == 2
            diff = res.get_diff_summary().groupby("column")["max_diff"].sum().to_dict()
            assert abs(diff["a"] - 0.02) < 1e-5
        res = compare(df1, df2, "id", abs_tol=0.1)
        assert res.get_row_counts() == {3: 2}
        assert len(res.get_unequal_samples()[0]) == 0
        assert len(res.get_unequal_samples()[1]) == 0
        diff = res.get_diff_summary().groupby("column")["max_diff"].sum().to_dict()
        assert abs(diff["a"] - 0.02) < 1e-5
        res = compare(df1, df2, "id", rel_tol=0.03)
        assert res.get_row_counts() == {3: 2}
        assert len(res.get_unequal_samples()[0]) == 0
        assert len(res.get_unequal_samples()[1]) == 0
        diff = res.get_diff_summary().groupby("column")["max_diff"].sum().to_dict()
        assert abs(diff["a"] - 0.02) < 1e-5


@ft.fugue_test_suite("pandas", mark_test=True)
class PandasCompareTests(CompareTests):
    pass


@ft.fugue_test_suite("duckdb", mark_test=True)
class DuckDBCompareTests(CompareTests):
    pass


try:
    import ray

    @ft.fugue_test_suite("ray", mark_test=True)
    class RayCompareTests(CompareTests):
        pass

except ImportError:
    pass
