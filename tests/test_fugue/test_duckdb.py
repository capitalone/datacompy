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
"""Test fugue functionality with duckdb."""

import pytest
from datacompy import (
    all_columns_match,
    all_rows_overlap,
    count_matching_rows,
    intersect_columns,
    is_match,
    unq_columns,
)
from ordered_set import OrderedSet
from pytest import raises

duckdb = pytest.importorskip("duckdb")


def test_is_match_duckdb(
    ref_df,
    shuffle_df,
    float_off_df,
    upper_case_df,
    space_df,
    upper_col_df,
):
    with duckdb.connect():
        rdf = duckdb.from_df(ref_df[0])

        assert is_match(rdf, shuffle_df, join_columns="a")

        assert not is_match(rdf, float_off_df, join_columns="a")
        assert not is_match(rdf, float_off_df, abs_tol=0.00001, join_columns="a")
        assert is_match(rdf, float_off_df, abs_tol=0.001, join_columns="a")
        assert is_match(rdf, float_off_df, abs_tol=0.001, join_columns="a")

        assert not is_match(rdf, upper_case_df, join_columns="a")
        assert is_match(rdf, upper_case_df, join_columns="a", ignore_case=True)

        assert not is_match(rdf, space_df, join_columns="a")
        assert is_match(rdf, space_df, join_columns="a", ignore_spaces=True)

        assert is_match(rdf, upper_col_df, join_columns="a")
        with raises(AssertionError):
            is_match(rdf, upper_col_df, join_columns="a", cast_column_names_lower=False)

        assert is_match(
            duckdb.sql("SELECT 'a' AS a, 'b' AS b"),
            duckdb.sql("SELECT 'a' AS a, 'b' AS b"),
            join_columns="a",
        )


def test_unique_columns_duckdb(ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    with duckdb.connect():
        ddf1 = duckdb.from_df(df1)
        ddf1_copy = duckdb.from_df(df1_copy)
        ddf2 = duckdb.from_df(df2)
        ddf3 = duckdb.from_df(df3)

        assert unq_columns(ddf1, ddf1_copy) == OrderedSet()
        assert unq_columns(ddf1, ddf2) == OrderedSet(["c"])
        assert unq_columns(ddf1, ddf3) == OrderedSet(["a", "b"])
        assert unq_columns(ddf1_copy, ddf1) == OrderedSet()
        assert unq_columns(ddf3, ddf2) == OrderedSet(["c"])


def test_intersect_columns_duckdb(ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    with duckdb.connect():
        ddf1 = duckdb.from_df(df1)
        ddf1_copy = duckdb.from_df(df1_copy)
        ddf2 = duckdb.from_df(df2)
        ddf3 = duckdb.from_df(df3)

        assert intersect_columns(ddf1, ddf1_copy) == OrderedSet(["a", "b", "c"])
        assert intersect_columns(ddf1, ddf2) == OrderedSet(["a", "b"])
        assert intersect_columns(ddf1, ddf3) == OrderedSet(["c"])
        assert intersect_columns(ddf1_copy, ddf1) == OrderedSet(["a", "b", "c"])
        assert intersect_columns(ddf3, ddf2) == OrderedSet()


def test_all_columns_match_duckdb(ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    with duckdb.connect():
        df1 = duckdb.from_df(df1)
        df1_copy = duckdb.from_df(df1_copy)
        df2 = duckdb.from_df(df2)
        df3 = duckdb.from_df(df3)

        assert all_columns_match(df1, df1_copy) is True
        assert all_columns_match(df1, df2) is False
        assert all_columns_match(df1, df3) is False
        assert all_columns_match(df1_copy, df1) is True
        assert all_columns_match(df3, df2) is False


def test_all_rows_overlap_duckdb(
    ref_df,
    shuffle_df,
):
    with duckdb.connect():
        rdf = duckdb.from_df(ref_df[0])
        rdf_copy = duckdb.from_df(ref_df[0].copy())
        rdf4 = duckdb.from_df(ref_df[4])
        sdf = duckdb.from_df(shuffle_df)

        assert all_rows_overlap(rdf, rdf_copy, join_columns="a")
        assert all_rows_overlap(rdf, sdf, join_columns="a")
        assert not all_rows_overlap(rdf, rdf4, join_columns="a")
        assert all_rows_overlap(
            duckdb.sql("SELECT 'a' AS a, 'b' AS b"),
            duckdb.sql("SELECT 'a' AS a, 'b' AS b"),
            join_columns="a",
        )


def test_count_matching_rows_duckdb(count_matching_rows_df):
    with duckdb.connect():
        df1 = duckdb.from_df(count_matching_rows_df[0])
        df1_copy = duckdb.from_df(count_matching_rows_df[0])
        df2 = duckdb.from_df(count_matching_rows_df[1])

        assert (
            count_matching_rows(
                df1,
                df1_copy,
                join_columns="a",
            )
            == 100
        )
        assert count_matching_rows(df1, df2, join_columns="a") == 10
        # Fugue

        assert (
            count_matching_rows(
                df1,
                df1_copy,
                join_columns="a",
                parallelism=2,
            )
            == 100
        )
        assert (
            count_matching_rows(
                df1,
                df2,
                join_columns="a",
                parallelism=2,
            )
            == 10
        )
