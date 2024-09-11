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
"""Test fugue and polars."""

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

pl = pytest.importorskip("polars")


def test_is_match_polars(
    ref_df,
    shuffle_df,
    float_off_df,
    upper_case_df,
    space_df,
    upper_col_df,
):
    rdf = pl.from_pandas(ref_df[0])

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


def test_unique_columns_polars(ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    pdf1 = pl.from_pandas(df1)
    pdf1_copy = pl.from_pandas(df1_copy)
    pdf2 = pl.from_pandas(df2)
    pdf3 = pl.from_pandas(df3)

    assert unq_columns(pdf1, pdf1_copy) == OrderedSet()
    assert unq_columns(pdf1, pdf2) == OrderedSet(["c"])
    assert unq_columns(pdf1, pdf3) == OrderedSet(["a", "b"])
    assert unq_columns(pdf1_copy, pdf1) == OrderedSet()
    assert unq_columns(pdf3, pdf2) == OrderedSet(["c"])


def test_intersect_columns_polars(ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    pdf1 = pl.from_pandas(df1)
    pdf1_copy = pl.from_pandas(df1_copy)
    pdf2 = pl.from_pandas(df2)
    pdf3 = pl.from_pandas(df3)

    assert intersect_columns(pdf1, pdf1_copy) == OrderedSet(["a", "b", "c"])
    assert intersect_columns(pdf1, pdf2) == OrderedSet(["a", "b"])
    assert intersect_columns(pdf1, pdf3) == OrderedSet(["c"])
    assert intersect_columns(pdf1_copy, pdf1) == OrderedSet(["a", "b", "c"])
    assert intersect_columns(pdf3, pdf2) == OrderedSet()


def test_all_columns_match_polars(ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    df1 = pl.from_pandas(df1)
    df1_copy = pl.from_pandas(df1_copy)
    df2 = pl.from_pandas(df2)
    df3 = pl.from_pandas(df3)

    assert all_columns_match(df1, df1_copy) is True
    assert all_columns_match(df1, df2) is False
    assert all_columns_match(df1, df3) is False
    assert all_columns_match(df1_copy, df1) is True
    assert all_columns_match(df3, df2) is False


def test_all_rows_overlap_polars(
    ref_df,
    shuffle_df,
):
    rdf = pl.from_pandas(ref_df[0])
    rdf_copy = pl.from_pandas(ref_df[0].copy())
    rdf4 = pl.from_pandas(ref_df[4])
    sdf = pl.from_pandas(shuffle_df)

    assert all_rows_overlap(rdf, rdf_copy, join_columns="a")
    assert all_rows_overlap(rdf, sdf, join_columns="a")
    assert not all_rows_overlap(rdf, rdf4, join_columns="a")


def test_count_matching_rows_polars(count_matching_rows_df):
    df1 = pl.from_pandas(count_matching_rows_df[0])
    df2 = pl.from_pandas(count_matching_rows_df[1])
    assert (
        count_matching_rows(
            df1,
            df1.clone(),
            join_columns="a",
        )
        == 100
    )
    assert count_matching_rows(df1, df2, join_columns="a") == 10
    # Fugue

    assert (
        count_matching_rows(
            df1,
            df1.clone(),
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
