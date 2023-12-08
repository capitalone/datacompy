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
"""Test the fugue functionality with pandas."""
from io import StringIO
import pandas as pd
from ordered_set import OrderedSet
from pytest import raises

from datacompy import (
    Compare,
    all_columns_match,
    all_rows_overlap,
    intersect_columns,
    is_match,
    report,
    unq_columns,
)

from test_fugue_helpers import _compare_report


def test_is_match_native(
    ref_df,
    shuffle_df,
    float_off_df,
    upper_case_df,
    space_df,
    upper_col_df,
):
    # defaults to Compare class
    assert is_match(ref_df[0], ref_df[0].copy(), join_columns="a")
    assert not is_match(ref_df[0], shuffle_df, join_columns="a")
    # Fugue
    assert is_match(ref_df[0], shuffle_df, join_columns="a", parallelism=2)

    assert not is_match(ref_df[0], float_off_df, join_columns="a", parallelism=2)
    assert not is_match(
        ref_df[0], float_off_df, abs_tol=0.00001, join_columns="a", parallelism=2
    )
    assert is_match(
        ref_df[0], float_off_df, abs_tol=0.001, join_columns="a", parallelism=2
    )
    assert is_match(
        ref_df[0], float_off_df, abs_tol=0.001, join_columns="a", parallelism=2
    )

    assert not is_match(ref_df[0], upper_case_df, join_columns="a", parallelism=2)
    assert is_match(
        ref_df[0], upper_case_df, join_columns="a", ignore_case=True, parallelism=2
    )

    assert not is_match(ref_df[0], space_df, join_columns="a", parallelism=2)
    assert is_match(
        ref_df[0], space_df, join_columns="a", ignore_spaces=True, parallelism=2
    )

    assert is_match(ref_df[0], upper_col_df, join_columns="a", parallelism=2)

    with raises(AssertionError):
        is_match(
            ref_df[0],
            upper_col_df,
            join_columns="a",
            cast_column_names_lower=False,
            parallelism=2,
        )


def test_doc_case():
    data1 = """acct_id,dollar_amt,name,float_fld,date_fld
    10000001234,123.45,George Maharis,14530.1555,2017-01-01
    10000001235,0.45,Michael Bluth,1,2017-01-01
    10000001236,1345,George Bluth,,2017-01-01
    10000001237,123456,Bob Loblaw,345.12,2017-01-01
    10000001239,1.05,Lucille Bluth,,2017-01-01
    """

    data2 = """acct_id,dollar_amt,name,float_fld
    10000001234,123.4,George Michael Bluth,14530.155
    10000001235,0.45,Michael Bluth,
    10000001236,1345,George Bluth,1
    10000001237,123456,Robert Loblaw,345.12
    10000001238,1.05,Loose Seal Bluth,111
    """

    df1 = pd.read_csv(StringIO(data1))
    df2 = pd.read_csv(StringIO(data2))

    assert not is_match(
        df1,
        df2,
        join_columns="acct_id",
        abs_tol=0,
        rel_tol=0,
        df1_name="Original",
        df2_name="New",
        parallelism=2,
    )


def test_report_pandas(
    simple_diff_df1,
    simple_diff_df2,
    no_intersection_diff_df1,
    no_intersection_diff_df2,
    large_diff_df1,
    large_diff_df2,
):
    comp = Compare(simple_diff_df1, simple_diff_df2, join_columns=["aa"])
    a = report(simple_diff_df1, simple_diff_df2, ["aa"])
    _compare_report(comp.report(), a)
    a = report(simple_diff_df1, simple_diff_df2, "aa", parallelism=2)
    _compare_report(comp.report(), a)

    comp = Compare(
        no_intersection_diff_df1, no_intersection_diff_df2, join_columns=["x"]
    )
    a = report(no_intersection_diff_df1, no_intersection_diff_df2, ["x"])
    _compare_report(comp.report(), a)
    a = report(no_intersection_diff_df1, no_intersection_diff_df2, "x", parallelism=2)
    _compare_report(comp.report(), a)

    # due to https://github.com/capitalone/datacompy/issues/221
    # we can have y as a constant to ensure all the x matches are equal

    comp = Compare(large_diff_df1, large_diff_df2, join_columns=["x"])
    a = report(large_diff_df1, large_diff_df2, ["x"])
    _compare_report(comp.report(), a, truncate=True)
    a = report(large_diff_df1, large_diff_df2, "x", parallelism=2)
    _compare_report(comp.report(), a, truncate=True)


def test_unique_columns_native(ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    assert unq_columns(df1, df1.copy()) == OrderedSet()
    assert unq_columns(df1, df2) == OrderedSet(["c"])
    assert unq_columns(df1, df3) == OrderedSet(["a", "b"])
    assert unq_columns(df1.copy(), df1) == OrderedSet()
    assert unq_columns(df3, df2) == OrderedSet(["c"])


def test_intersect_columns_native(ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    assert intersect_columns(df1, df1_copy) == OrderedSet(["a", "b", "c"])
    assert intersect_columns(df1, df2) == OrderedSet(["a", "b"])
    assert intersect_columns(df1, df3) == OrderedSet(["c"])
    assert intersect_columns(df1_copy, df1) == OrderedSet(["a", "b", "c"])
    assert intersect_columns(df3, df2) == OrderedSet()


def test_all_columns_match_native(ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    assert all_columns_match(df1, df1_copy) is True
    assert all_columns_match(df1, df2) is False
    assert all_columns_match(df1, df3) is False
    assert all_columns_match(df1_copy, df1) is True
    assert all_columns_match(df3, df2) is False


def test_all_rows_overlap_native(
    ref_df,
    shuffle_df,
):
    # defaults to Compare class
    assert all_rows_overlap(ref_df[0], ref_df[0].copy(), join_columns="a")
    assert all_rows_overlap(ref_df[0], shuffle_df, join_columns="a")
    assert not all_rows_overlap(ref_df[0], ref_df[4], join_columns="a")
    # Fugue
    assert all_rows_overlap(ref_df[0], shuffle_df, join_columns="a", parallelism=2)
    assert not all_rows_overlap(ref_df[0], ref_df[4], join_columns="a", parallelism=2)
