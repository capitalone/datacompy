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
"""Test fugue and spark."""
import pytest
from datacompy import (
    Compare,
    all_columns_match,
    all_rows_overlap,
    intersect_columns,
    is_match,
    report,
    unq_columns,
)
from ordered_set import OrderedSet
from pytest import raises

from test_fugue_helpers import _compare_report

pyspark = pytest.importorskip("pyspark")


def test_is_match_spark(
    spark_session,
    ref_df,
    shuffle_df,
    float_off_df,
    upper_case_df,
    space_df,
    upper_col_df,
):
    ref_df[0].iteritems = ref_df[0].items  # pandas 2 compatibility
    rdf = spark_session.createDataFrame(ref_df[0])

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
        spark_session.sql("SELECT 'a' AS a, 'b' AS b"),
        spark_session.sql("SELECT 'a' AS a, 'b' AS b"),
        join_columns="a",
    )


def test_report_spark(
    spark_session,
    simple_diff_df1,
    simple_diff_df2,
    no_intersection_diff_df1,
    no_intersection_diff_df2,
    large_diff_df1,
    large_diff_df2,
):
    simple_diff_df1.iteritems = simple_diff_df1.items  # pandas 2 compatibility
    simple_diff_df2.iteritems = simple_diff_df2.items  # pandas 2 compatibility
    no_intersection_diff_df1.iteritems = (
        no_intersection_diff_df1.items
    )  # pandas 2 compatibility
    no_intersection_diff_df2.iteritems = (
        no_intersection_diff_df2.items
    )  # pandas 2 compatibility
    large_diff_df1.iteritems = large_diff_df1.items  # pandas 2 compatibility
    large_diff_df2.iteritems = large_diff_df2.items  # pandas 2 compatibility

    df1 = spark_session.createDataFrame(simple_diff_df1)
    df2 = spark_session.createDataFrame(simple_diff_df2)
    comp = Compare(simple_diff_df1, simple_diff_df2, join_columns="aa")
    a = report(df1, df2, ["aa"])
    _compare_report(comp.report(), a)

    df1 = spark_session.createDataFrame(no_intersection_diff_df1)
    df2 = spark_session.createDataFrame(no_intersection_diff_df2)
    comp = Compare(no_intersection_diff_df1, no_intersection_diff_df2, join_columns="x")
    a = report(df1, df2, ["x"])
    _compare_report(comp.report(), a)

    # due to https://github.com/capitalone/datacompy/issues/221
    # we can have y as a constant to ensure all the x matches are equal

    df1 = spark_session.createDataFrame(large_diff_df1)
    df2 = spark_session.createDataFrame(large_diff_df2)
    comp = Compare(large_diff_df1, large_diff_df2, join_columns="x")
    a = report(df1, df2, ["x"])
    _compare_report(comp.report(), a, truncate=True)


def test_unique_columns_spark(spark_session, ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    df1.iteritems = df1.items  # pandas 2 compatibility
    df1_copy.iteritems = df1_copy.items  # pandas 2 compatibility
    df2.iteritems = df2.items  # pandas 2 compatibility
    df3.iteritems = df3.items  # pandas 2 compatibility

    sdf1 = spark_session.createDataFrame(df1)
    sdf1_copy = spark_session.createDataFrame(df1_copy)
    sdf2 = spark_session.createDataFrame(df2)
    sdf3 = spark_session.createDataFrame(df3)

    assert unq_columns(sdf1, sdf1_copy) == OrderedSet()
    assert unq_columns(sdf1, sdf2) == OrderedSet(["c"])
    assert unq_columns(sdf1, sdf3) == OrderedSet(["a", "b"])
    assert unq_columns(sdf1_copy, sdf1) == OrderedSet()
    assert unq_columns(sdf3, sdf2) == OrderedSet(["c"])


def test_intersect_columns_spark(spark_session, ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    df1.iteritems = df1.items  # pandas 2 compatibility
    df1_copy.iteritems = df1_copy.items  # pandas 2 compatibility
    df2.iteritems = df2.items  # pandas 2 compatibility
    df3.iteritems = df3.items  # pandas 2 compatibility

    sdf1 = spark_session.createDataFrame(df1)
    sdf1_copy = spark_session.createDataFrame(df1_copy)
    sdf2 = spark_session.createDataFrame(df2)
    sdf3 = spark_session.createDataFrame(df3)

    assert intersect_columns(sdf1, sdf1_copy) == OrderedSet(["a", "b", "c"])
    assert intersect_columns(sdf1, sdf2) == OrderedSet(["a", "b"])
    assert intersect_columns(sdf1, sdf3) == OrderedSet(["c"])
    assert intersect_columns(sdf1_copy, sdf1) == OrderedSet(["a", "b", "c"])
    assert intersect_columns(sdf3, sdf2) == OrderedSet()


def test_all_columns_match_spark(spark_session, ref_df):
    df1 = ref_df[0]
    df1_copy = ref_df[1]
    df2 = ref_df[2]
    df3 = ref_df[3]

    df1.iteritems = df1.items  # pandas 2 compatibility
    df1_copy.iteritems = df1_copy.items  # pandas 2 compatibility
    df2.iteritems = df2.items  # pandas 2 compatibility
    df3.iteritems = df3.items  # pandas 2 compatibility

    df1 = spark_session.createDataFrame(df1)
    df1_copy = spark_session.createDataFrame(df1_copy)
    df2 = spark_session.createDataFrame(df2)
    df3 = spark_session.createDataFrame(df3)

    assert all_columns_match(df1, df1_copy) is True
    assert all_columns_match(df1, df2) is False
    assert all_columns_match(df1, df3) is False
    assert all_columns_match(df1_copy, df1) is True
    assert all_columns_match(df3, df2) is False


def test_all_rows_overlap_spark(
    spark_session,
    ref_df,
    shuffle_df,
):
    ref_df[0].iteritems = ref_df[0].items  # pandas 2 compatibility
    ref_df[4].iteritems = ref_df[4].items  # pandas 2 compatibility
    shuffle_df.iteritems = shuffle_df.items  # pandas 2 compatibility
    rdf = spark_session.createDataFrame(ref_df[0])
    rdf_copy = spark_session.createDataFrame(ref_df[0])
    rdf4 = spark_session.createDataFrame(ref_df[4])
    sdf = spark_session.createDataFrame(shuffle_df)

    assert all_rows_overlap(rdf, rdf_copy, join_columns="a")
    assert all_rows_overlap(rdf, sdf, join_columns="a")
    assert not all_rows_overlap(rdf, rdf4, join_columns="a")
    assert all_rows_overlap(
        spark_session.sql("SELECT 'a' AS a, 'b' AS b"),
        spark_session.sql("SELECT 'a' AS a, 'b' AS b"),
        join_columns="a",
    )
