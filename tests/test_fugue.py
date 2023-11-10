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
Testing out the fugue is_match functionality
"""
from io import StringIO

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest
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


@pytest.fixture
def ref_df():
    np.random.seed(0)

    df1 = pd.DataFrame(
        dict(
            a=np.random.randint(0, 10, 100),
            b=np.random.rand(100),
            c=np.random.choice(["aaa", "b_c", "csd"], 100),
        )
    )
    df1_copy = df1.copy()
    df2 = df1.copy().drop(columns=["c"])
    df3 = df1.copy().drop(columns=["a", "b"])
    df4 = pd.DataFrame(
        dict(
            a=np.random.randint(1, 12, 100),  # shift the join col
            b=np.random.rand(100),
            c=np.random.choice(["aaa", "b_c", "csd"], 100),
        )
    )
    return [df1, df1_copy, df2, df3, df4]


@pytest.fixture
def shuffle_df(ref_df):
    return ref_df[0].sample(frac=1.0)


@pytest.fixture
def float_off_df(shuffle_df):
    return shuffle_df.assign(b=shuffle_df.b + 0.0001)


@pytest.fixture
def upper_case_df(shuffle_df):
    return shuffle_df.assign(c=shuffle_df.c.str.upper())


@pytest.fixture
def space_df(shuffle_df):
    return shuffle_df.assign(c=shuffle_df.c + " ")


@pytest.fixture
def upper_col_df(shuffle_df):
    return shuffle_df.rename(columns={"a": "A"})


@pytest.fixture
def simple_diff_df1():
    return pd.DataFrame(dict(aa=[0, 1, 0], bb=[2.1, 3.1, 4.1])).convert_dtypes()


@pytest.fixture
def simple_diff_df2():
    return pd.DataFrame(
        dict(aa=[1, 0, 1], bb=[3.1, 4.1, 5.1], cc=["a", "b", "c"])
    ).convert_dtypes()


@pytest.fixture
def no_intersection_diff_df1():
    np.random.seed(0)
    return pd.DataFrame(dict(x=["a"], y=[0.1])).convert_dtypes()


@pytest.fixture
def no_intersection_diff_df2():
    return pd.DataFrame(dict(x=["b"], y=[1.1])).convert_dtypes()


@pytest.fixture
def large_diff_df1():
    np.random.seed(0)
    data = np.random.randint(0, 7, size=10000)
    return pd.DataFrame({"x": data, "y": np.array([9] * 10000)}).convert_dtypes()


@pytest.fixture
def large_diff_df2():
    np.random.seed(0)
    data = np.random.randint(6, 11, size=10000)
    return pd.DataFrame({"x": data, "y": np.array([9] * 10000)}).convert_dtypes()


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


def _compare_report(expected, actual, truncate=False):
    if truncate:
        expected = expected.split("Sample Rows", 1)[0]
        actual = actual.split("Sample Rows", 1)[0]
    assert expected == actual


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
