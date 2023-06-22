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
import fugue.api as fa
import numpy as np
import pandas as pd
import polars as pl
import pytest
from ordered_set import OrderedSet
from pytest import raises

from datacompy import Compare, intersect_columns, is_match, report, unq_columns


@pytest.fixture
def ref_df():
    np.random.seed(0)
    return pd.DataFrame(
        dict(
            a=np.random.randint(0, 10, 100),
            b=np.random.rand(100),
            c=np.random.choice(["aaa", "b_c", "csd"], 100),
        )
    )


@pytest.fixture
def shuffle_df(ref_df):
    return ref_df.sample(frac=1.0)


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
    return pd.DataFrame(dict(aa=[0, 1, 0], bb=[2.1, 3.1, 4.1]))


@pytest.fixture
def simple_diff_df2():
    return pd.DataFrame(dict(aa=[1, 0, 1], bb=[3.1, 4.1, 5.1], cc=["a", "b", "c"]))


def test_is_match_native(
    ref_df,
    shuffle_df,
    float_off_df,
    upper_case_df,
    space_df,
    upper_col_df,
):
    # defaults to Compare class
    assert is_match(ref_df, ref_df.copy(), join_columns="a")
    assert not is_match(ref_df, shuffle_df, join_columns="a")
    # Fugue
    assert is_match(ref_df, shuffle_df, join_columns="a", parallelism=2)

    assert not is_match(ref_df, float_off_df, join_columns="a", parallelism=2)
    assert not is_match(
        ref_df, float_off_df, abs_tol=0.00001, join_columns="a", parallelism=2
    )
    assert is_match(
        ref_df, float_off_df, abs_tol=0.001, join_columns="a", parallelism=2
    )
    assert is_match(
        ref_df, float_off_df, abs_tol=0.001, join_columns="a", parallelism=2
    )

    assert not is_match(ref_df, upper_case_df, join_columns="a", parallelism=2)
    assert is_match(
        ref_df, upper_case_df, join_columns="a", ignore_case=True, parallelism=2
    )

    assert not is_match(ref_df, space_df, join_columns="a", parallelism=2)
    assert is_match(
        ref_df, space_df, join_columns="a", ignore_spaces=True, parallelism=2
    )

    assert is_match(ref_df, upper_col_df, join_columns="a", parallelism=2)

    with raises(AssertionError):
        is_match(
            ref_df,
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
    ref_df.iteritems = ref_df.items  # pandas 2 compatibility
    rdf = spark_session.createDataFrame(ref_df)

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
    rdf = pl.from_pandas(ref_df)

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
        rdf = duckdb.from_df(ref_df)

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


def test_report_pandas(simple_diff_df1, simple_diff_df2):
    comp = Compare(simple_diff_df1, simple_diff_df2, join_columns=["aa"])
    a = report(simple_diff_df1, simple_diff_df2, ["aa"])
    assert a == comp.report()
    a = report(simple_diff_df1, simple_diff_df2, "aa", parallelism=2)
    assert a == comp.report()


def test_report_spark(spark_session, simple_diff_df1, simple_diff_df2):
    simple_diff_df1.iteritems = simple_diff_df1.items  # pandas 2 compatibility
    simple_diff_df2.iteritems = simple_diff_df2.items  # pandas 2 compatibility
    df1 = spark_session.createDataFrame(simple_diff_df1)
    df2 = spark_session.createDataFrame(simple_diff_df2)
    comp = Compare(simple_diff_df1, simple_diff_df2, join_columns="aa")
    a = report(df1, df2, ["aa"])
    assert a == comp.report()


def test_unique_columns_native(ref_df):
    df1 = ref_df
    df2 = ref_df.copy().drop(columns=["c"])
    df3 = ref_df.copy().drop(columns=["a", "b"])

    assert unq_columns(df1, df1.copy()) == OrderedSet()
    assert unq_columns(df1, df2) == OrderedSet(["c"])
    assert unq_columns(df1, df3) == OrderedSet(["a", "b"])
    assert unq_columns(df1.copy(), df1) == OrderedSet()
    assert unq_columns(df3, df2) == OrderedSet(["c"])


def test_unique_columns_spark(spark_session, ref_df):
    df1 = ref_df
    df1_copy = ref_df.copy()
    df2 = ref_df.copy().drop(columns=["c"])
    df3 = ref_df.copy().drop(columns=["a", "b"])

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
    df1 = ref_df
    df2 = ref_df.copy().drop(columns=["c"])
    df3 = ref_df.copy().drop(columns=["a", "b"])

    pdf1 = pl.from_pandas(df1)
    pdf1_copy = pl.from_pandas(df1.copy())
    pdf2 = pl.from_pandas(df2)
    pdf3 = pl.from_pandas(df3)

    assert unq_columns(pdf1, pdf1_copy) == OrderedSet()
    assert unq_columns(pdf1, pdf2) == OrderedSet(["c"])
    assert unq_columns(pdf1, pdf3) == OrderedSet(["a", "b"])
    assert unq_columns(pdf1_copy, pdf1) == OrderedSet()
    assert unq_columns(pdf3, pdf2) == OrderedSet(["c"])


def test_unique_columns_duckdb(ref_df):
    df1 = ref_df
    df2 = ref_df.copy().drop(columns=["c"])
    df3 = ref_df.copy().drop(columns=["a", "b"])

    with duckdb.connect():
        ddf1 = duckdb.from_df(df1)
        ddf1_copy = duckdb.from_df(df1.copy())
        ddf2 = duckdb.from_df(df2)
        ddf3 = duckdb.from_df(df3)

        assert unq_columns(ddf1, ddf1_copy) == OrderedSet()
        assert unq_columns(ddf1, ddf2) == OrderedSet(["c"])
        assert unq_columns(ddf1, ddf3) == OrderedSet(["a", "b"])
        assert unq_columns(ddf1_copy, ddf1) == OrderedSet()
        assert unq_columns(ddf3, ddf2) == OrderedSet(["c"])


def test_intersect_columns_native(ref_df):
    df1 = ref_df
    df2 = ref_df.copy().drop(columns=["c"])
    df3 = ref_df.copy().drop(columns=["a", "b"])

    assert intersect_columns(df1, df1.copy()) == OrderedSet(["a", "b", "c"])
    assert intersect_columns(df1, df2) == OrderedSet(["a", "b"])
    assert intersect_columns(df1, df3) == OrderedSet(["c"])
    assert intersect_columns(df1.copy(), df1) == OrderedSet(["a", "b", "c"])
    assert intersect_columns(df3, df2) == OrderedSet()


def test_intersect_columns_spark(spark_session, ref_df):
    df1 = ref_df
    df1_copy = ref_df.copy()
    df2 = ref_df.copy().drop(columns=["c"])
    df3 = ref_df.copy().drop(columns=["a", "b"])

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
    df1 = ref_df
    df2 = ref_df.copy().drop(columns=["c"])
    df3 = ref_df.copy().drop(columns=["a", "b"])

    pdf1 = pl.from_pandas(df1)
    pdf1_copy = pl.from_pandas(df1.copy())
    pdf2 = pl.from_pandas(df2)
    pdf3 = pl.from_pandas(df3)

    assert intersect_columns(pdf1, pdf1_copy) == OrderedSet(["a", "b", "c"])
    assert intersect_columns(pdf1, pdf2) == OrderedSet(["a", "b"])
    assert intersect_columns(pdf1, pdf3) == OrderedSet(["c"])
    assert intersect_columns(pdf1_copy, pdf1) == OrderedSet(["a", "b", "c"])
    assert intersect_columns(pdf3, pdf2) == OrderedSet()


def test_intersect_columns_duckdb(ref_df):
    df1 = ref_df
    df2 = ref_df.copy().drop(columns=["c"])
    df3 = ref_df.copy().drop(columns=["a", "b"])

    with duckdb.connect():
        ddf1 = duckdb.from_df(df1)
        ddf1_copy = duckdb.from_df(df1.copy())
        ddf2 = duckdb.from_df(df2)
        ddf3 = duckdb.from_df(df3)

        assert intersect_columns(ddf1, ddf1_copy) == OrderedSet(["a", "b", "c"])
        assert intersect_columns(ddf1, ddf2) == OrderedSet(["a", "b"])
        assert intersect_columns(ddf1, ddf3) == OrderedSet(["c"])
        assert intersect_columns(ddf1_copy, ddf1) == OrderedSet(["a", "b", "c"])
        assert intersect_columns(ddf3, ddf2) == OrderedSet()
