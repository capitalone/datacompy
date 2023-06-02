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
from pytest import raises

from datacompy import Compare, is_match, report


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
    df1 = spark_session.createDataFrame(simple_diff_df1)
    df2 = spark_session.createDataFrame(simple_diff_df2)
    comp = Compare(simple_diff_df1, simple_diff_df2, join_columns="aa")
    a = report(df1, df2, ["aa"])
    assert a == comp.report()
