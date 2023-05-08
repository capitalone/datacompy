import numpy as np
import pandas as pd
import pytest
import fugue.api as fa
from datacompy import is_match


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
    assert not is_match(
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
    assert not is_match(
        rdf, upper_col_df, join_columns="a", cast_column_names_lower=False
    )
