import pytest
import numpy as np
import pandas as pd


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
