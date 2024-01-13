from datacompy.fsql import FugueSQLBuilder
import pandas as pd
import numpy as np
import datetime


def generate_data(size: int = 1000) -> pd.DataFrame:
    np.random.seed(0)

    base = pd.DataFrame(
        dict(
            id=np.arange(0, size),
            a=np.random.randint(0, 10, size),
            b=np.random.rand(size),
            c=np.random.choice(["aaa", "bbb", "ccc"], size),
            d=np.random.randint(0, 10, size),
            e=np.random.rand(size),
            f=np.random.choice(["aaa", "bbb", "ccc"], size),
            g=np.random.randint(0, 10, size),
            h=np.random.rand(size),
            i=np.random.choice(["aaa", "bbb", "ccc"], size),
        )
    )

    compare = pd.DataFrame(
        dict(
            id=np.arange(0, size),
            d=np.random.randint(0, 10, size),
            e=np.random.rand(size),
            f=np.random.choice(["aaa", "bbb", "ccc"], size),
            g=np.random.randint(0, 10, size),
            h=np.random.rand(size),
            i=np.random.choice(["aaa", "bbb", "ccc"], size),
            j=np.random.randint(0, 10, size),
            k=np.random.rand(size),
            l=np.random.choice(["aaa", "bbb", "ccc"], size),
        )
    )

    return base, compare


def test_fsql_builder(shuffle_df):
    df = shuffle_df.drop_duplicates(["a"])
    # df = shuffle_df
    df1 = df
    df2 = df
    builder = FugueSQLBuilder(df1, df2, "a", dedup=True)
    print(builder.gen_fsql(sample_count=0))
    print(builder.run_duckdb())
    df1, df2 = generate_data(10000000)
    start = datetime.datetime.now()
    builder = FugueSQLBuilder(df1, df1, "id", dedup=True)
    print(builder.gen_fsql(sample_count=0))
    res = builder.run_duckdb(sample_count=10)
    end = datetime.datetime.now()
    print(res)
    print((end - start).total_seconds())
