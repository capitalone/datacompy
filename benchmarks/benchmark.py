import os  # noqa: D100

import datacompy
import pandas as pd
import polars as pl
import pytest

FOLDER = "data"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, FOLDER)


def run_polars(base, compare):  # noqa: D103
    comp = datacompy.polars.PolarsCompare(base, compare, join_columns="id")
    comp.report()
    return None


def run_pandas(base, compare):  # noqa: D103
    comp = datacompy.pandas.PandasCompare(base, compare, join_columns="id")
    comp.report()
    return None


@pytest.mark.parametrize("size", [1000, 100000])
def test_pandas(benchmark, size):  # noqa: D103
    base = pd.read_parquet(f"{DATA_DIR}/{size}/base/")
    compare = pd.read_parquet(f"{DATA_DIR}/{size}/compare/")
    benchmark.pedantic(target=run_pandas, args=[base, compare], iterations=1, rounds=5)


@pytest.mark.parametrize("size", [1000, 100000])
def test_polars(benchmark, size):  # noqa: D103
    base = pl.read_parquet(f"{DATA_DIR}/{size}/base/*.parquet")
    compare = pl.read_parquet(f"{DATA_DIR}/{size}/compare/*.parquet")
    benchmark.pedantic(target=run_polars, args=[base, compare], iterations=1, rounds=5)


if __name__ == "__main__":
    # make sure to install pytest-benchmark
    retcode = pytest.main(["benchmark.py"])
