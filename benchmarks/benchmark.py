import os  # noqa: D100

import datacompy
import pandas as pd
import polars as pl
import pytest

FOLDER = "data"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, FOLDER)

# Define test configurations: (size, num_columns)
TEST_CONFIGS = [
    (1000, 9),
    (1000, 50),
    (1000, 100),
    (100000, 9),
    (100000, 50),
    (100000, 100),
]


def run_polars(base, compare):  # noqa: D103
    comp = datacompy.polars.PolarsCompare(base, compare, join_columns="id")
    comp.report()
    return None


def run_pandas(base, compare):  # noqa: D103
    comp = datacompy.pandas.PandasCompare(base, compare, join_columns="id")
    comp.report()
    return None


@pytest.mark.parametrize("size,num_cols", TEST_CONFIGS)
def test_pandas(benchmark, size, num_cols):  # noqa: D103
    data_path = f"{DATA_DIR}/{size}_{num_cols}cols"
    base = pd.read_parquet(f"{data_path}/base/")
    compare = pd.read_parquet(f"{data_path}/compare/")
    benchmark.pedantic(target=run_pandas, args=[base, compare], iterations=1, rounds=5)


@pytest.mark.parametrize("size,num_cols", TEST_CONFIGS)
def test_polars(benchmark, size, num_cols):  # noqa: D103
    data_path = f"{DATA_DIR}/{size}_{num_cols}cols"
    base = pl.read_parquet(f"{data_path}/base/*.parquet")
    compare = pl.read_parquet(f"{data_path}/compare/*.parquet")
    benchmark.pedantic(target=run_polars, args=[base, compare], iterations=1, rounds=5)
