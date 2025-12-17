import os  # noqa: D100
import shutil

import numpy as np
import polars as pl
import pyarrow.dataset as ds


def generate_data(size=1000, folder="data"):
    """
    Generate two datasets for benchmarking.

    :param size: dataset size to generate
    """
    rng = np.random.default_rng()

    base = pl.DataFrame(
        {
            "id": np.arange(0, size),
            "a": rng.integers(0, 10, size),
            "b": rng.random(size),
            "c": rng.choice(["aaa", "bbb", "ccc"], size),
            "d": rng.integers(0, 10, size),
            "e": rng.random(size),
            "f": rng.choice(["aaa", "bbb", "ccc"], size),
            "g": rng.integers(0, 10, size),
            "h": rng.random(size),
            "i": rng.choice(["aaa", "bbb", "ccc"], size),
        }
    ).to_arrow()

    compare = pl.DataFrame(
        {
            "id": np.arange(0, size),
            "d": rng.integers(0, 10, size),
            "e": rng.random(size),
            "f": rng.choice(["aaa", "bbb", "ccc"], size),
            "g": rng.integers(0, 10, size),
            "h": rng.random(size),
            "i": rng.choice(["aaa", "bbb", "ccc"], size),
            "j": rng.integers(0, 10, size),
            "k": rng.random(size),
            "l": rng.choice(["aaa", "bbb", "ccc"], size),
        }
    ).to_arrow()

    # write out the datasets as parquet files
    ds.write_dataset(
        base,
        f"{folder}/{size}/base/",
        basename_template="{i}.parquet",
        format="parquet",
        max_rows_per_file=1_000_000,
        max_rows_per_group=1_000_000,
        existing_data_behavior="overwrite_or_ignore",
    )
    ds.write_dataset(
        compare,
        f"{folder}/{size}/compare/",
        basename_template="{i}.parquet",
        format="parquet",
        max_rows_per_file=1_000_000,
        max_rows_per_group=1_000_000,
        existing_data_behavior="overwrite_or_ignore",
    )


# typicall this is only needs to run once to generate the data
if __name__ == "__main__":
    folder = "data"
    # delete and recreate the folder
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

    generate_data(size=1000, folder=folder)
    generate_data(size=100_000, folder=folder)
    generate_data(size=10_000_000, folder=folder)
    generate_data(size=50_000_000, folder=folder)
    generate_data(size=100_000_000, folder=folder)
    generate_data(size=500_000_000, folder=folder)
