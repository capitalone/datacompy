"""Generate benchmark datasets of varying shapes using Polars LazyFrame and sink_parquet."""

import logging
import os
import shutil
import string

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def generate_columns(num_columns: int, size: int, seed: int = 42) -> dict:
    """
    Generate a dictionary of column data for benchmarking.

    Parameters
    ----------
    num_columns : int
        Number of data columns to generate (excluding id column).
    size : int
        Number of rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary mapping column names to data arrays.
    """
    rng = np.random.default_rng(seed)
    columns = {"id": np.arange(0, size)}

    # Generate column names using alphabet letters, then aa, ab, etc.
    col_names = list(string.ascii_lowercase)
    for first in string.ascii_lowercase:
        for second in string.ascii_lowercase:
            col_names.append(first + second)

    for i in range(num_columns):
        col_name = col_names[i]
        col_type = i % 3  # Cycle through int, float, string
        if col_type == 0:
            columns[col_name] = rng.integers(0, 10, size)
        elif col_type == 1:
            columns[col_name] = rng.random(size)
        else:
            columns[col_name] = rng.choice(["aaa", "bbb", "ccc"], size)

    return columns


def generate_data(
    size: int = 1000,
    folder: str = "data",
    num_base_columns: int = 9,
    num_compare_columns: int = 9,
    overlap_columns: int = 6,
) -> None:
    """
    Generate two datasets for benchmarking using Polars LazyFrame.

    Parameters
    ----------
    size : int
        Dataset size (number of rows) to generate.
    folder : str
        Output folder for parquet files.
    num_base_columns : int
        Number of data columns for base dataset (excluding id).
    num_compare_columns : int
        Number of data columns for compare dataset (excluding id).
    overlap_columns : int
        Number of columns that overlap between base and compare datasets.
        These will be the last N columns of base and first N columns of compare.
    """
    # Ensure overlap doesn't exceed either dataset's column count
    overlap_columns = min(overlap_columns, num_base_columns, num_compare_columns)

    # Generate base columns
    base_cols = generate_columns(num_base_columns, size, seed=42)

    # Generate compare columns with overlapping names for the first overlap_columns
    # The overlap columns share names but have different data
    compare_cols = {"id": np.arange(0, size)}
    rng = np.random.default_rng(123)  # Different seed for compare data

    col_names = list(string.ascii_lowercase)
    for first in string.ascii_lowercase:
        for second in string.ascii_lowercase:
            col_names.append(first + second)

    # Start column naming from where overlap begins in base
    base_col_names = list(base_cols.keys())[1:]  # Exclude 'id'
    overlap_start = num_base_columns - overlap_columns
    overlap_col_names = base_col_names[overlap_start:]

    # Add overlapping columns (same names, different data)
    for i, col_name in enumerate(overlap_col_names):
        col_type = (overlap_start + i) % 3
        if col_type == 0:
            compare_cols[col_name] = rng.integers(0, 10, size)
        elif col_type == 1:
            compare_cols[col_name] = rng.random(size)
        else:
            compare_cols[col_name] = rng.choice(["aaa", "bbb", "ccc"], size)

    # Add unique compare columns
    unique_compare_cols = num_compare_columns - overlap_columns
    next_col_idx = num_base_columns  # Start after base columns
    for i in range(unique_compare_cols):
        col_name = col_names[next_col_idx + i]
        col_type = (next_col_idx + i) % 3
        if col_type == 0:
            compare_cols[col_name] = rng.integers(0, 10, size)
        elif col_type == 1:
            compare_cols[col_name] = rng.random(size)
        else:
            compare_cols[col_name] = rng.choice(["aaa", "bbb", "ccc"], size)

    # Create output directories
    base_path = f"{folder}/{size}_{num_base_columns}cols/base"
    compare_path = f"{folder}/{size}_{num_base_columns}cols/compare"
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(compare_path, exist_ok=True)

    # Use LazyFrame and sink_parquet to write data
    base_lf = pl.LazyFrame(base_cols)
    compare_lf = pl.LazyFrame(compare_cols)

    sinks = []
    sinks.append(
        base_lf.sink_parquet(
            f"{base_path}/data.parquet", row_group_size=1_000_000, lazy=True
        )
    )
    sinks.append(
        compare_lf.sink_parquet(
            f"{compare_path}/data.parquet", row_group_size=1_000_000, lazy=True
        )
    )
    pl.collect_all(sinks)


# typically this is only needs to run once to generate the data
if __name__ == "__main__":
    logger.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    folder = "data"
    # delete and recreate the folder
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

    # Generate datasets with different sizes and column counts
    sizes = [1000, 100_000, 10_000_000]
    column_configs = [
        {"num_base_columns": 9, "num_compare_columns": 9, "overlap_columns": 6},
        {"num_base_columns": 50, "num_compare_columns": 50, "overlap_columns": 40},
        {"num_base_columns": 100, "num_compare_columns": 100, "overlap_columns": 80},
    ]

    for size in sizes:
        for config in column_configs:
            logger.info(
                f"Generating data: size={size}, columns={config['num_base_columns']}"
            )
            generate_data(size=size, folder=folder, **config)
