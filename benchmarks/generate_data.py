"""Generate benchmark datasets of varying shapes using Polars LazyFrame and sink_parquet."""

import logging
import os
import shutil
import string
from itertools import product

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def write_partitioned_parquet(
    df: pl.DataFrame,
    output_path: str,
    num_partitions: int = 20,
) -> None:
    """
    Write a DataFrame to multiple parquet files, similar to Spark partitioning.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to write.
    output_path : str
        The output directory path.
    num_partitions : int
        Number of parquet files to create.
    """
    total_rows = df.height
    rows_per_partition = max(1, total_rows // num_partitions)

    for i in range(num_partitions):
        start_idx = i * rows_per_partition
        # Last partition gets remaining rows
        if i == num_partitions - 1:
            end_idx = total_rows
        else:
            end_idx = start_idx + rows_per_partition

        if start_idx >= total_rows:
            break

        partition_df = df.slice(start_idx, end_idx - start_idx)
        partition_file = f"{output_path}/part-{i:05d}.parquet"
        partition_df.write_parquet(partition_file)


def generate_base_and_compare_columns(
    size: int, num_base_columns, num_compare_columns, overlap_columns
) -> tuple[dict, dict]:
    """
    Generate a tuple containing two dictionaries: one for base columns and one for compare columns.

    Parameters
    ----------
    size : int
        Number of rows for each column.
    num_base_columns : int
        Number of data columns for base dataset (excluding id).
    num_compare_columns : int
        Number of data columns for compare dataset (excluding id).
    overlap_columns: int
        Number of columns that overlap between base and compare datasets.

    Returns
    -------
    tuple
        A tuple containing:
        - dict: Containing column names and their generator functions for the base dataset.
        - dict: Containing column names and their generator functions for the compare dataset.
    """
    # Ensure overlap doesn't exceed either dataset's column count
    overlap_columns = min(overlap_columns, num_base_columns, num_compare_columns)
    unique_compare_columns = num_compare_columns - overlap_columns

    rng = np.random.default_rng(42)
    base_columns = {"id": np.arange(0, size)}

    # Generate column names using alphabet letters, then aa, ab, etc.
    column_names = [
        f"{x}{y}" for x, y in product(list(string.ascii_lowercase), repeat=2)
    ]

    generators = [
        lambda: rng.integers(0, 10, size),
        lambda: rng.random(size),
        lambda: rng.choice(["aaa", "bbb", "ccc"], size),
    ]
    base_columns.update(
        {column_names[i]: generators[i % 3]() for i in range(num_base_columns)}
    )

    # Generate compare columns with overlapping names for the first overlap_columns
    # The overlap columns share names but have different data
    compare_columns = {"id": np.arange(0, size)}
    rng = np.random.default_rng(123)  # Different seed for compare data
    compare_generators = [
        lambda: rng.integers(0, 10, size),
        lambda: rng.random(size),
        lambda: rng.choice(["aaa", "bbb", "ccc"], size),
    ]

    # Start column naming from where overlap begins in base
    base_column_names = list(base_columns.keys())[1:]  # Exclude 'id'
    overlap_start = num_base_columns - overlap_columns
    overlap_column_names = base_column_names[overlap_start:]

    # Create overlapping columns (same names, different data)
    compare_columns.update(
        {
            column_name: compare_generators[(overlap_start + i) % 3]()
            for i, column_name in enumerate(overlap_column_names)
        }
    )

    # Add unique compare columns
    compare_columns.update(
        {
            column_names[num_base_columns + i]: compare_generators[
                (num_base_columns + i) % 3
            ]()
            for i in range(unique_compare_columns)
        }
    )

    assert len(base_columns) == len(compare_columns)
    assert set(base_columns.keys()) & set(compare_columns.keys()) == set(
        overlap_column_names
    ) ^ {"id"}
    assert (
        len(set(base_columns.keys()) & set(compare_columns.keys()))
        == overlap_columns + 1  # +1 for 'id'
    )

    return base_columns, compare_columns


def generate_data(
    size: int = 1000,
    folder: str = "data",
    num_base_columns: int = 9,
    num_compare_columns: int = 9,
    overlap_columns: int = 6,
    num_partitions: int = 20,
) -> None:
    """
    Generate two datasets for benchmarking using Polars.

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
    num_partitions : int
        Number of parquet files to create per dataset (similar to Spark partitions).
    """
    base_columns, compare_columns = generate_base_and_compare_columns(
        size, num_base_columns, num_compare_columns, overlap_columns
    )

    base_path = f"{folder}/{size}_{num_base_columns}cols/base"
    compare_path = f"{folder}/{size}_{num_base_columns}cols/compare"
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(compare_path, exist_ok=True)

    # Collect and write parquet files with row_group_size for chunking
    base_df = pl.DataFrame(base_columns)
    compare_df = pl.DataFrame(compare_columns)

    # Adjust partitions for small datasets
    effective_partitions = min(num_partitions, size)

    write_partitioned_parquet(base_df, base_path, effective_partitions)
    write_partitioned_parquet(compare_df, compare_path, effective_partitions)


# typically this is only needs to run once to generate the data
if __name__ == "__main__":
    logging.basicConfig(
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
