"""Time how long ``SparkSQLCompare`` takes to build its column statistics.

The comparison is constructed (which runs the merge and ``_intersect_compare``)
for wide frames, and the wall time and the number of Spark jobs it launches
are reported. Data is generated with Spark, written to parquet once, and read
back, so every run starts from files on disk like a real comparison would.

Usage::

    python benchmarks/benchmark_spark_column_stats.py --rows 1000000 --cols 10 50
    python benchmarks/benchmark_spark_column_stats.py --rows 200000 --cols 200
"""

import argparse
import os
import statistics
import tempfile
import time

import datacompy
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def emit(line):  # noqa: D103
    print(line, flush=True)  # noqa: T201


def write_frames(spark, path, rows, cols):  # noqa: D103
    base = spark.range(rows)
    data_cols = [F.rand(seed=i).alias(f"c{i}") for i in range(cols)]
    base = base.select("id", *data_cols)
    # Roughly 10% of the values in each column differ by 0.001.
    compare = base.select(
        "id",
        *[
            F.when(F.rand(seed=1000 + i) < 0.1, F.col(f"c{i}") + 0.001)
            .otherwise(F.col(f"c{i}"))
            .alias(f"c{i}")
            for i in range(cols)
        ],
    )
    base.write.mode("overwrite").parquet(os.path.join(path, "base"))
    compare.write.mode("overwrite").parquet(os.path.join(path, "compare"))


def run_once(spark, path):  # noqa: D103
    base = spark.read.parquet(os.path.join(path, "base"))
    compare = spark.read.parquet(os.path.join(path, "compare"))
    sc = spark.sparkContext
    group = f"bench-{time.time_ns()}"
    sc.setJobGroup(group, group)
    start = time.perf_counter()
    comp = datacompy.SparkSQLCompare(spark, base, compare, join_columns="id")
    elapsed = time.perf_counter() - start
    jobs = len(sc.statusTracker().getJobIdsForGroup(group))
    comp.intersect_rows.unpersist()
    return elapsed, jobs


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--cols", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("datacompy-column-stats-benchmark")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    emit(f"datacompy {datacompy.__version__}, pyspark {spark.version}")
    emit("rows,cols,median_s,min_s,max_s,spark_jobs")
    with tempfile.TemporaryDirectory() as tmp:
        for cols in args.cols:
            path = os.path.join(tmp, f"{args.rows}_{cols}")
            write_frames(spark, path, args.rows, cols)
            run_once(spark, path)  # warm up the JVM and file listing
            results = [run_once(spark, path) for _ in range(args.runs)]
            times = [r[0] for r in results]
            emit(
                f"{args.rows},{cols},{statistics.median(times):.2f},"
                f"{min(times):.2f},{max(times):.2f},{results[-1][1]}"
            )
    spark.stop()


if __name__ == "__main__":
    main()
