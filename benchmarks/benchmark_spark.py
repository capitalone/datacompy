import os  # noqa: D100

import datacompy
import pytest
from pyspark.sql import SparkSession

FOLDER = "data"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, FOLDER)


def run_spark(spark_session, base, compare):  # noqa: D103
    comp = datacompy.spark.SparkSQLCompare(
        spark_session, base, compare, join_columns="id"
    )
    comp.report()
    return None


@pytest.mark.parametrize("size", [1000, 100000])
def test_spark(benchmark, size):  # noqa: D103
    spark = (
        SparkSession.builder.appName("datacompy-benchmark")
        .config("spark.sql.ansi.enabled", "false")
        .enableHiveSupport()
        .getOrCreate()
    )
    base = spark.read.parquet(f"{DATA_DIR}/{size}/base/")
    compare = spark.read.parquet(f"{DATA_DIR}/{size}/compare/")
    benchmark.pedantic(
        target=run_spark, args=[spark, base, compare], iterations=1, rounds=5
    )


if __name__ == "__main__":
    # make sure to install pytest-benchmark
    retcode = pytest.main(["benchmark_spark.py"])
