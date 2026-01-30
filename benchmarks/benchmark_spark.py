import os  # noqa: D100

import datacompy
import pytest
from pyspark.sql import SparkSession

FOLDER = "data"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, FOLDER)

# Define test configurations: (size, num_columns)
TEST_CONFIGS = [
    (1000, 9),
    (1000, 50),
    (100000, 9),
    (100000, 50),
]


def run_spark(spark_session, base, compare):  # noqa: D103
    comp = datacompy.spark.SparkSQLCompare(
        spark_session, base, compare, join_columns="id"
    )
    comp.report()
    return None


@pytest.mark.parametrize("size,num_cols", TEST_CONFIGS)
def test_spark(benchmark, size, num_cols):  # noqa: D103
    spark = (
        SparkSession.builder.appName("datacompy-benchmark")
        .config("spark.sql.ansi.enabled", "false")
        .enableHiveSupport()
        .getOrCreate()
    )
    data_path = f"{DATA_DIR}/{size}_{num_cols}cols"
    base = spark.read.parquet(f"{data_path}/base/")
    compare = spark.read.parquet(f"{data_path}/compare/")
    benchmark.pedantic(
        target=run_spark, args=[spark, base, compare], iterations=1, rounds=5
    )


if __name__ == "__main__":
    # make sure to install pytest-benchmark
    retcode = pytest.main(["benchmark_spark.py"])
