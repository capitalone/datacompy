import logging
from pyspark.sql import SparkSession
from datacompy.spark import SparkSQLCompare

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_spark(app_name="DataFrameComparison"):
    """
    Initialize Spark session.
    """
    try:
        logger.info("Starting Spark session")
        spark = SparkSession.builder.appName(app_name).getOrCreate()
        return spark
    except Exception as e:
        logger.error(f"Error initializing Spark session: {e}")
        raise

def create_sample_data(spark):
    """
    Create sample dataframes for comparison.
    """
    try:
        logger.info("Creating sample dataframes")
        data_1 = [(1, "Alice"), (2, "Bob"), (3, "Charlie")]
        data_2 = [(1, "Alice"), (2, "Bob"), (4, "David")]

        df1 = spark.createDataFrame(data_1, ["ID", "Name"])
        df2 = spark.createDataFrame(data_2, ["ID", "Name"])

        return df1, df2
    except Exception as e:
        logger.error(f"Error creating dataframes: {e}")
        raise

def compare_dataframes(df1, df2, join_columns):
    """
    Compare two Spark DataFrames using SparkSQLCompare.
    """
    try:
        logger.info(f"Comparing dataframes on columns: {join_columns}")
        compare = SparkSQLCompare(df1, df2, join_columns=join_columns)
        result = compare.report()
        logger.info("Comparison completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        raise

def main():
    # Step 1: Initialize Spark session
    spark = initialize_spark()

    # Step 2: Create sample data
    df1, df2 = create_sample_data(spark)

    # Step 3: Perform comparison
    comparison_result = compare_dataframes(df1, df2, join_columns=["ID"])

    # Step 4: Display result
    print(comparison_result)

    # Step 5: Stop Spark session
    spark.stop()
    logger.info("Spark session stopped")

if __name__ == "__main__":
    main()
