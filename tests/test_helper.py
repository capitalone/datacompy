"""Test the helper functions with all supported DataFrame types."""

import logging
import sys

import pandas as pd
import pytest
from datacompy import Compare
from datacompy.helper import analyze_join_columns, get_recommended_join_columns

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# =========================================================================
# Basic pandas tests
# =========================================================================


def test_analyze_join_columns():
    df1 = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
    df2 = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "f"]})

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712

    df3 = pd.DataFrame({"id": [1, 1, 1, 1, 1], "name": ["a", "b", "c", "d", "e"]})
    df4 = pd.DataFrame({"id": [1, 1, 1, 1, 1], "name": ["a", "b", "c", "d", "f"]})

    stats = analyze_join_columns(df3, df4)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == False  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712


def test_analyze_join_columns_with_nulls():
    df1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
            "with_nulls": [
                None,
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
            ],
        }
    )
    df2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l"],
            "with_nulls": [
                "a",
                None,
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "l",
            ],
        }
    )

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}
    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712
    assert stats_by_column["with_nulls"]["recommended"] == False  # noqa: E712

    stats = analyze_join_columns(df1, df2, allow_nulls=True)
    stats_by_column = {stat["column"]: stat for stat in stats}
    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712
    assert stats_by_column["with_nulls"]["recommended"] == True  # noqa: E712


def test_analyze_join_columns_with_threshold():
    df1 = pd.DataFrame({"id": [1, 2, 3, 1, 2], "name": ["a", "b", "c", "d", "e"]})
    df2 = pd.DataFrame({"id": [1, 2, 3, 1, 4], "name": ["a", "b", "c", "d", "f"]})

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}
    assert stats_by_column["id"]["recommended"] == False  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712

    stats = analyze_join_columns(df1, df2, uniqueness_threshold=50)
    stats_by_column = {stat["column"]: stat for stat in stats}
    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712


def test_analyze_join_columns_unique_in_only_one_df():
    df1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],  # 100% unique
            "value": [10, 20, 30, 40, 50],
        }
    )
    df2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 1, 2],  # 60% unique, below the default 90% threshold
            "value": [11, 21, 31, 12, 22],
        }
    )

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == False  # noqa: E712

    # Using a lower threshold should recommend it
    stats = analyze_join_columns(df1, df2, uniqueness_threshold=50)
    stats_by_column = {stat["column"]: stat for stat in stats}
    assert stats_by_column["id"]["recommended"] == True  # noqa: E712


def test_get_recommended_join_columns():
    df1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
            "common": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "unique_60": [1, 2, 3, 1, 2, 5, 6, 7, 8, 9, 10],
            "with_nulls": [None, "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        }
    )
    df2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l"],
            "common": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "unique_60": [1, 2, 3, 1, 2, 5, 6, 7, 8, 9, 10],
            "with_nulls": ["a", None, "c", "d", "e", "f", "g", "h", "i", "j", "l"],
        }
    )

    recommended = get_recommended_join_columns(df1, df2)
    assert "id" in recommended
    assert "name" in recommended
    assert "common" not in recommended
    assert "with_nulls" not in recommended

    recommended = get_recommended_join_columns(df1, df2, allow_nulls=True)
    assert "id" in recommended
    assert "name" in recommended
    assert "common" not in recommended
    assert "with_nulls" in recommended
    assert "unique_60" not in recommended

    recommended = get_recommended_join_columns(df1, df2, uniqueness_threshold=50)
    assert "id" in recommended
    assert "name" in recommended
    assert "unique_60" in recommended
    assert "common" not in recommended
    assert "with_nulls" not in recommended


def test_get_recommended_join_columns_directly_in_compare():
    df1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "value1": [10, 20, 30, 40, 50],
        }
    )
    df2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 6],
            "name": ["a", "b", "c", "d", "f"],
            "value2": [11, 21, 31, 41, 61],
        }
    )

    compare = Compare(df1, df2, join_columns=get_recommended_join_columns(df1, df2))

    assert compare.df1_unq_rows is not None
    assert compare.df2_unq_rows is not None
    assert len(compare.df1_unq_rows) == 1
    assert len(compare.df2_unq_rows) == 1


# =========================================================================
# Polars tests
# =========================================================================

def test_analyze_join_columns_polars():
    # Using try/except instead of importorskip to avoid skipping tests
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars is not installed")

    # Convert pandas DataFrames to polars
    df1_pd = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
    df2_pd = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "f"]})

    df1 = pl.from_pandas(df1_pd)
    df2 = pl.from_pandas(df2_pd)

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712

    df3_pd = pd.DataFrame({"id": [1, 1, 1, 1, 1], "name": ["a", "b", "c", "d", "e"]})
    df4_pd = pd.DataFrame({"id": [1, 1, 1, 1, 1], "name": ["a", "b", "c", "d", "f"]})

    df3 = pl.from_pandas(df3_pd)
    df4 = pl.from_pandas(df4_pd)

    stats = analyze_join_columns(df3, df4)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == False  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712


def test_analyze_join_columns_with_nulls_polars():
    # Using try/except instead of importorskip to avoid skipping tests
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars is not installed")

    df1_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        "with_nulls": [None, "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    })
    df2_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l"],
        "with_nulls": ["a", None, "c", "d", "e", "f", "g", "h", "i", "j", "l"]
    })

    df1 = pl.from_pandas(df1_pd)
    df2 = pl.from_pandas(df2_pd)

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712
    assert stats_by_column["with_nulls"]["recommended"] == False  # noqa: E712

    stats = analyze_join_columns(df1, df2, allow_nulls=True)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712
    assert stats_by_column["with_nulls"]["recommended"] == True  # noqa: E712


def test_get_recommended_join_columns_polars():
    # Using try/except instead of importorskip to avoid skipping tests
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars is not installed")

    df1_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        "common": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "with_nulls": [None, "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    })
    df2_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l"],
        "common": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "with_nulls": ["a", None, "c", "d", "e", "f", "g", "h", "i", "j", "l"]
    })

    df1 = pl.from_pandas(df1_pd)
    df2 = pl.from_pandas(df2_pd)

    recommended = get_recommended_join_columns(df1, df2)
    assert "id" in recommended
    assert "name" in recommended
    assert "common" not in recommended
    assert "with_nulls" not in recommended

    recommended = get_recommended_join_columns(df1, df2, allow_nulls=True)
    assert "id" in recommended
    assert "name" in recommended
    assert "common" not in recommended
    assert "with_nulls" in recommended


# =========================================================================
# Spark tests
# =========================================================================

# Helper function to check if a module is installed
def is_module_installed(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not is_module_installed("pyspark"),
    reason="PySpark is not installed"
)
@pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason="unsupported python version"
)
def test_analyze_join_columns_spark(spark_session):
    # Skip check removed since it's now at the decorator level

    # Convert pandas DataFrames to Spark
    df1_pd = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
    df2_pd = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "f"]})

    df1 = spark_session.createDataFrame(df1_pd)
    df2 = spark_session.createDataFrame(df2_pd)

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712

    df3_pd = pd.DataFrame({"id": [1, 1, 1, 1, 1], "name": ["a", "b", "c", "d", "e"]})
    df4_pd = pd.DataFrame({"id": [1, 1, 1, 1, 1], "name": ["a", "b", "c", "d", "f"]})

    df3 = spark_session.createDataFrame(df3_pd)
    df4 = spark_session.createDataFrame(df4_pd)

    stats = analyze_join_columns(df3, df4)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == False  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712


@pytest.mark.skipif(
    not is_module_installed("pyspark"),
    reason="PySpark is not installed"
)
@pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason="unsupported python version"
)
def test_analyze_join_columns_with_nulls_spark(spark_session):
    # Skip check removed since it's now at the decorator level

    # Need 11 rows to achieve >90% uniqueness with nulls
    df1_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        "with_nulls": [None, "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    })
    df2_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l"],
        "with_nulls": ["a", None, "c", "d", "e", "f", "g", "h", "i", "j", "l"]
    })

    df1 = spark_session.createDataFrame(df1_pd)
    df2 = spark_session.createDataFrame(df2_pd)

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712
    assert stats_by_column["with_nulls"]["recommended"] == False  # noqa: E712

    stats = analyze_join_columns(df1, df2, allow_nulls=True)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712
    assert stats_by_column["with_nulls"]["recommended"] == True  # noqa: E712


@pytest.mark.skipif(
    not is_module_installed("pyspark"),
    reason="PySpark is not installed"
)
@pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason="unsupported python version"
)
def test_get_recommended_join_columns_spark(spark_session):
    df1_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        "common": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "with_nulls": [None, "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    })
    df2_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l"],
        "common": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "with_nulls": ["a", None, "c", "d", "e", "f", "g", "h", "i", "j", "l"]
    })

    df1 = spark_session.createDataFrame(df1_pd)
    df2 = spark_session.createDataFrame(df2_pd)

    recommended = get_recommended_join_columns(df1, df2)
    assert "id" in recommended
    assert "name" in recommended
    assert "common" not in recommended
    assert "with_nulls" not in recommended

    recommended = get_recommended_join_columns(df1, df2, allow_nulls=True)
    assert "id" in recommended
    assert "name" in recommended
    assert "common" not in recommended
    assert "with_nulls" in recommended


# =========================================================================
# Snowflake tests
# =========================================================================

@pytest.mark.skipif(
    not is_module_installed("snowflake.snowpark"),
    reason="Snowflake is not installed"
)
def test_analyze_join_columns_snowflake(snowpark_session):
    # Convert pandas DataFrames to Snowflake
    df1_pd = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
    df2_pd = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "f"]})

    df1 = snowpark_session.createDataFrame(df1_pd)
    df2 = snowpark_session.createDataFrame(df2_pd)

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}

    # In Snowflake, column names are uppercase by default
    assert stats_by_column["ID"]["recommended"] == True  # noqa: E712
    assert stats_by_column["NAME"]["recommended"] == True  # noqa: E712

    df3_pd = pd.DataFrame({"id": [1, 1, 1, 1, 1], "name": ["a", "b", "c", "d", "e"]})
    df4_pd = pd.DataFrame({"id": [1, 1, 1, 1, 1], "name": ["a", "b", "c", "d", "f"]})

    df3 = snowpark_session.createDataFrame(df3_pd)
    df4 = snowpark_session.createDataFrame(df4_pd)

    stats = analyze_join_columns(df3, df4)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["ID"]["recommended"] == False  # noqa: E712
    assert stats_by_column["NAME"]["recommended"] == True  # noqa: E712


@pytest.mark.skipif(
    not is_module_installed("snowflake.snowpark"),
    reason="Snowflake is not installed"
)
def test_analyze_join_columns_with_nulls_snowflake(snowpark_session):

    df1_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        "with_nulls": [None, "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    })
    df2_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l"],
        "with_nulls": ["a", None, "c", "d", "e", "f", "g", "h", "i", "j", "l"]
    })

    df1 = snowpark_session.createDataFrame(df1_pd)
    df2 = snowpark_session.createDataFrame(df2_pd)

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["ID"]["recommended"] == True  # noqa: E712
    assert stats_by_column["NAME"]["recommended"] == True  # noqa: E712
    assert stats_by_column["WITH_NULLS"]["recommended"] == False  # noqa: E712

    stats = analyze_join_columns(df1, df2, allow_nulls=True)
    stats_by_column = {stat["column"]: stat for stat in stats}

    assert stats_by_column["ID"]["recommended"] == True  # noqa: E712
    assert stats_by_column["NAME"]["recommended"] == True  # noqa: E712
    assert stats_by_column["WITH_NULLS"]["recommended"] == True  # noqa: E712


@pytest.mark.skipif(
    not is_module_installed("snowflake.snowpark"),
    reason="Snowflake is not installed"
)
def test_get_recommended_join_columns_snowflake(snowpark_session):
    # Skip check removed since it's now at the decorator level

    df1_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        "common": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "with_nulls": [None, "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    })
    df2_pd = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l"],
        "common": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "with_nulls": ["a", None, "c", "d", "e", "f", "g", "h", "i", "j", "l"]
    })

    df1 = snowpark_session.createDataFrame(df1_pd)
    df2 = snowpark_session.createDataFrame(df2_pd)

    recommended = get_recommended_join_columns(df1, df2)
    assert "ID" in recommended
    assert "NAME" in recommended
    assert "COMMON" not in recommended
    assert "WITH_NULLS" not in recommended

    recommended = get_recommended_join_columns(df1, df2, allow_nulls=True)
    assert "ID" in recommended
    assert "NAME" in recommended
    assert "COMMON" not in recommended
    assert "WITH_NULLS" in recommended
