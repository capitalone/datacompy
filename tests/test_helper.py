"""Test the helper functions."""

import pandas as pd
from datacompy import Compare
from datacompy.helper import analyze_join_columns, get_recommended_join_columns


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
    # Need 11 rows to achieve >90% uniqueness with nulls
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
            ],  # 10 unique values in 11 rows = 90.9% uniqueness
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
            ],  # 10 unique values in 11 rows = 90.9% uniqueness
        }
    )

    stats = analyze_join_columns(df1, df2)
    stats_by_column = {stat["column"]: stat for stat in stats}
    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712
    assert stats_by_column["with_nulls"]["recommended"] == False  # noqa: E712  # Has nulls, so not recommended

    stats = analyze_join_columns(df1, df2, allow_nulls=True)
    stats_by_column = {stat["column"]: stat for stat in stats}
    assert stats_by_column["id"]["recommended"] == True  # noqa: E712
    assert stats_by_column["name"]["recommended"] == True  # noqa: E712
    assert stats_by_column["with_nulls"]["recommended"] == True  # noqa: E712  # Allowing nulls, so it's recommended


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
