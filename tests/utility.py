from string import ascii_letters

from pandas.testing import assert_series_equal

VALID_COLUMN_CHARS = ascii_letters + "_"


def assert_columns_equal(columns, actual, expected):
    actual = spark_to_pandas(actual)
    expected = spark_to_pandas(expected)
    for col in columns:
        assert_series_equal(actual[col], expected[col], check_dtype=False)


def spark_to_pandas(dataframe):
    try:
        pandas_df = dataframe.toPandas()
        return pandas_df
    except AttributeError:
        raise ValueError("Must be a Spark DataFrame")
