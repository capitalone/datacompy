from snowflake.snowpark.functions import trim
from snowflake.snowpark.mock import ColumnEmulator, ColumnType, patch
from snowflake.snowpark.types import StringType


@patch(trim)
def mock_trim(column: ColumnEmulator, trim_string=None) -> ColumnEmulator:
    ret_column = ColumnEmulator(
        data=[row.strip() if row is not None else row for row in column]
    )
    ret_column.sf_type = ColumnType(StringType(), True)
    return ret_column


@patch(abs)
def mock_abs(column: ColumnEmulator) -> ColumnEmulator:
    ret_column = ColumnEmulator(
        data=[abs(row) if row is not None else row for row in column]
    )
    ret_column.sf_type = column.sf_type
    return ret_column
