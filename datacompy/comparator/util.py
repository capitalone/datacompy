#
# Copyright 2025 Capital One Services, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility and helper functions for data comparison."""

import polars as pl

from datacompy.comparator._optional_imports import (
    ps,
    sp,
)


# function which checks 2 polars series are of the same shape and of the same type
def validate_polars_series(col1: pl.Series, col2: pl.Series, type_check: str) -> bool:
    """Validate two Polars Series objects based on their shape, data type, and a specified type check.

    Parameters
    ----------
    col1 : pl.Series
        The first Polars Series to validate.
    col2 : pl.Series
        The second Polars Series to validate.
    type_check : str
        The type check to perform. Currently supports "numeric" to validate
        if both series are numeric.

    Returns
    -------
    bool
        True if the series pass the validation based on the specified type check,
        otherwise False.
    """
    if type_check == "numeric":
        return (
            col1.shape == col2.shape
            and col1.dtype == col2.dtype
            and pl.Series.is_numeric(col1)
        )
    return False


def get_spark_column_dtypes(
    dataframe: ps.sql.DataFrame, col_1: str, col_2: str
) -> tuple[str, str]:
    """Get the dtypes of two columns.

    Parameters
    ----------
    dataframe: pyspark.sql.DataFrame
        DataFrame to do comparison on
    col_1 : str
        The first column to look at
    col_2 : str
        The second column

    Returns
    -------
    tuple(str, str)
        Tuple of base and compare datatype
    """
    base_dtype = next(d[1] for d in dataframe.dtypes if d[0].upper() == col_1.upper())
    compare_dtype = next(
        d[1] for d in dataframe.dtypes if d[0].upper() == col_2.upper()
    )
    return base_dtype, compare_dtype


def get_snowflake_column_dtypes(
    dataframe: sp.DataFrame, col_1: str, col_2: str
) -> tuple[str, str]:
    """Get the dtypes of two columns.

    Parameters
    ----------
    dataframe: sp.DataFrame
        DataFrame to do comparison on
    col_1 : str
        The first column to look at
    col_2 : str
        The second column

    Returns
    -------
    Tuple(str, str)
        Tuple of base and compare datatype
    """
    df_raw_dtypes = [
        (name, field.datatype)
        for name, field in zip(
            dataframe.schema.names, dataframe.schema.fields, strict=False
        )
    ]
    base_dtype = next(d[1] for d in df_raw_dtypes if d[0].upper() == col_1.upper())
    compare_dtype = next(d[1] for d in df_raw_dtypes if d[0].upper() == col_2.upper())
    return base_dtype, compare_dtype
