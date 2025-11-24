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

# Optional dependencies initialization
ps = None
sp = None

try:
    import pyspark as ps
except ImportError:
    pass

try:
    import snowflake.snowpark as sp
except ImportError:
    pass


def get_spark_column_dtypes(
    dataframe: "ps.sql.DataFrame", col_1: str, col_2: str
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
    dataframe: "sp.DataFrame", col_1: str, col_2: str
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
    base_dtype = next(
        d[1] for d in dataframe.dtypes if d[0].strip('"').upper() == col_1.upper()
    )
    compare_dtype = next(
        d[1] for d in dataframe.dtypes if d[0].strip('"').upper() == col_2.upper()
    )
    return base_dtype, compare_dtype
