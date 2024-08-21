#
# Copyright 2024 Capital One Services, LLC
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
"""DataComPy is a package to compare two Pandas DataFrames.

Originally started to be something of a replacement for SAS's PROC COMPARE for Pandas DataFrames with some more functionality than just Pandas.DataFrame.equals(Pandas.DataFrame) (in that it prints out some stats, and lets you tweak how accurate matches have to be).
Then extended to carry that functionality over to Spark Dataframes.
"""

__version__ = "0.14.1"

import platform
from warnings import warn

from datacompy.base import BaseCompare, temp_column_name
from datacompy.core import (
    Compare,
    calculate_max_diff,
    columns_equal,
    compare_string_and_date_columns,
    generate_id_within_group,
    get_merged_columns,
    render,
)
from datacompy.fugue import (
    all_columns_match,
    all_rows_overlap,
    count_matching_rows,
    intersect_columns,
    is_match,
    report,
    unq_columns,
)
from datacompy.polars import PolarsCompare
from datacompy.spark.sql import SparkSQLCompare

__all__ = [
    "BaseCompare",
    "Compare",
    "PolarsCompare",
    "SparkSQLCompare",
    "all_columns_match",
    "all_rows_overlap",
    "calculate_max_diff",
    "columns_equal",
    "compare_string_and_date_columns",
    "count_matching_rows",
    "generate_id_within_group",
    "get_merged_columns",
    "intersect_columns",
    "is_match",
    "render",
    "report",
    "temp_column_name",
    "unq_columns",
]

major = platform.python_version_tuple()[0]
minor = platform.python_version_tuple()[1]

if major == "3" and minor >= "12":
    warn(
        "Python 3.12 and above currently is not supported by Spark and Ray. "
        "Please note that some functionality will not work and currently is not supported.",
        UserWarning,
        stacklevel=2,
    )

# numpy 2.0 check
from numpy import __version__ as np_version

if np_version.split(".")[0] >= "2":
    warn(
        "SparkPandasCompare currently only supports Numpy < 2."
        "Please note that the SparkPandasCompare functionality will not work and currently is not supported.",
        UserWarning,
        stacklevel=2,
    )
else:
    from datacompy.spark.pandas import SparkPandasCompare  # noqa: F401

    __all__.append("SparkPandasCompare")
