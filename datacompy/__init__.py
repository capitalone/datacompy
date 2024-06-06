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

__version__ = "0.13.0"

import platform
from warnings import warn

from .core import *  # noqa: F403
from .fugue import (  # noqa: F401
    all_columns_match,
    all_rows_overlap,
    count_matching_rows,
    intersect_columns,
    is_match,
    report,
    unq_columns,
)
from .polars import PolarsCompare  # noqa: F401
from .spark.pandas import SparkPandasCompare  # noqa: F401
from .spark.sql import SparkSQLCompare  # noqa: F401

major = platform.python_version_tuple()[0]
minor = platform.python_version_tuple()[1]

if major == "3" and minor >= "12":
    warn(
        "Python 3.12 and above currently is not supported by Spark and Ray. "
        "Please note that some functionality will not work and currently is not supported.",
        UserWarning,
        stacklevel=2,
    )
