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

from datacompy.core import *  # noqa: F403
from datacompy.fugue import all_columns_match  # noqa: F401
from datacompy.fugue import all_rows_overlap  # noqa: F401
from datacompy.fugue import count_matching_rows  # noqa: F401
from datacompy.fugue import intersect_columns  # noqa: F401
from datacompy.fugue import is_match  # noqa: F401
from datacompy.fugue import report  # noqa: F401
from datacompy.fugue import unq_columns  # noqa: F401
from datacompy.polars import PolarsCompare  # noqa: F401
from datacompy.spark import SparkCompare  # noqa: F401
from datacompy.vspark import VSparkCompare  # noqa: F401

major = platform.python_version_tuple()[0]
minor = platform.python_version_tuple()[1]

if major == "3" and minor >= "12":
    warn(
        "Python 3.12 and above currently is not supported by Spark and Ray. "
        "Please note that some functionality will not work and currently is not supported.",
        UserWarning,
        stacklevel=2,
    )
