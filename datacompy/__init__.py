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
"""DataComPy is a package to compare two Pandas DataFrames.

Originally started to be something of a replacement for SAS's PROC COMPARE for Pandas DataFrames with some more functionality than just Pandas.DataFrame.equals(Pandas.DataFrame) (in that it prints out some stats, and lets you tweak how accurate matches have to be).
Then extended to carry that functionality over to Spark Dataframes.
"""

__version__ = "1.0.0"

import platform
from warnings import warn

from datacompy.base import BaseCompare
from datacompy.pandas import PandasCompare
from datacompy.polars import PolarsCompare
from datacompy.snowflake import SnowflakeCompare
from datacompy.spark import SparkSQLCompare

__all__ = [
    "BaseCompare",
    "PandasCompare",
    "PolarsCompare",
    "SnowflakeCompare",
    "SparkSQLCompare",
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
