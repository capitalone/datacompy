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

__version__ = "1.0.0a3"

from datacompy.base import BaseCompare
from datacompy.pandas import PandasCompare
from datacompy.polars import PolarsCompare

__all__ = [
    "BaseCompare",
    "PandasCompare",
    "PolarsCompare",
]

try:
    from datacompy.snowflake import SnowflakeCompare  # noqa: F401

    __all__.append("SnowflakeCompare")
except ImportError:
    pass

try:
    from datacompy.spark import SparkSQLCompare  # noqa: F401

    __all__.append("SparkSQLCompare")
except ImportError:
    pass
