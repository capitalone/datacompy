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

__version__ = "0.11.1"

from datacompy.core import *
from datacompy.fugue import (
    all_columns_match,
    all_rows_overlap,
    intersect_columns,
    is_match,
    report,
    unq_columns,
)
from datacompy.polars import PolarsCompare
from datacompy.spark import NUMERIC_SPARK_TYPES, SparkCompare
