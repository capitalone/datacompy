#
# Copyright 2026 Capital One Services, LLC
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

"""Boolean comparator classes."""

from typing import Any

import pandas as pd

from datacompy.comparator.base import BaseComparator


class PandasBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in Pandas."""

    def compare(
        self,
        col1: pd.Series,
        col2: pd.Series,
        **kwargs: Any,
    ) -> pd.Series | None:
        """Compare columns when either Pandas dtype is Boolean.

        Boolean comparisons are exact and null-safe. When a Boolean column is
        compared with another dtype, normal Pandas equality semantics are used;
        for example, ``True`` matches ``1`` and ``False`` matches ``0``.
        """
        if not (
            pd.api.types.is_bool_dtype(col1.dtype)
            or pd.api.types.is_bool_dtype(col2.dtype)
        ):
            return None

        if col1.shape != col2.shape:
            return None

        return (col1.eq(col2) | (col1.isna() & col2.isna())).fillna(False).astype(bool)
