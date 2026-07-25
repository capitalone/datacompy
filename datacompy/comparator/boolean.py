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
import polars as pl

from datacompy.comparator.base import BaseComparator


class PandasBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in Pandas."""

    def compare(
        self,
        col1: pd.Series,
        col2: pd.Series,
        **kwargs: Any,
    ) -> pd.Series | None:
        """Compare columns when either column holds Boolean values.

        Boolean comparisons are exact and null-safe. When a Boolean column is
        compared with another dtype, normal Pandas equality semantics are used;
        for example, ``True`` matches ``1`` and ``False`` matches ``0``.

        Parameters
        ----------
        col1 : pd.Series
            The first Pandas Series to compare.
        col2 : pd.Series
            The second Pandas Series to compare.
        **kwargs : Any
            Unused; accepted so this comparator matches the pipeline signature.

        Returns
        -------
        pd.Series
            A Pandas Series of booleans indicating whether the values in `col1`
            and `col2` are equal. Two nulls are treated as equal.
        None
            if the columns are not comparable.

        Notes
        -----
        Detection uses ``infer_dtype`` rather than the column ``dtype`` because
        Pandas represents Boolean data as ``object`` in common cases: a Boolean
        column containing a null, and any Boolean column that has been through
        an outer merge (which upcasts ``bool`` to ``object``).
        """
        if col1.shape != col2.shape:
            return None

        if not (
            pd.api.types.infer_dtype(col1, skipna=True) == "boolean"
            or pd.api.types.infer_dtype(col2, skipna=True) == "boolean"
        ):
            return None

        return (col1.eq(col2) | (col1.isna() & col2.isna())).fillna(False).astype(bool)


class PolarsBooleanComparator(BaseComparator):
    """Comparator for Boolean columns in Polars."""

    def compare(
        self,
        col1: pl.Series,
        col2: pl.Series,
        **kwargs: Any,
    ) -> pl.Series | None:
        """Compare columns when either Polars dtype is Boolean.

        Boolean comparisons are exact and null-safe. When a Boolean column is
        compared with another dtype, normal Polars equality semantics are used;
        for example, ``True`` matches ``1`` and ``False`` matches ``0``.

        Parameters
        ----------
        col1 : pl.Series
            The first Polars Series to compare.
        col2 : pl.Series
            The second Polars Series to compare.
        **kwargs : Any
            Unused; accepted so this comparator matches the pipeline signature.

        Returns
        -------
        pl.Series
            A Polars Series of booleans indicating whether the values in `col1`
            and `col2` are equal. Two nulls are treated as equal.
        None
            if the columns are not comparable.

        Notes
        -----
        ``eq_missing`` is used rather than ``==`` because Polars propagates
        nulls through ``==``; ``eq_missing`` treats two nulls as equal and a
        null against a value as unequal.
        """
        if col1.shape != col2.shape:
            return None

        if not (col1.dtype == pl.Boolean or col2.dtype == pl.Boolean):
            return None

        try:
            return col1.eq_missing(col2)
        except Exception:
            # Polars raises when the dtypes have no comparison supertype (a
            # Boolean against a List, Date, Struct, ...). Signal that this
            # comparator cannot handle the pair so the pipeline continues.
            return None
