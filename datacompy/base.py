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

"""
Compare two Pandas DataFrames

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from ordered_set import OrderedSet

LOG = logging.getLogger(__name__)


class BaseCompare(ABC):
    @property
    def df1(self) -> Any:
        return self._df1  # type: ignore

    @df1.setter
    @abstractmethod
    def df1(self, df1: Any) -> None:
        """Check that it is a dataframe and has the join columns"""
        pass

    @property
    def df2(self) -> Any:
        return self._df2  # type: ignore

    @df2.setter
    @abstractmethod
    def df2(self, df2: Any) -> None:
        """Check that it is a dataframe and has the join columns"""
        pass

    @abstractmethod
    def _validate_dataframe(
        self, index: str, cast_column_names_lower: bool = True
    ) -> None:
        """Check that it is a dataframe and has the join columns"""
        pass

    @abstractmethod
    def _compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        """Actually run the comparison.  This tries to run df1.equals(df2)
        first so that if they're truly equal we can tell.

        This method will log out information about what is different between
        the two dataframes, and will also return a boolean.
        """
        pass

    @abstractmethod
    def df1_unq_columns(self) -> OrderedSet[str]:
        """Get columns that are unique to df1"""
        pass

    @abstractmethod
    def df2_unq_columns(self) -> OrderedSet[str]:
        """Get columns that are unique to df2"""
        pass

    @abstractmethod
    def intersect_columns(self) -> OrderedSet[str]:
        """Get columns that are shared between the two dataframes"""
        pass

    @abstractmethod
    def _dataframe_merge(self, ignore_spaces: bool) -> None:
        """Merge df1 to df2 on the join columns, to get df1 - df2, df2 - df1
        and df1 & df2

        If ``on_index`` is True, this will join on index values, otherwise it
        will join on the ``join_columns``.
        """
        pass

    @abstractmethod
    def _intersect_compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        pass

    @abstractmethod
    def all_columns_match(self) -> bool:
        pass

    @abstractmethod
    def all_rows_overlap(self) -> bool:
        pass

    @abstractmethod
    def count_matching_rows(self) -> int:
        pass

    @abstractmethod
    def intersect_rows_match(self) -> bool:
        pass

    @abstractmethod
    def matches(self, ignore_extra_columns: bool = False) -> bool:
        pass

    @abstractmethod
    def subset(self) -> bool:
        pass

    @abstractmethod
    def sample_mismatch(
        self, column: str, sample_count: int = 10, for_display: bool = False
    ) -> Any:
        pass

    @abstractmethod
    def all_mismatch(self, ignore_matching_cols: bool = False) -> Any:
        pass

    @abstractmethod
    def report(
        self,
        sample_count: int = 10,
        column_count: int = 10,
        html_file: Optional[str] = None,
    ) -> str:
        pass
