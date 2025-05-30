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

"""
Compare two Pandas DataFrames.

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from ordered_set import OrderedSet

LOG = logging.getLogger(__name__)


class BaseCompare(ABC):
    """Base comparison class."""

    @property
    def df1(self) -> Any:
        """Get the first dataframe."""
        return self._df1  # type: ignore

    @df1.setter
    @abstractmethod
    def df1(self, df1: Any) -> None:
        """Check that it is a dataframe and has the join columns."""
        pass

    @property
    def df2(self) -> Any:
        """Get the second dataframe."""
        return self._df2  # type: ignore

    @df2.setter
    @abstractmethod
    def df2(self, df2: Any) -> None:
        """Check that it is a dataframe and has the join columns."""
        pass

    @abstractmethod
    def _validate_dataframe(
        self, index: str, cast_column_names_lower: bool = True
    ) -> None:
        """Check that it is a dataframe and has the join columns."""
        pass

    @abstractmethod
    def _compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        """Run the comparison.

        This tries to run df1.equals(df2)
        first so that if they're truly equal we can tell.

        This method will log out information about what is different between
        the two dataframes, and will also return a boolean.
        """
        pass

    @abstractmethod
    def df1_unq_columns(self) -> OrderedSet[str]:
        """Get columns that are unique to df1."""
        pass

    @abstractmethod
    def df2_unq_columns(self) -> OrderedSet[str]:
        """Get columns that are unique to df2."""
        pass

    @abstractmethod
    def intersect_columns(self) -> OrderedSet[str]:
        """Get columns that are shared between the two dataframes."""
        pass

    @abstractmethod
    def _dataframe_merge(self, ignore_spaces: bool) -> None:
        """Merge df1 to df2 on the join columns.

        To get df1 - df2, df2 - df1
        and df1 & df2.

        If ``on_index`` is True, this will join on index values, otherwise it
        will join on the ``join_columns``.
        """
        pass

    @abstractmethod
    def _intersect_compare(self, ignore_spaces: bool, ignore_case: bool) -> None:
        """Compare the intersection of the two dataframes."""
        pass

    @abstractmethod
    def all_columns_match(self) -> bool:
        """Check if all columns match."""
        pass

    @abstractmethod
    def all_rows_overlap(self) -> bool:
        """Check if all rows overlap."""
        pass

    @abstractmethod
    def count_matching_rows(self) -> int:
        """Count the number of matching rows."""
        pass

    @abstractmethod
    def intersect_rows_match(self) -> bool:
        """Check if the intersection of rows match."""
        pass

    @abstractmethod
    def matches(self, ignore_extra_columns: bool = False) -> bool:
        """Check if the dataframes match."""
        pass

    @abstractmethod
    def subset(self) -> bool:
        """Check if one dataframe is a subset of the other."""
        pass

    @abstractmethod
    def sample_mismatch(
        self, column: str, sample_count: int = 10, for_display: bool = False
    ) -> Any:
        """Get a sample of rows that mismatch."""
        pass

    @abstractmethod
    def all_mismatch(self, ignore_matching_cols: bool = False) -> Any:
        """Get all rows that mismatch."""
        pass

    @abstractmethod
    def report(
        self,
        sample_count: int = 10,
        column_count: int = 10,
        html_file: str | None = None,
    ) -> str:
        """Return a string representation of a report."""
        pass

    def only_join_columns(self) -> bool:
        """Boolean on if the only columns are the join columns."""
        return set(self.join_columns) == set(self.df1.columns) == set(self.df2.columns)


def temp_column_name(*dataframes) -> str:
    """Get a temp column name that isn't included in columns of any dataframes.

    Parameters
    ----------
    dataframes : list of DataFrames
        The DataFrames to create a temporary column name for

    Returns
    -------
    str
        String column name that looks like '_temp_x' for some integer x
    """
    i = 0
    columns = []
    for dataframe in dataframes:
        columns = columns + list(dataframe.columns)
    columns = set(columns)

    while True:
        temp_column = f"_temp_{i}"
        unique = True

        if temp_column in columns:
            i += 1
            unique = False
        if unique:
            return temp_column
