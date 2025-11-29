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

"""Base Comparator Class."""

from abc import ABC, abstractmethod
from typing import Any


class BaseComparator(ABC):
    """Base class for all comparators.

    This class serves as an abstract base class for implementing
    specific comparator logic in derived classes.
    """

    @abstractmethod
    def compare(self, col1: Any, col2: Any, **kwargs) -> Any:
        """Check if two columns are equal.

        This method should be implemented in derived classes to provide
        specific comparison logic.

        Parameters
        ----------
        col1 : Any
            The first column to compare.
        col2 : Any
            The second column to compare.
        **kwargs : Any
            Additional keyword arguments for comparison.

        Returns
        -------
        Any
            Comparison result. (implementation-specific)
        """
        raise NotImplementedError()
