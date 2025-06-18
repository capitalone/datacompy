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
    def compare(self) -> Any:
        """Check if two columns are equal."""
        raise NotImplementedError()


class BaseStringComparator(BaseComparator):
    """Base class for all string comparators.

    This class serves as an abstract base class for implementing
    specific string comparator logic in derived classes.
    """

    def __init__(self, ignore_space: bool = True, ignore_case: bool = True):
        self.ignore_space = ignore_space
        self.ignore_case = ignore_case

    def compare(self, *args, **kwargs):  # noqa: D102
        pass


class BaseNumericComparator(BaseComparator):
    """Base class for all numeric comparators.

    This class serves as an abstract base class for implementing
    specific numeric comparator logic in derived classes.
    """

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        self.rtol = rtol
        self.atol = atol

    def compare(self, *args, **kwargs):  # noqa: D102
        pass
