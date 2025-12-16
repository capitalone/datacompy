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

import pytest
from datacompy.comparator.base import (
    BaseComparator,
)


class ValidComparator(BaseComparator):
    def compare(self):
        return True


def test_base_comparator_abstract_method():
    """Test that BaseComparator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseComparator()


def test_base_comparator_compare_not_implemented():
    """Test that the compare method raises TypeError."""

    class TestComparator(BaseComparator):
        pass

    with pytest.raises(TypeError):
        TestComparator()

    comparator = ValidComparator()
    assert comparator.compare() is True
