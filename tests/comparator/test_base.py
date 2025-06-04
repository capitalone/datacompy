import pytest
from datacompy.comparator.base import BaseComparator


def test_base_comparator_abstract_method():
    """Test that BaseComparator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseComparator()


def test_base_comparator_compare_not_implemented():
    """Test that the compare method raises NotImplementedError."""

    class TestComparator(BaseComparator):
        pass

    with pytest.raises(TypeError):
        TestComparator()

    class ValidComparator(BaseComparator):
        def compare(self):
            return True

    comparator = ValidComparator()
    assert comparator.compare() is True
