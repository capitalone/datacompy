import pytest
from datacompy.comparator.base import (
    BaseComparator,
    BaseNumericComparator,
    BaseStringComparator,
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


def test_base_string_comparator_initialization():
    """Test initialization of BaseStringComparator with default values."""
    comparator = BaseStringComparator()
    assert comparator.ignore_space is True
    assert comparator.ignore_case is True


def test_base_string_comparator_custom_initialization():
    """Test initialization of BaseStringComparator with custom values."""
    comparator = BaseStringComparator(ignore_space=False, ignore_case=False)
    assert comparator.ignore_space is False
    assert comparator.ignore_case is False


def test_base_numeric_comparator_initialization():
    """Test initialization of BaseNumericComparator with default values."""
    comparator = BaseNumericComparator()
    assert comparator.rtol == 1e-5
    assert comparator.atol == 1e-8


def test_base_numeric_comparator_custom_initialization():
    """Test initialization of BaseNumericComparator with custom values."""
    comparator = BaseNumericComparator(rtol=1e-3, atol=1e-6)
    assert comparator.rtol == 1e-3
    assert comparator.atol == 1e-6
