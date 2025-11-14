import pytest
from datacompy.utility import check_module_available


def test_check_module_available_success():
    """
    Test that the decorated function executes successfully when the module is available.
    """

    @check_module_available(True, "test_extra")
    def dummy_function():
        return "Function executed"

    assert dummy_function() == "Function executed"


def test_check_module_available_failure():
    """
    Test that ImportError is raised when the module is not available.
    """

    @check_module_available(False, "test_extra")
    def dummy_function():
        pass

    with pytest.raises(ImportError) as excinfo:
        dummy_function()

    assert "The 'test_extra' extra is not installed." in str(excinfo.value)
    assert "pip install datacompy[test_extra]" in str(excinfo.value)
