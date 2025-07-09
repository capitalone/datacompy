"""
Tests for base.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path to allow importing from datacompy
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict

from datacompy.base import (
    BaseCompare,
    _resolve_template_path,
    _validate_tolerance_parameter,
    df_to_str,
    get_column_tolerance,
    render,
    save_html_report,
    temp_column_name,
)


def test_temp_column_name_no_dataframes():
    """Test with no dataframes provided."""
    result = temp_column_name()
    assert result.startswith("_temp_")
    assert result == "_temp_0"  # First call should be _temp_0


def test_temp_column_name_with_columns():
    """Test with dataframes that have columns."""
    df1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df2 = pd.DataFrame({"col3": [5, 6], "_temp_0": [7, 8]})

    # Should skip _temp_0 since it's in df2
    result = temp_column_name(df1, df2)
    assert result == "_temp_1"


def test_temp_column_name_multiple_calls():
    """Test multiple calls to temp_column_name."""
    # First call
    result1 = temp_column_name()
    assert result1 == "_temp_0"

    # Second call with a dataframe that has _temp_0
    df = pd.DataFrame({"_temp_0": [1, 2]})
    result2 = temp_column_name(df)
    assert result2 == "_temp_1"


def test_df_to_str_pandas_dataframe():
    """Test with pandas DataFrame."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    result = df_to_str(df)
    expected = df.to_string()
    assert result == expected


def test_df_to_str_pandas_dataframe_with_index():
    """Test with pandas DataFrame with index."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["x", "y"])
    result = df_to_str(df, on_index=True)
    expected = df.to_string()
    assert result == expected


def test_df_to_str_pandas_dataframe_without_index():
    """Test with pandas DataFrame without index."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["x", "y"])
    result = df_to_str(df, on_index=False)
    expected = df.reset_index(drop=True).to_string()
    assert result == expected


def test_df_to_str_other_type():
    """Test with a non-DataFrame type."""
    result = df_to_str([1, 2, 3])
    assert result == "[1, 2, 3]"


def test_render_template(tmp_path):
    """Test rendering a simple template."""
    # Create a test template file
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "test_template.j2"

    # Simple template with variables and a loop
    template_content = """Hello {{ name }}!

{% for item in items %}- {{ item }}
{% endfor %}

{% if show_message %}
Message: {{ message }}
{% endif %}"""

    template_file.write_text(template_content)

    # Test rendering with context
    context = {
        "name": "World",
        "items": ["Apple", "Banana", "Cherry"],
        "show_message": True,
        "message": "This is a test message",
    }

    # Render the template
    result = render(str(template_file), **context)

    # Verify the output
    expected_output = """Hello World!

- Apple
- Banana
- Cherry

Message: This is a test message"""

    # Normalize line endings for cross-platform compatibility
    assert (
        result.replace("\r\n", "\n").strip()
        == expected_output.replace("\r\n", "\n").strip()
    )

    # Test with a template in the specified templates directory
    default_template = template_dir / "default_test.j2"
    default_template.write_text("Default template: {{ value }}")

    # Use the full path to the template file
    result = render(str(default_template), value=42)
    assert result == "Default template: 42"


def test_save_html_report(tmp_path):
    """Test saving an HTML report."""
    temp_file = tmp_path / "report.html"

    # Call the function
    report_text = "Test report content"
    save_html_report(report_text, temp_file)

    # Check if file exists
    assert temp_file.exists()

    # Check file content
    content = temp_file.read_text()
    assert "Test report content" in content
    assert "<html>" in content
    assert "</html>" in content


def test_save_html_report_nonexistent_dir(tmp_path):
    """Test saving to a non-existent directory."""
    temp_path = tmp_path / "nonexistent" / "report.html"

    # This should create the directory and save the file
    save_html_report("Test report", temp_path)

    # Check if file exists
    assert temp_path.exists()


class TestCompareImplementation(BaseCompare):
    """Test implementation of BaseCompare for testing."""

    def __init__(self):
        self._df1 = None
        self._df2 = None
        self.join_columns = []

    @property
    def df1(self):
        return self._df1

    @df1.setter
    def df1(self, df1):
        self._df1 = df1

    @property
    def df2(self):
        return self._df2

    @df2.setter
    def df2(self, df2):
        self._df2 = df2

    def _validate_dataframe(self, index, cast_column_names_lower=True):
        pass

    def _compare(self, ignore_spaces, ignore_case):
        pass

    def df1_unq_columns(self):
        return set()

    def df2_unq_columns(self):
        return set()

    def intersect_columns(self):
        return set()

    def _dataframe_merge(self, ignore_spaces):
        pass

    def _intersect_compare(self, ignore_spaces, ignore_case):
        pass

    def all_columns_match(self):
        return True

    def all_rows_overlap(self):
        return True

    def count_matching_rows(self):
        return 0

    def intersect_rows_match(self):
        return True

    def matches(self, ignore_extra_columns=False):
        return True

    def subset(self):
        return True

    def sample_mismatch(self, column, sample_count=10, for_display=False):
        return None

    def all_mismatch(self, ignore_matching_cols=False):
        return None

    def report(self, sample_count=10, column_count=10, html_file=None):
        return ""


def test_base_compare_implementation():
    """Test that BaseCompare can be properly implemented and used."""
    test = TestCompareImplementation()
    assert isinstance(test, BaseCompare)

    # Test only_join_columns method
    test.join_columns = ["id"]
    test.df1 = pd.DataFrame({"id": [1, 2]})
    test.df2 = pd.DataFrame({"id": [1, 2]})
    assert test.only_join_columns()

    test.df2 = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    assert not test.only_join_columns()


def test_resolve_absolute_path(tmp_path):
    """Test resolving an absolute path to a template file."""
    # Create a test template file
    template_file = tmp_path / "test_template.txt"
    template_file.write_text("Test template")

    # Test resolving the absolute path
    template_dir, resolved_name = _resolve_template_path(str(template_file))

    assert template_dir == str(template_file.parent)
    assert resolved_name == template_file.name


def test_resolve_relative_path(tmp_path, monkeypatch):
    """Test resolving a relative path to a template file."""
    # Create a test templates directory and file
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    template_file = templates_dir / "test_template.j2"
    template_file.write_text("Test template")

    # Mock the __file__ to point to our test directory
    monkeypatch.setattr("datacompy.base.__file__", str(tmp_path / "base.py"))

    # Test resolving the relative path
    template_dir, resolved_name = _resolve_template_path("test_template.j2")

    assert template_dir == str(templates_dir)
    assert resolved_name == "test_template.j2"


def test_resolve_with_j2_extension(tmp_path, monkeypatch):
    """Test resolving a template without .j2 extension."""
    # Create a test templates directory and file with .j2 extension
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    template_file = templates_dir / "test_template.j2"
    template_file.write_text("Test template")

    # Mock the __file__ to point to our test directory
    monkeypatch.setattr("datacompy.base.__file__", str(tmp_path / "base.py"))

    # Test resolving without .j2 extension
    template_dir, resolved_name = _resolve_template_path("test_template")

    assert template_dir == str(templates_dir)
    assert resolved_name == "test_template.j2"


def test_resolve_without_j2_extension(tmp_path, monkeypatch):
    """Test resolving a template that doesn't have .j2 extension."""
    # Create a test templates directory and file without .j2 extension
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    template_file = templates_dir / "test_template"
    template_file.write_text("Test template")

    # Mock the __file__ to point to our test directory
    monkeypatch.setattr("datacompy.base.__file__", str(tmp_path / "base.py"))

    # Test resolving with .j2 extension
    template_dir, resolved_name = _resolve_template_path("test_template.j2")

    assert template_dir == str(templates_dir)
    assert resolved_name == "test_template"


def test_resolve_nonexistent_file(tmp_path, monkeypatch):
    """Test resolving a non-existent template file raises FileNotFoundError."""
    # Mock the __file__ to point to our test directory
    monkeypatch.setattr("datacompy.base.__file__", str(tmp_path / "base.py"))

    # Test with a non-existent file
    with pytest.raises(FileNotFoundError) as excinfo:
        _resolve_template_path("nonexistent_template.j2")

    # Check that the error message contains the expected paths
    error_msg = str(excinfo.value)
    assert "nonexistent_template.j2" in error_msg
    assert "tried:" in error_msg


def test_resolve_with_subdirectories(tmp_path, monkeypatch):
    """Test resolving a template in a subdirectory."""
    # Create a test templates directory with subdirectories
    templates_dir = tmp_path / "templates"
    sub_dir = templates_dir / "subdir"
    sub_dir.mkdir(parents=True)

    template_file = sub_dir / "test_template.j2"
    template_file.write_text("Test template")

    # Mock the __file__ to point to our test directory
    monkeypatch.setattr("datacompy.base.__file__", str(tmp_path / "base.py"))

    # Test resolving with a relative path
    template_dir, resolved_name = _resolve_template_path("subdir/test_template.j2")

    assert template_dir == str(templates_dir)
    assert resolved_name == "subdir/test_template.j2"


def test_validate_tolerance_float() -> None:
    """Test validation of float tolerance values."""
    assert _validate_tolerance_parameter(0.1, "abs_tol") == {"default": 0.1}
    assert _validate_tolerance_parameter(0, "abs_tol") == {"default": 0.0}
    with pytest.raises(ValueError, match="abs_tol cannot be negative"):
        _validate_tolerance_parameter(-0.1, "abs_tol")


def test_validate_tolerance_dict() -> None:
    """Test validation of dictionary tolerance values."""
    tol_dict: Dict[str, float] = {"col1": 0.1, "col2": 0.2, "default": 0.05}
    result = _validate_tolerance_parameter(tol_dict, "abs_tol")
    assert result == {"col1": 0.1, "col2": 0.2, "default": 0.05}

    # Test dictionary without default value
    tol_dict = {"col1": 0.1, "col2": 0.2}
    result = _validate_tolerance_parameter(tol_dict, "abs_tol")
    assert result == {"col1": 0.1, "col2": 0.2, "default": 0.0}

    # Test invalid values
    with pytest.raises(ValueError, match="must be numeric"):
        _validate_tolerance_parameter({"col1": "invalid"}, "abs_tol")  # type: ignore
    with pytest.raises(ValueError, match="cannot be negative"):
        _validate_tolerance_parameter({"col1": -0.1}, "abs_tol")


def test_case_sensitivity() -> None:
    """Test case sensitivity handling."""
    tol_dict = {"COL1": 0.1, "Col2": 0.2}

    # Test with case sensitivity lower
    result = _validate_tolerance_parameter(tol_dict, "abs_tol", case_mode="lower")
    assert result == {"col1": 0.1, "col2": 0.2, "default": 0.0}

    # Test with case sensitivity disabled
    result = _validate_tolerance_parameter(tol_dict, "abs_tol", case_mode="preserve")
    assert result == {"COL1": 0.1, "Col2": 0.2, "default": 0.0}

    # Test with case sensitivity upper
    result = _validate_tolerance_parameter(tol_dict, "abs_tol", case_mode="upper")
    assert result == {"COL1": 0.1, "COL2": 0.2, "default": 0.0}


def test_get_column_tolerance_exact_match():
    """Test get_column_tolerance returns the value for an exact column match."""
    tol_dict = {"col1": 0.1, "col2": 0.2, "default": 0.05}
    assert get_column_tolerance("col1", tol_dict) == 0.1
    assert get_column_tolerance("col2", tol_dict) == 0.2


def test_get_column_tolerance_default():
    """Test get_column_tolerance returns the default value if column not found."""
    tol_dict = {"col1": 0.1, "default": 0.05}
    assert get_column_tolerance("colX", tol_dict) == 0.05


def test_get_column_tolerance_no_default():
    """Test get_column_tolerance returns 0.0 if column and default not found."""
    tol_dict = {"col1": 0.1}
    assert get_column_tolerance("colX", tol_dict) == 0.0


def test_get_column_tolerance_empty_dict():
    """Test get_column_tolerance returns 0.0 if tol_dict is empty."""
    tol_dict = {}
    assert get_column_tolerance("col1", tol_dict) == 0.0


def test_get_column_tolerance_column_is_default():
    """Test get_column_tolerance returns the value for 'default' if column is literally 'default'."""
    tol_dict = {"default": 0.07}
    assert get_column_tolerance("default", tol_dict) == 0.07
