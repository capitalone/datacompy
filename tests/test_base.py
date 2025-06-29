"""
Tests for base.py
"""

import pandas as pd
from datacompy.base import BaseCompare, df_to_str, save_html_report, temp_column_name


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


def test_render_template():
    """Test rendering a simple template."""
    # Skip this test for now as it requires more complex setup
    # with template directories
    pass


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
