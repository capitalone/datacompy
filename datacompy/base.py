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
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape
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
        template_path: str | None = None,
    ) -> str:
        """Return a string representation of a report.

        Parameters
        ----------
        sample_count : int, optional
            The number of sample records to return. Defaults to 10.

        column_count : int, optional
            The number of columns to display in the sample records output. Defaults to 10.

        html_file : str, optional
            HTML file name to save report output to. If ``None`` the file creation will be skipped.

        template_path : str, optional
            Path to a custom Jinja2 template file to use for report generation.
            If ``None``, the default template will be used.

        Returns
        -------
        str
            The report, formatted according to the template.
        """
        pass

    def only_join_columns(self) -> bool:
        """Boolean on if the only columns are the join columns."""
        return set(self.join_columns) == set(self.df1.columns) == set(self.df2.columns)


def _resolve_template_path(template_name: str) -> tuple[str, str]:
    """Resolve template path and return (template_dir, template_file).

    Handles both absolute and relative paths, and manages .j2 extension automatically.
    """
    template_path = Path(template_name)

    # Handle absolute paths
    if template_path.is_absolute():
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        return str(template_path.parent), template_path.name

    # Handle relative paths in templates directory
    template_dir = str(Path(__file__).parent / "templates")
    template_file = str(template_path)
    full_path = Path(template_dir) / template_file

    # Try different file variations in order:
    # 1. As given
    # 2. With .j2 extension
    # 3. Without .j2 extension (if input had it)
    if full_path.exists():
        return template_dir, template_file

    j2_path = full_path.with_suffix(".j2")
    if j2_path.exists():
        return template_dir, j2_path.name

    if template_file.endswith(".j2"):
        base_path = full_path.with_suffix("")
        if base_path.exists():
            return template_dir, base_path.name

    # If we get here, no variation exists
    raise FileNotFoundError(
        f"Template file not found: {template_name} "
        f"(tried: {full_path}, {j2_path}"
        + (f", {full_path.with_suffix('')}" if template_file.endswith(".j2") else "")
        + ")"
    )


def render(template_name: str, **context: Any) -> str:
    """Render a template using Jinja2.

    Parameters
    ----------
    template_name : str
        The name of the template file to render. This can be:
        - A filename in the default templates directory (with or without .j2 extension)
        - A relative path from the default templates directory
        - An absolute path to a template file
    **context : dict
        The context variables to pass to the template

    Returns
    -------
    str
        The rendered template

    Raises
    ------
    FileNotFoundError
        If the template file cannot be found in any of the expected locations
    """
    template_dir, template_file = _resolve_template_path(template_name)

    # Create Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_file)
    return template.render(**context).strip()


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
    for df in dataframes:
        if df is not None:
            columns.extend(df.columns)
    while True:
        tmp = f"_temp_{i}"
        if tmp not in columns:
            return tmp
        i += 1


def save_html_report(report: str, html_file: str | Path) -> None:
    """Save a text report as an HTML file.

    Parameters
    ----------
    report : str
        The text report to convert to HTML
    html_file : str or Path
        The path where the HTML file should be saved
    """
    html_file = Path(html_file)
    html_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a simple HTML template with the report in a pre tag
    html_content = f"""<html><head><title>DataComPy Report</title></head><body><pre>{report}</pre></body></html>"""
    # Save the HTML file
    html_file.write_text(html_content, encoding="utf-8")


def df_to_str(df: Any, sample_count: int | None = None, on_index: bool = False) -> str:
    """Convert a DataFrame to a string representation.

    This is a centralized function to handle DataFrame to string conversion for different
    DataFrame types (pandas, Spark, Polars, etc.)

    Parameters
    ----------
    df : Any
        The DataFrame to convert to string. Can be pandas, Spark, or Polars DataFrame.
    sample_count : int, optional
        For distributed DataFrames (like Spark), limit the number of rows to convert.
    on_index : bool, default False
        If True, include the index in the output.

    Returns
    -------
    str
        String representation of the DataFrame
    """
    # Handle pandas DataFrame
    if hasattr(df, "to_string"):
        if not on_index and hasattr(df, "reset_index"):
            df = df.reset_index(drop=True)
        return df.to_string()

    # Handle Spark DataFrame and Snowflake DataFrame
    if hasattr(df, "toPandas"):
        if sample_count is not None:
            df = df.limit(sample_count)
        return df.toPandas().to_string()

    # Handle Polars DataFrame
    if hasattr(df, "to_pandas"):
        return df.to_pandas().to_string()

    # Fallback to str() if we can't determine the type
    return str(df)


def get_column_tolerance(column: str, tol_dict: Dict[str, float]) -> float:
    """
    Return the tolerance value for a given column from a dictionary of tolerances.

    Parameters
    ----------
    column : str
        The name of the column for which to retrieve the tolerance.
    tol_dict : dict of str to float
        Dictionary mapping column names to their tolerance values.
        May contain a "default" key for columns not explicitly listed.

    Returns
    -------
    float
        The tolerance value for the specified column, or the "default" tolerance if the column is not found.
        Returns 0 if neither the column nor "default" is present in the dictionary.
    """
    return tol_dict.get(column, tol_dict.get("default", 0.0))


def _validate_tolerance_parameter(
    param_value: float | Dict[str, float],
    param_name: str,
    case_mode: str = "lower",
) -> Dict[str, float]:
    """Validate and normalize tolerance parameter input.

    Parameters
    ----------
    param_value : float or dict
        The tolerance value to validate. Can be either a float or a dictionary mapping
        column names to float values.
    param_name : str
        Name of the parameter being validated ('abs_tol' or 'rel_tol')
    case_mode : str
        How to handle column name case. Options are:
        - "lower": convert to lowercase
        - "upper": convert to uppercase
        - "preserve": keep original case

    Returns
    -------
    dict
        Normalized dictionary of tolerance values

    Raises
    ------
    TypeError
        If param_value is not a float or dict
    ValueError
        If any tolerance values are not numeric or negative or if case_mode is invalid
    """
    if case_mode not in ["lower", "upper", "preserve"]:
        raise ValueError("case_mode must be 'lower', 'upper', or 'preserve'")

    # If float, convert to dict with default value
    if isinstance(param_value, int | float):
        if param_value < 0:
            raise ValueError(f"{param_name} cannot be negative")
        return {"default": float(param_value)}

    # If dict, validate values and format
    if isinstance(param_value, dict):
        result = {}

        # Convert all values to float and validate
        for col, value in param_value.items():
            if not isinstance(value, int | float):
                raise ValueError(
                    f"Value for column '{col}' in {param_name} must be numeric"
                )
            if value < 0:
                raise ValueError(
                    f"Value for column '{col}' in {param_name} cannot be negative"
                )

            # Handle column name case according to case_mode
            col_key = str(col)
            if case_mode == "lower":
                col_key = col_key.lower()
            elif case_mode == "upper":
                col_key = col_key.upper()

            result[col_key] = float(value)

        # If no default provided, add 0.0
        if "default" not in result:
            result["default"] = 0.0

        return result

    raise TypeError(f"{param_name} must be a float or dictionary")
