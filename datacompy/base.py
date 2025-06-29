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
from typing import Any

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
    ) -> str:
        """Return a string representation of a report."""
        pass

    def only_join_columns(self) -> bool:
        """Boolean on if the only columns are the join columns."""
        return set(self.join_columns) == set(self.df1.columns) == set(self.df2.columns)


def render(template_name: str, **context: Any) -> str:
    """Render a template using Jinja2.

    Parameters
    ----------
    template_name : str
        The name of the template file to render (with or without .j2 extension)
    **context : dict
        The context variables to pass to the template

    Returns
    -------
    str
        The rendered template
    """
    # Initialize Jinja2 environment
    template_path = Path(__file__).parent / "templates"

    # Create Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(template_path),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Ensure template has .j2 extension
    if not template_name.endswith(".j2"):
        template_name = f"{template_name}.j2"

    template = env.get_template(template_name)
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
