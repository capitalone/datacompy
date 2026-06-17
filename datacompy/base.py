#
# Copyright 2026 Capital One Services, LLC
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
Base module for comparing two DataFrames.

Originally this package was meant to provide similar functionality to
PROC COMPARE in SAS - i.e. human-readable reporting on the difference between
two dataframes.
"""

import logging
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import datacompy.report

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

    @property
    def sensitive_columns(self) -> List[str] | None:
        """Get the list of sensitive columns."""
        return self._sensitive_columns

    def _set_and_validate_sensitive_columns(
        self, sensitive_columns: List[str] | None
    ) -> None:
        """Set and validate sensitive columns.

        Normalizes empty lists to None so there is only one representation
        for "no sensitive columns".
        """
        self._sensitive_columns = sensitive_columns or None
        if not self._sensitive_columns:
            return

        if not all(isinstance(c, str) for c in self.sensitive_columns):
            raise TypeError("sensitive_columns must be a list of strings")

        # Cast to lowercase if applicable
        if self.cast_column_names_lower:
            self._sensitive_columns = [col.lower() for col in self.sensitive_columns]

        # Check duplicates
        duplicates = {c for c, n in Counter(self.sensitive_columns).items() if n > 1}
        if duplicates:
            raise ValueError(f"duplicate columns: {duplicates}")

        # Warn if column not found in either dataframe
        unused = [
            col
            for col in self.sensitive_columns
            if (col not in self.df1.columns) and (col not in self.df2.columns)
        ]
        if unused:
            LOG.warning(
                f"sensitive columns not found in either df1 or df2 will be ignored: {unused}"
            )

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

    # ------------------------------------------------------------------
    # Backend primitives — override in subclasses where the default
    # (.shape / .columns) does not apply (e.g. Spark / Snowflake).
    # ------------------------------------------------------------------

    def _table_shape(self, df: Any) -> tuple[int, int]:
        """Return ``(rows, columns)`` for *df*.

        Spark and Snowflake backends override this to call ``.count()``.
        """
        return df.shape  # type: ignore[no-any-return]

    def _row_count(self, df: Any, _cache: Dict[int, int] | None = None) -> int:
        """Return the row count for *df*, using *_cache* when provided.

        Parameters
        ----------
        df :
            The DataFrame whose rows to count.
        _cache : dict, optional
            Mapping from ``id(df)`` to cached count.  Spark and Snowflake
            backends pass this in to avoid redundant ``.count()`` actions.
        """
        if _cache is not None:
            key = id(df)
            if key not in _cache:
                _cache[key] = int(df.shape[0])
            return _cache[key]
        return int(df.shape[0])

    def _select_first_n_columns(self, df: Any, n: int) -> Any:
        """Return *df* with only its first *n* columns.

        Pandas overrides this with ``.iloc[:, :n]``.
        """
        return df.select(list(df.columns)[:n])

    def _column_names(self, df: Any) -> List[str]:
        """Return column names of *df* as a plain list."""
        return list(df.columns)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Concrete report-building — shared across all backends
    # ------------------------------------------------------------------

    def build_report_data(
        self, sample_count: int = 10, column_count: int = 10
    ) -> "datacompy.report.ReportData":
        """Build a typed :class:`~datacompy.report.ReportData` from this comparison.

        Parameters
        ----------
        sample_count : int, optional
            Maximum number of sample rows to include per section. Defaults to 10.
        column_count : int, optional
            Maximum number of columns to show in unique-row samples. Defaults to 10.

        Returns
        -------
        datacompy.report.ReportData
            Immutable data object suitable for rendering or programmatic use.

        Examples
        --------
        >>> data = compare.build_report_data()
        >>> print(data.row_summary.unequal_rows)
        """
        from datacompy.report import (
            ColumnComparison,
            ColumnSummary,
            MismatchStat,
            MismatchStats,
            ReportData,
            RowSummary,
            UniqueRowsData,
        )

        # Per-call row-count cache so Spark/Snowflake only trigger .count() once
        # per unique DataFrame object.
        row_count_cache: Dict[int, int] = {}

        # ---- column summary ------------------------------------------
        df1_unq_cols = self.df1_unq_columns()
        df2_unq_cols = self.df2_unq_columns()
        column_summary = ColumnSummary(
            common_columns=len(self.intersect_columns()),
            df1_unique=len(df1_unq_cols),
            df1_unique_columns=tuple(df1_unq_cols),
            df2_unique=len(df2_unq_cols),
            df2_unique_columns=tuple(df2_unq_cols),
            df1_name=self.df1_name,
            df2_name=self.df2_name,
        )

        # ---- row summary ---------------------------------------------
        intersect_count = self._row_count(self.intersect_rows, row_count_cache)
        df1_unq_count = self._row_count(self.df1_unq_rows, row_count_cache)
        df2_unq_count = self._row_count(self.df2_unq_rows, row_count_cache)
        matching_rows = self.count_matching_rows()

        on_index: bool = getattr(self, "on_index", False)
        row_summary = RowSummary(
            match_columns=tuple(self.join_columns),
            on_index=on_index,
            has_duplicates=bool(self._any_dupes),
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            common_rows=intersect_count,
            df1_unique=df1_unq_count,
            df2_unique=df2_unq_count,
            unequal_rows=intersect_count - matching_rows,
            equal_rows=matching_rows,
            df1_name=self.df1_name,
            df2_name=self.df2_name,
        )

        # ---- column comparison ---------------------------------------
        column_comparison = ColumnComparison(
            unequal_columns=len([c for c in self.column_stats if c["unequal_cnt"] > 0]),
            equal_columns=len([c for c in self.column_stats if c["unequal_cnt"] == 0]),
            unequal_values=sum(c["unequal_cnt"] for c in self.column_stats),
        )

        # ---- mismatch stats ------------------------------------------
        stat_list: List[MismatchStat] = []
        sample_list: List[Any] = []
        any_mismatch = False

        for col in self.column_stats:
            if not col["all_match"]:
                any_mismatch = True
                stat_list.append(
                    MismatchStat(
                        column=col["column"],
                        dtype1=col["dtype1"],
                        dtype2=col["dtype2"],
                        unequal_cnt=col["unequal_cnt"],
                        max_diff=col["max_diff"],
                        null_diff=col["null_diff"],
                        rel_tol=col["rel_tol"],
                        abs_tol=col["abs_tol"],
                    )
                )
                if col["unequal_cnt"] > 0:
                    sample_list.append(
                        self.sample_mismatch(
                            col["column"], sample_count, for_display=True
                        )
                    )

        if any_mismatch:
            mismatch_stats = MismatchStats(
                has_mismatches=True,
                has_samples=len(sample_list) > 0 and sample_count > 0,
                stats=tuple(sorted(stat_list, key=lambda s: s.column)),
                samples=tuple(df_to_str(s) for s in sample_list),
                df1_name=self.df1_name,
                df2_name=self.df2_name,
            )
        else:
            mismatch_stats = MismatchStats(has_mismatches=False, has_samples=False)

        # ---- unique rows data ----------------------------------------
        def _unique_rows_data(df: Any, unq_count: int) -> "UniqueRowsData":
            min_sample = min(sample_count, unq_count)
            min_cols = min(column_count, len(self._column_names(df)))
            if unq_count > 0:
                rows_str = df_to_str(
                    self._select_first_n_columns(df, min_cols),
                    sample_count=min_sample,
                )
            else:
                rows_str = ""
            return UniqueRowsData(has_rows=min_sample > 0, rows=rows_str)

        df1_unique_rows = _unique_rows_data(self.df1_unq_rows, df1_unq_count)
        df2_unique_rows = _unique_rows_data(self.df2_unq_rows, df2_unq_count)

        # ---- assemble ------------------------------------------------
        return ReportData(
            df1_name=self.df1_name,
            df2_name=self.df2_name,
            df1_shape=self._table_shape(self.df1),
            df2_shape=self._table_shape(self.df2),
            column_count=column_count,
            column_summary=column_summary,
            row_summary=row_summary,
            column_comparison=column_comparison,
            mismatch_stats=mismatch_stats,
            df1_unique_rows=df1_unique_rows,
            df2_unique_rows=df2_unique_rows,
        )

    def report(
        self,
        sample_count: int = 10,
        column_count: int = 10,
        html_file: str | None = None,
        template_path: str | None = None,
    ) -> str:
        """Return a string representation of a report.

        The representation can then be printed or saved to a file. You can customize the
        report's appearance by providing a custom Jinja2 template.

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
            If ``None``, the default template will be used. The template receives the
            context variables documented on :class:`~datacompy.report.ReportData`.

        Returns
        -------
        str
            The report, formatted according to the template.

        See Also
        --------
        build_report_data : Access the structured data without rendering.
        """
        data = self.build_report_data(sample_count, column_count)
        text = data.render(template_path=template_path)
        if html_file:
            save_html_report(text, html_file)
        return text

    def reveal_sensitive_columns(self) -> None:
        """Reveals all sensitive columns.

        Notes
        -----
        - This re-runs the full comparison to restore original values.
        - Revealing sensitive columns when there aren't any is treated as a NOP
          to avoid redundant computations.
        """
        # Don't do anything if there aren't any sensitive columns
        if not self.sensitive_columns:
            return

        LOG.debug("Revealing sensitive columns and re-comparing dfs")
        self._set_and_validate_sensitive_columns(None)
        self.column_stats.clear()
        self._compare(ignore_spaces=self.ignore_spaces, ignore_case=self.ignore_case)

    def only_join_columns(self) -> bool:
        """Boolean on if the only columns are the join columns."""
        return set(self.join_columns) == set(self.df1.columns) == set(self.df2.columns)

    def columns_with_mismatches(self) -> list[str]:
        """Return a list of column names where at least one row has a mismatch.

        This method identifies columns that have differences between df1 and df2,
        excluding the join columns. This is useful for identifying problematic
        columns and potentially rerunning comparisons on a subset of the data.

        Returns
        -------
        list[str]
            A list of column names that have at least one mismatch.

        Examples
        --------
        >>> compare = PandasCompare(df1, df2, join_columns=['id'])
        >>> mismatched_cols = compare.columns_with_mismatches()
        >>> print(mismatched_cols)
        ['col_a', 'col_b']
        """
        return [
            col["column"]
            for col in self.column_stats
            if col["unequal_cnt"] > 0 and col["column"] not in self.join_columns
        ]


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
        if sample_count is not None and len(df) > sample_count:
            df = df.head(sample_count)
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
        if sample_count is not None and len(df) > sample_count:
            df = df.head(sample_count)
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


def validate_tolerance_parameter(
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
