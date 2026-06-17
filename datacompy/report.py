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

"""Typed data model for DataComPy report generation.

All backends produce a :class:`ReportData` instance via
``compare.build_report_data()``.  :class:`ReportData` owns rendering to
text and HTML, and exposes ``to_dict()`` for programmatic consumers.

Examples
--------
Programmatic access without rendering:

>>> data = compare.build_report_data()
>>> print(data.row_summary.unequal_rows)
>>> print(data.mismatch_stats.stats[0].column)

Render to text and save HTML:

>>> data = compare.build_report_data()
>>> print(data.render())
>>> data.save("comparison.html")

Export as JSON-serializable dict:

>>> import json
>>> json.dumps(data.to_dict())
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from datacompy.base import render, save_html_report


@dataclass(frozen=True)
class ColumnSummary:
    """Summary of column overlap between the two DataFrames.

    Attributes
    ----------
    common_columns : int
        Number of columns present in both DataFrames.
    df1_unique : int
        Number of columns only in df1.
    df1_unique_columns : tuple of str
        Names of columns only in df1.
    df2_unique : int
        Number of columns only in df2.
    df2_unique_columns : tuple of str
        Names of columns only in df2.
    df1_name : str
        Label for the first DataFrame.
    df2_name : str
        Label for the second DataFrame.
    """

    common_columns: int
    df1_unique: int
    df1_unique_columns: Tuple[str, ...]
    df2_unique: int
    df2_unique_columns: Tuple[str, ...]
    df1_name: str
    df2_name: str


@dataclass(frozen=True)
class RowSummary:
    """Summary of row overlap and match statistics.

    Attributes
    ----------
    match_columns : tuple of str
        Column(s) used to join/match rows.  Empty tuple when matching on index.
    on_index : bool
        True when the comparison was joined on DataFrame index rather than columns.
    has_duplicates : bool
        Whether duplicate join-key values were found.
    abs_tol : float or dict
        Absolute tolerance used for numeric comparisons.
    rel_tol : float or dict
        Relative tolerance used for numeric comparisons.
    common_rows : int
        Number of rows present in both DataFrames after joining.
    df1_unique : int
        Number of rows only in df1.
    df2_unique : int
        Number of rows only in df2.
    unequal_rows : int
        Number of common rows where at least one compared column differs.
    equal_rows : int
        Number of common rows where all compared columns match.
    df1_name : str
        Label for the first DataFrame.
    df2_name : str
        Label for the second DataFrame.
    """

    match_columns: Tuple[str, ...]
    on_index: bool
    has_duplicates: bool
    abs_tol: Any
    rel_tol: Any
    common_rows: int
    df1_unique: int
    df2_unique: int
    unequal_rows: int
    equal_rows: int
    df1_name: str
    df2_name: str


@dataclass(frozen=True)
class ColumnComparison:
    """Aggregate column comparison counts.

    Attributes
    ----------
    unequal_columns : int
        Number of compared columns with at least one unequal value.
    equal_columns : int
        Number of compared columns where all values matched.
    unequal_values : int
        Total number of individual cell values that did not match.
    """

    unequal_columns: int
    equal_columns: int
    unequal_values: int


@dataclass(frozen=True)
class MismatchStat:
    """Per-column mismatch statistics.

    Attributes
    ----------
    column : str
        Column name.
    dtype1 : str
        Data type in df1.
    dtype2 : str
        Data type in df2.
    unequal_cnt : int
        Number of rows where the values differ.
    max_diff : float
        Maximum absolute numeric difference (0.0 for non-numeric columns).
    null_diff : int
        Number of rows where one value is null and the other is not.
    rel_tol : float
        Relative tolerance applied to this column.
    abs_tol : float
        Absolute tolerance applied to this column.
    """

    column: str
    dtype1: str
    dtype2: str
    unequal_cnt: int
    max_diff: float
    null_diff: int
    rel_tol: float
    abs_tol: float


@dataclass(frozen=True)
class MismatchStats:
    """Mismatch statistics across all compared columns.

    Attributes
    ----------
    has_mismatches : bool
        True when at least one column has unequal values or types.
    has_samples : bool
        True when sample rows are available for display.
    stats : tuple of MismatchStat
        Per-column statistics, sorted by column name.
    samples : tuple of str
        Pre-rendered string tables of sample mismatched rows (one per column).
    df1_name : str
        Label for the first DataFrame.
    df2_name : str
        Label for the second DataFrame.
    """

    has_mismatches: bool
    has_samples: bool
    stats: Tuple[MismatchStat, ...] = ()
    samples: Tuple[str, ...] = ()
    df1_name: str = ""
    df2_name: str = ""


@dataclass(frozen=True)
class UniqueRowsData:
    """Sample rows that exist in only one of the two DataFrames.

    Attributes
    ----------
    has_rows : bool
        True when there is at least one unique row to display.
    rows : str
        Pre-rendered string table of sample rows.
    """

    has_rows: bool
    rows: str = ""


@dataclass(frozen=True, repr=False)
class ReportData:
    """Complete data model for a DataComPy comparison report.

    Produced by ``compare.build_report_data()``.  All fields are immutable
    so the object can be safely cached or passed across threads.

    Attributes
    ----------
    df1_name : str
        Label for the first DataFrame.
    df2_name : str
        Label for the second DataFrame.
    df1_shape : tuple of (int, int)
        ``(rows, columns)`` of df1.
    df2_shape : tuple of (int, int)
        ``(rows, columns)`` of df2.
    column_count : int
        Maximum number of columns displayed in unique-row samples.
    column_summary : ColumnSummary
    row_summary : RowSummary
    column_comparison : ColumnComparison
    mismatch_stats : MismatchStats
    df1_unique_rows : UniqueRowsData
    df2_unique_rows : UniqueRowsData
    """

    df1_name: str
    df2_name: str
    df1_shape: Tuple[int, int]
    df2_shape: Tuple[int, int]
    column_count: int
    column_summary: ColumnSummary
    row_summary: RowSummary
    column_comparison: ColumnComparison
    mismatch_stats: MismatchStats
    df1_unique_rows: UniqueRowsData
    df2_unique_rows: UniqueRowsData

    def to_template_context(self) -> Dict[str, Any]:
        """Return a dict suitable for passing directly to the Jinja2 template.

        The template contract uses pre-formatted strings for a few fields
        (``row_summary.match_columns`` as comma-joined str, ``has_duplicates``
        as ``"Yes"``/``"No"``).  This method emits those stringified values
        while keeping the dataclass fields typed.

        Returns
        -------
        dict
            Context dict accepted by ``report_template.j2`` and custom templates.
        """
        cs = self.column_summary
        rs = self.row_summary

        match_columns_str = "index" if rs.on_index else ", ".join(rs.match_columns)
        has_dupes_str = "Yes" if rs.has_duplicates else "No"

        # Render df1/df2_unique as "N ['col1', 'col2', ...]" when there are
        # unique columns (consistent across all backends), or plain N when empty.
        def _fmt_unique(count: int, cols: Tuple[str, ...]) -> Any:
            return f"{count} {list(cols)}" if cols else count

        return {
            "df1_name": self.df1_name,
            "df2_name": self.df2_name,
            "df1_shape": self.df1_shape,
            "df2_shape": self.df2_shape,
            "column_count": self.column_count,
            "column_summary": {
                "common_columns": cs.common_columns,
                "df1_unique": _fmt_unique(cs.df1_unique, cs.df1_unique_columns),
                "df2_unique": _fmt_unique(cs.df2_unique, cs.df2_unique_columns),
                "df1_name": cs.df1_name,
                "df2_name": cs.df2_name,
            },
            "row_summary": {
                "match_columns": match_columns_str,
                "has_duplicates": has_dupes_str,
                "abs_tol": rs.abs_tol,
                "rel_tol": rs.rel_tol,
                "common_rows": rs.common_rows,
                "df1_unique": rs.df1_unique,
                "df2_unique": rs.df2_unique,
                "unequal_rows": rs.unequal_rows,
                "equal_rows": rs.equal_rows,
                "df1_name": rs.df1_name,
                "df2_name": rs.df2_name,
            },
            "column_comparison": dataclasses.asdict(self.column_comparison),
            "mismatch_stats": {
                "has_mismatches": self.mismatch_stats.has_mismatches,
                "has_samples": self.mismatch_stats.has_samples,
                "stats": [dataclasses.asdict(s) for s in self.mismatch_stats.stats],
                "samples": list(self.mismatch_stats.samples),
                "df1_name": self.mismatch_stats.df1_name,
                "df2_name": self.mismatch_stats.df2_name,
            },
            "df1_unique_rows": dataclasses.asdict(self.df1_unique_rows),
            "df2_unique_rows": dataclasses.asdict(self.df2_unique_rows),
        }

    def render(self, template_path: str | None = None) -> str:
        """Render the report to a text string.

        Parameters
        ----------
        template_path : str, optional
            Path to a custom Jinja2 template.  When ``None`` the default
            ``report_template.j2`` is used.

        Returns
        -------
        str
            Formatted report text.
        """
        return render(
            template_path or "report_template.j2", **self.to_template_context()
        )

    def to_html(self, template_path: str | None = None) -> str:
        """Return the report wrapped in a minimal HTML page.

        Parameters
        ----------
        template_path : str, optional
            Path to a custom Jinja2 template.  When ``None`` the default
            ``report_template.j2`` is used.

        Returns
        -------
        str
            HTML string with the text report inside a ``<pre>`` block.
        """
        text = self.render(template_path)
        return (
            f"<html><head><title>DataComPy Report</title></head>"
            f"<body><pre>{text}</pre></body></html>"
        )

    def save(self, path: str | Path, template_path: str | None = None) -> None:
        """Save the report as an HTML file.

        Parameters
        ----------
        path : str or Path
            Destination file path.  Parent directories are created if needed.
        template_path : str, optional
            Path to a custom Jinja2 template.  When ``None`` the default
            ``report_template.j2`` is used.
        """
        save_html_report(self.render(template_path), path)

    def to_dict(self) -> Dict[str, Any]:
        """Return the report data as a JSON-serializable dict.

        Returns
        -------
        dict
            ``dataclasses.asdict(self)`` — safe for ``json.dumps()``.
        """
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        """Return the rendered report as a string."""
        return self.render()

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return (
            f"ReportData(df1={self.df1_name!r}, df2={self.df2_name!r}, "
            f"shape1={self.df1_shape}, shape2={self.df2_shape})"
        )
