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

"""Report rendering and delivery for the DataComPy CLI.

Rendering and destination are independent: ``--report-format`` chooses between
text, JSON, and HTML, and ``--output`` chooses between stdout and a file. All
three renderings come from :class:`datacompy.report.ReportData`, so the CLI adds
no templating of its own.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from datacompy.cli.errors import OutputError
from datacompy.report import ReportData


def _json_default(obj: Any) -> Any:
    """Coerce numpy scalars, which the stdlib JSON encoder cannot serialise."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def render(report_data: ReportData, report_format: str) -> str:
    """Render *report_data* in the requested format.

    Parameters
    ----------
    report_data : datacompy.report.ReportData
        Structured comparison result from ``compare.build_report_data()``.
    report_format : {"text", "json", "html"}
        Rendering to produce.

    Returns
    -------
    str
        The rendered report.
    """
    if report_format == "json":
        return json.dumps(report_data.to_dict(), indent=2, default=_json_default)
    if report_format == "html":
        return report_data.to_html()
    return report_data.render()


def emit(
    report_data: ReportData,
    report_format: str,
    output: Path | None,
    *,
    quiet: bool,
) -> None:
    """Write the report to stdout, to *output*, or to both.

    ``quiet`` suppresses stdout only. A file requested with ``--output`` is
    always written, since asking for a file is an explicit request for it.

    Raises
    ------
    OutputError
        When the destination file cannot be written.
    """
    if quiet and output is None:
        return

    # Rendered once and reused, so asking for a file and stdout together does
    # not template the same ReportData twice.
    rendered = render(report_data, report_format)

    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            raise OutputError(f"cannot write {output}: {exc}") from exc

    if not quiet:
        print(rendered)


def print_error(message: str) -> None:
    """Write *message* to stderr with a ``datacompy:`` prefix."""
    print(f"datacompy: {message}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Write *message* to stderr with a ``datacompy: warning:`` prefix.

    Unlike :func:`print_error` this does not accompany a non-zero exit. It is
    for cases the CLI can see are probably wrong but cannot prove, such as a
    comparison about to run on a file that looks misparsed.
    """
    print(f"datacompy: warning: {message}", file=sys.stderr)
