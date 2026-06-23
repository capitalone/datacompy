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

"""Output helpers: emit a :class:`~datacompy.report.ReportData` to stdout."""

import json as _json
import sys
from typing import Any

import numpy as np

from datacompy.report import ReportData


def _json_default(obj: Any) -> Any:
    """Handle numpy scalar types that the stdlib JSON encoder can't serialise."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def emit(report_data: ReportData, *, as_json: bool, quiet: bool) -> None:
    """Write the comparison report to stdout.

    Parameters
    ----------
    report_data:
        Populated report object from ``compare.build_report_data()``.
    as_json:
        When ``True``, serialise via ``ReportData.to_dict()`` and emit
        compact JSON.  When ``False``, emit the text rendering.
    quiet:
        When ``True`` and *as_json* is ``False``, suppress all stdout
        output (useful for pipelines that only inspect the exit code).
    """
    if as_json:
        print(_json.dumps(report_data.to_dict(), indent=2, default=_json_default))  # noqa: T201
    elif not quiet:
        print(report_data.render())  # noqa: T201


def print_error(msg: str) -> None:
    """Write *msg* to stderr with a ``datacompy:`` prefix."""
    print(f"datacompy: {msg}", file=sys.stderr)  # noqa: T201
