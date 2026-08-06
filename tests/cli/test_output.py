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

"""Tests for report rendering and delivery.

Rendering (``--report-format``) and destination (``--output``) are independent
axes, so the tests cover the combinations rather than a single flag each.
"""

import json

import pytest

from tests.cli.conftest import TOTAL_DIFFERING_ROWS, UNEQUAL_ROWS, UNIQUE_ROWS

MISMATCH = 1
ERROR = 2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_text_is_the_default_rendering(cli, capsys):
    assert cli() == MISMATCH
    out = capsys.readouterr().out
    assert "DataComPy Comparison" in out
    assert not out.lstrip().startswith("{")


def test_json_rendering_is_parseable_and_carries_the_counts(cli, capsys):
    assert cli("--report-format", "json") == MISMATCH
    payload = json.loads(capsys.readouterr().out)

    assert payload["row_summary"]["unequal_rows"] == UNEQUAL_ROWS
    assert payload["row_summary"]["df1_unique"] == 1
    assert payload["row_summary"]["df2_unique"] == 1
    assert payload["column_summary"]["common_columns"] == 3
    assert payload["df1_name"] == "left"
    assert payload["df2_name"] == "right"


def test_json_rendering_survives_numpy_scalars(cli, capsys):
    """``max_diff`` and friends arrive as numpy scalars, which stdlib JSON rejects."""
    assert cli("--report-format", "json") == MISMATCH
    payload = json.loads(capsys.readouterr().out)
    stats = payload["mismatch_stats"]["stats"]
    assert stats, "expected at least one mismatching column"
    assert stats[0]["max_diff"] == pytest.approx(0.005)


def test_json_default_coerces_types_the_stdlib_encoder_rejects():
    """Spark and Snowflake surface numpy scalars that ``json.dumps`` cannot encode."""
    import datetime

    import numpy as np
    from datacompy.cli.output import _json_default

    assert _json_default(np.int64(7)) == 7
    assert isinstance(_json_default(np.int64(7)), int)
    assert _json_default(np.float64(1.5)) == pytest.approx(1.5)
    assert isinstance(_json_default(np.float64(1.5)), float)
    assert _json_default(np.bool_(True)) is True
    # Anything else degrades to its string form rather than blowing up a report.
    assert _json_default(datetime.date(2026, 7, 31)) == "2026-07-31"


def test_html_rendering_wraps_the_text_report(cli, capsys):
    assert cli("--report-format", "html") == MISMATCH
    out = capsys.readouterr().out
    assert out.lstrip().startswith("<html>")
    assert "DataComPy Comparison" in out


# ---------------------------------------------------------------------------
# Destination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("report_format", ["text", "json", "html"])
def test_output_writes_a_file_and_creates_parent_directories(
    cli, tmp_path, report_format, capsys
):
    destination = tmp_path / "nested" / "deeper" / f"report.{report_format}"
    assert (
        cli("--report-format", report_format, "--output", str(destination)) == MISMATCH
    )

    content = destination.read_text()
    assert content
    if report_format == "json":
        assert json.loads(content)["row_summary"]["unequal_rows"] == UNEQUAL_ROWS
    elif report_format == "html":
        assert content.lstrip().startswith("<html>")
    else:
        assert "DataComPy Comparison" in content


def test_output_still_prints_to_stdout_by_default(cli, tmp_path, capsys):
    destination = tmp_path / "report.txt"
    assert cli("--output", str(destination)) == MISMATCH
    assert "DataComPy Comparison" in capsys.readouterr().out
    assert "DataComPy Comparison" in destination.read_text()


def test_quiet_suppresses_stdout_but_still_writes_the_file(cli, tmp_path, capsys):
    destination = tmp_path / "report.html"
    assert (
        cli("--quiet", "--report-format", "html", "--output", str(destination))
        == MISMATCH
    )
    assert capsys.readouterr().out == ""
    assert destination.read_text().lstrip().startswith("<html>")


def test_quiet_without_output_prints_nothing_at_all(cli, capsys):
    assert cli("--quiet") == MISMATCH
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_quiet_and_json_still_suppresses_stdout(cli, capsys):
    """PR 534's --quiet was a no-op alongside --json; format and silence are separate."""
    assert cli("--quiet", "--report-format", "json") == MISMATCH
    assert capsys.readouterr().out == ""


def test_unwritable_output_exits_two(cli, tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    assert cli("--output", str(blocker / "report.txt")) == ERROR
    assert "cannot write" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Report shaping
# ---------------------------------------------------------------------------


def test_sample_count_zero_suppresses_sample_rows(cli, capsys):
    assert cli("--sample-count", "0", "--report-format", "json") == MISMATCH
    payload = json.loads(capsys.readouterr().out)
    assert payload["mismatch_stats"]["has_samples"] is False
    assert payload["df1_unique_rows"]["has_rows"] is False


def test_column_count_is_reflected_in_the_report_data(cli, capsys):
    assert cli("--column-count", "2", "--report-format", "json") == MISMATCH
    assert json.loads(capsys.readouterr().out)["column_count"] == 2


def test_dataset_labels_can_be_overridden(cli, capsys):
    assert cli("--df1-name", "before", "--df2-name", "after") == MISMATCH
    out = capsys.readouterr().out
    assert "before" in out
    assert "after" in out


def test_the_fixture_counts_are_what_the_report_says(cli, capsys):
    assert cli("--report-format", "json") == MISMATCH
    summary = json.loads(capsys.readouterr().out)["row_summary"]
    total = summary["unequal_rows"] + summary["df1_unique"] + summary["df2_unique"]
    assert total == TOTAL_DIFFERING_ROWS
    assert summary["df1_unique"] + summary["df2_unique"] == UNIQUE_ROWS
