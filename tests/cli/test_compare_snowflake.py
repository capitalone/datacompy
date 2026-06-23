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

"""CLI integration tests for the Snowflake backend.

Skipped automatically when snowflake.snowpark is not installed.

The ``test_snowflake_missing_account_exits_2`` test runs without a real
Snowflake account (it only tests the error path in ``get_snowflake_session``).

The comparison test ``test_snowflake_compare`` fully mocks the compare
layer so it works without any live Snowflake session.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

snowpark = pytest.importorskip("snowflake.snowpark")

from tests.cli.conftest import run


def test_snowflake_missing_account_exits_2(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_AUTHENTICATOR",
    ):
        monkeypatch.delenv(var, raising=False)
    code, _, err = run(
        [
            "compare",
            "--left",
            "dummy_left.csv",
            "--right",
            "dummy_right.csv",
            "--on",
            "id",
            "--backend",
            "snowflake",
        ],
        capsys,
    )
    assert code == 2
    assert (
        "SNOWFLAKE_ACCOUNT" in err
        or "SNOWFLAKE_USER" in err
        or "missing" in err.lower()
    )


def test_snowflake_compare_match(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mock out the full compare stack so no live Snowflake session is needed."""
    import datacompy.report as report_mod  # noqa: F401

    mock_row_summary = MagicMock()
    mock_row_summary.unequal_rows = 0
    mock_report = MagicMock()
    mock_report.render.return_value = "DataComPy Comparison\nMatch: True"
    mock_report.row_summary = mock_row_summary

    mock_compare = MagicMock()
    mock_compare.build_report_data.return_value = mock_report
    mock_compare.matches.return_value = True

    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,val\n1,a\n")
    right.write_text("id,val\n1,a\n")

    with (
        patch("datacompy.cli.sessions.get_snowflake_session") as mock_sess,
        patch("datacompy.cli.commands.compare.load_snowflake") as mock_load,
        patch(
            "datacompy.cli.commands.compare.make_snowflake_compare",
            return_value=mock_compare,
        ),
    ):
        mock_sess.return_value = MagicMock()
        mock_load.return_value = "DB.SCHEMA.TABLE"

        code, out, err = run(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--backend",
                "snowflake",
            ],
            capsys,
        )

    assert code == 0, f"exit code {code}, stderr: {err!r}"
    assert "DataComPy Comparison" in out


def test_snowflake_two_part_ref_expanded(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """2-part schema.table ref is expanded to db.schema.table using session context."""
    mock_compare = MagicMock()
    mock_compare.build_report_data.return_value = MagicMock(
        render=MagicMock(return_value="DataComPy Comparison\nMatch: True"),
        row_summary=MagicMock(unequal_rows=0),
    )
    mock_compare.matches.return_value = True

    mock_session = MagicMock()
    mock_session.get_current_database.return_value = "PROD"

    with (
        patch(
            "datacompy.cli.sessions.get_snowflake_session", return_value=mock_session
        ),
        patch(
            "datacompy.cli.commands.compare.load_snowflake",
            side_effect=lambda session, ref, fmt: ref,
        ),
        patch(
            "datacompy.cli.commands.compare.make_snowflake_compare",
            return_value=mock_compare,
        ),
    ):
        from datacompy.cli.loaders import _expand_table_ref

        assert (
            _expand_table_ref(mock_session, "ANALYTICS.SALES_FACT")
            == "PROD.ANALYTICS.SALES_FACT"
        )
        assert (
            _expand_table_ref(mock_session, "PROD.ANALYTICS.SALES_FACT")
            == "PROD.ANALYTICS.SALES_FACT"
        )


def test_snowflake_two_part_ref_no_db_exits_2(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2-part ref with no current database on the session exits 2 with a clear message."""
    mock_session = MagicMock()
    mock_session.get_current_database.return_value = None

    with patch(
        "datacompy.cli.sessions.get_snowflake_session", return_value=mock_session
    ):
        code, _, err = run(
            [
                "compare",
                "--left",
                "ANALYTICS.SALES_FACT",
                "--right",
                "ANALYTICS.SALES_PREV",
                "--on",
                "ID",
                "--backend",
                "snowflake",
            ],
            capsys,
        )

    assert code == 2
    assert "SNOWFLAKE_DATABASE" in err or "db.schema.table" in err
