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

The comparison test ``test_snowflake_compare_match`` fully mocks the compare
layer so it works without any live Snowflake session.
"""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

snowpark = pytest.importorskip("snowflake.snowpark")

from datacompy.cli.errors import BadArgsError
from datacompy.cli.loaders import _expand_table_ref
from datacompy.cli.sessions import get_snowflake_session


def test_snowflake_missing_account_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    for var in (
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_AUTHENTICATOR",
    ):
        monkeypatch.delenv(var, raising=False)
    code, _, err = cli(
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
        ]
    )
    assert code == 2
    assert (
        "SNOWFLAKE_ACCOUNT" in err
        or "SNOWFLAKE_USER" in err
        or "missing" in err.lower()
    )


def test_snowflake_compare_match(
    tmp_path: Path,
    mock_snowflake_compare: Callable[[bool], MagicMock],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    """Mock out the full compare stack so no live Snowflake session is needed."""
    mock_compare = mock_snowflake_compare(matches=True)

    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,val\n1,a\n")
    right.write_text("id,val\n1,a\n")

    with (
        patch(
            "datacompy.cli.sessions.get_snowflake_session",
            return_value=MagicMock(),
        ),
        patch(
            "datacompy.cli.compare.load_snowflake",
            return_value="DB.SCHEMA.TABLE",
        ),
        patch(
            "datacompy.cli.compare.make_snowflake_compare",
            return_value=mock_compare,
        ),
    ):
        code, out, err = cli(
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
            ]
        )

    assert code == 0, f"exit code {code}, stderr: {err!r}"
    assert "DataComPy Comparison" in out


def test_snowflake_two_part_ref_expanded() -> None:
    """2-part schema.table ref is expanded to db.schema.table using session context."""
    mock_session = MagicMock()
    mock_session.get_current_database.return_value = "PROD"

    assert (
        _expand_table_ref(mock_session, "ANALYTICS.SALES_FACT")
        == "PROD.ANALYTICS.SALES_FACT"
    )
    assert (
        _expand_table_ref(mock_session, "PROD.ANALYTICS.SALES_FACT")
        == "PROD.ANALYTICS.SALES_FACT"
    )


def test_snowflake_two_part_ref_no_db_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    """2-part ref with no current database on the session exits 2 with a clear message."""
    mock_session = MagicMock()
    mock_session.get_current_database.return_value = None

    with patch(
        "datacompy.cli.sessions.get_snowflake_session", return_value=mock_session
    ):
        code, _, err = cli(
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
            ]
        )

    assert code == 2
    assert "SNOWFLAKE_DATABASE" in err or "db.schema.table" in err


def test_missing_snowflake_config_raises_bad_args_error(
    tmp_path: Path,
) -> None:
    """get_snowflake_session must raise BadArgsError (not FileNotFoundError)
    when --snowflake-config points to a non-existent file, so the caller
    gets a message that names both the bad path and the flag to fix."""
    missing = tmp_path / "no_such_conn.json"
    with pytest.raises(BadArgsError, match=r"no_such_conn\.json"):
        get_snowflake_session(config_path=missing)


def test_missing_snowflake_config_file_exits_2_with_clear_message(
    tmp_path: Path,
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    """End-to-end: --snowflake-config pointing to a missing file must exit 2
    with a message that identifies the bad path."""
    missing = tmp_path / "no_such_conn.json"

    code, _, err = cli(
        [
            "compare",
            "--left",
            "DB.SCHEMA.TABLE_L",
            "--right",
            "DB.SCHEMA.TABLE_R",
            "--on",
            "ID",
            "--backend",
            "snowflake",
            "--snowflake-config",
            str(missing),
        ]
    )

    assert code == 2
    assert "no_such_conn.json" in err
