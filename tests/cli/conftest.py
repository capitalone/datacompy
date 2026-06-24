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

"""Shared fixtures for CLI tests."""

import csv
from collections.abc import Callable, Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from datacompy.cli import main


@pytest.fixture()
def cli(
    capsys: pytest.CaptureFixture[str],
) -> Callable[[list[str]], tuple[int, str, str]]:
    """Return a callable that runs ``main(argv)`` and returns ``(exit_code, stdout, stderr)``."""

    def _run(argv: list[str]) -> tuple[int, str, str]:
        try:
            code = main(argv)
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else 2
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return _run


@pytest.fixture()
def csv_match(tmp_path: Path) -> Generator[tuple[Path, Path], None, None]:
    """Yield two identical 2-row CSVs ``(left, right)``."""
    rows = [["id", "val"], ["1", "a"], ["2", "b"]]
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    for p in (left, right):
        with p.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
    yield left, right


@pytest.fixture()
def csv_mismatch(tmp_path: Path) -> Generator[tuple[Path, Path], None, None]:
    """Yield two 2-row CSVs that differ on row 2 ``(left, right)``."""
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    with left.open("w", newline="") as f:
        csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"]])
    with right.open("w", newline="") as f:
        csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "X"]])
    yield left, right


@pytest.fixture()
def csv_pair(
    tmp_path: Path,
) -> Callable[[str, str], tuple[Path, Path]]:
    """Return a factory that writes two CSV strings to ``left.csv`` / ``right.csv``."""

    def _make(left_text: str, right_text: str) -> tuple[Path, Path]:
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        left.write_text(left_text)
        right.write_text(right_text)
        return left, right

    return _make


@pytest.fixture()
def tsv_pair(
    tmp_path: Path,
) -> Callable[[list[list[str]], list[list[str]]], tuple[Path, Path]]:
    """Return a factory that writes two TSV files to ``left.csv`` / ``right.csv``."""

    def _make(
        left_rows: list[list[str]], right_rows: list[list[str]]
    ) -> tuple[Path, Path]:
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        for path, rows in ((left, left_rows), (right, right_rows)):
            with path.open("w", newline="") as f:
                csv.writer(f, delimiter="\t").writerows(rows)
        return left, right

    return _make


@pytest.fixture()
def mock_snowflake_session() -> MagicMock:
    """Return a MagicMock Snowpark session with db=PROD, schema=PUBLIC."""
    sess = MagicMock()
    sess.get_current_database.return_value = "PROD"
    sess.get_current_schema.return_value = "PUBLIC"
    sess.write_pandas.return_value = None
    return sess


@pytest.fixture()
def mock_snowflake_compare() -> Callable[[bool], MagicMock]:
    """Return a factory that produces a pre-configured ``MagicMock`` compare object.

    Usage::

        def test_something(mock_snowflake_compare, cli):
            mock_compare = mock_snowflake_compare(matches=True)
            with patch("datacompy.cli.compare.make_snowflake_compare", return_value=mock_compare):
                ...
    """

    def _factory(matches: bool = True) -> MagicMock:
        mock_compare = MagicMock()
        mock_compare.build_report_data.return_value = MagicMock(
            render=MagicMock(
                return_value="DataComPy Comparison\nMatch: " + str(matches)
            ),
            row_summary=MagicMock(unequal_rows=0 if matches else 1),
        )
        mock_compare.matches.return_value = matches
        return mock_compare

    return _factory
