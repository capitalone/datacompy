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
from pathlib import Path
from typing import Generator

import pytest
from datacompy.cli import main


@pytest.fixture()
def tmp_csv(tmp_path: Path) -> Generator[tuple[Path, Path], None, None]:
    """Write two identical CSV files; return (left_path, right_path)."""
    rows = [["id", "val"], ["1", "a"], ["2", "b"], ["3", "c"]]
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    for p in (left, right):
        with p.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
    yield left, right


@pytest.fixture()
def tmp_csv_mismatch(tmp_path: Path) -> Generator[tuple[Path, Path], None, None]:
    """Write two CSVs that differ on row 2."""
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    rows_l = [["id", "val"], ["1", "a"], ["2", "b"]]
    rows_r = [["id", "val"], ["1", "a"], ["2", "X"]]
    for p, rows in ((left, rows_l), (right, rows_r)):
        with p.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
    yield left, right


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:  # type: ignore[type-arg]
    """Call ``main(argv)`` and return ``(exit_code, stdout, stderr)``."""
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err
