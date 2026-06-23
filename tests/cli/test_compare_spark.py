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

"""CLI integration tests for the Spark backend.

Skipped automatically when PySpark is not installed.
"""

import csv
from pathlib import Path

import pytest

pyspark = pytest.importorskip("pyspark")

from tests.cli.conftest import run


@pytest.fixture()
def tmp_match_csv(tmp_path: Path) -> tuple[Path, Path]:
    rows = [["id", "val"], ["1", "a"], ["2", "b"]]
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    for p in (left, right):
        with p.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
    return left, right


@pytest.fixture()
def tmp_mismatch_csv(tmp_path: Path) -> tuple[Path, Path]:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    with left.open("w", newline="") as f:
        csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"]])
    with right.open("w", newline="") as f:
        csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "X"]])
    return left, right


def test_spark_match_exits_0(
    tmp_match_csv: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_match_csv
    code, out, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--backend",
            "spark",
        ],
        capsys,
    )
    assert code == 0
    assert "DataComPy Comparison" in out


def test_spark_mismatch_exits_1(
    tmp_mismatch_csv: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_mismatch_csv
    code, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--backend",
            "spark",
        ],
        capsys,
    )
    assert code == 1
