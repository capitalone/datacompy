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

"""CLI integration tests for the Pandas backend."""

import csv
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from tests.cli.conftest import run


@pytest.fixture()
def tmp_match(tmp_path: Path) -> tuple[Path, Path]:
    rows = [["id", "val"], ["1", "a"], ["2", "b"]]
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    for p in (left, right):
        with p.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
    return left, right


@pytest.fixture()
def tmp_mismatch(tmp_path: Path) -> tuple[Path, Path]:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    with left.open("w", newline="") as f:
        csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"]])
    with right.open("w", newline="") as f:
        csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "X"]])
    return left, right


def test_match_exits_0(
    tmp_match: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_match
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
            "pandas",
        ],
        capsys,
    )
    assert code == 0
    assert "DataComPy Comparison" in out


def test_mismatch_exits_1(
    tmp_mismatch: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_mismatch
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
            "pandas",
        ],
        capsys,
    )
    assert code == 1


def test_json_output_is_valid(
    tmp_mismatch: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_mismatch
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
            "pandas",
            "--json",
        ],
        capsys,
    )
    assert code == 1
    data = json.loads(out)
    assert "row_summary" in data
    assert data["row_summary"]["unequal_rows"] == 1


def test_quiet_suppresses_output(
    tmp_match: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_match
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
            "pandas",
            "--quiet",
        ],
        capsys,
    )
    assert code == 0
    assert out == ""


def test_max_unequal_rows_threshold_fail(
    tmp_mismatch: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_mismatch
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
            "pandas",
            "--max-unequal-rows",
            "0",
        ],
        capsys,
    )
    assert code == 1


def test_max_unequal_rows_threshold_pass(
    tmp_mismatch: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_mismatch
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
            "pandas",
            "--max-unequal-rows",
            "9999",
        ],
        capsys,
    )
    assert code == 0


def test_missing_left_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    right = tmp_path / "right.csv"
    right.write_text("id,val\n1,a\n")
    code, _, err = run(
        [
            "compare",
            "--left",
            str(tmp_path / "missing.csv"),
            "--right",
            str(right),
            "--on",
            "id",
            "--backend",
            "pandas",
        ],
        capsys,
    )
    assert code == 2
    assert "missing.csv" in err or "not found" in err.lower()


def test_on_index_with_polars_exits_2(
    tmp_match: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_match
    code, _, err = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on-index",
            "--backend",
            "polars",
        ],
        capsys,
    )
    assert code == 2
    assert "--on-index" in err


def test_on_index_and_on_mutually_exclusive_exits_2(
    tmp_match: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_match
    code, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--on-index",
            "--backend",
            "pandas",
        ],
        capsys,
    )
    assert code == 2


def test_on_index_pandas_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"val": ["a", "b"]}).to_csv(left, index=False)
    pd.DataFrame({"val": ["a", "b"]}).to_csv(right, index=False)
    code, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on-index",
            "--backend",
            "pandas",
        ],
        capsys,
    )
    assert code == 0


def test_parquet_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    table = pa.table({"id": [1, 2], "val": ["a", "b"]})
    pq.write_table(table, left)
    pq.write_table(table, right)
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
            "pandas",
        ],
        capsys,
    )
    assert code == 0


def test_unknown_extension_without_format_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "data.xyz"
    right = tmp_path / "data2.xyz"
    left.write_text("id,val\n1,a\n")
    right.write_text("id,val\n1,a\n")
    code, _, err = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--backend",
            "pandas",
        ],
        capsys,
    )
    assert code == 2
    assert ".xyz" in err


def test_pandas_no_join_key_exits_2(
    tmp_match: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_match
    code, _, err = run(
        ["compare", "--left", str(left), "--right", str(right), "--backend", "pandas"],
        capsys,
    )
    assert code == 2
    assert "--on" in err


def test_negative_max_unequal_rows_exits_2(
    tmp_match: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_match
    code, _, err = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--backend",
            "pandas",
            "--max-unequal-rows",
            "-1",
        ],
        capsys,
    )
    assert code == 2
    assert "non-negative" in err


def test_csv_delimiter_semicolon(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    for p in (left, right):
        p.write_text("id;val\n1;a\n2;b\n")
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
            "pandas",
            "--csv-delimiter",
            ";",
        ],
        capsys,
    )
    assert code == 0


def test_txt_extension_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "data.txt"
    right = tmp_path / "data2.txt"
    for p in (left, right):
        p.write_text("id,val\n1,a\n")
    code, _, err = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--backend",
            "pandas",
        ],
        capsys,
    )
    assert code == 2
    assert ".txt" in err


def test_df_names_default_to_stem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "sales_before.csv"
    right = tmp_path / "sales_after.csv"
    for p in (left, right):
        with p.open("w", newline="") as f:
            csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"]])
    _, out, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--backend",
            "pandas",
        ],
        capsys,
    )
    assert "sales_before" in out
    assert "sales_after" in out
