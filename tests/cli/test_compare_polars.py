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

"""CLI integration tests for the Polars backend (default)."""

import csv
import json
from pathlib import Path

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


def test_match_exits_0_default_backend(
    tmp_match: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Polars is the default backend — no --backend flag needed."""
    left, right = tmp_match
    code, out, _ = run(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id"], capsys
    )
    assert code == 0
    assert "DataComPy Comparison" in out


def test_mismatch_exits_1(
    tmp_mismatch: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_mismatch
    code, _, _ = run(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id"], capsys
    )
    assert code == 1


def test_json_output(
    tmp_mismatch: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_mismatch
    code, out, _ = run(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id", "--json"],
        capsys,
    )
    assert code == 1
    data = json.loads(out)
    assert data["row_summary"]["unequal_rows"] == 1


def test_missing_on_exits_2(
    tmp_match: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_match
    code, _, err = run(["compare", "--left", str(left), "--right", str(right)], capsys)
    assert code == 2
    assert "--on" in err


def test_df_names_appear_in_output(
    tmp_mismatch: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    left, right = tmp_mismatch
    _, out, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--df1-name",
            "before",
            "--df2-name",
            "after",
        ],
        capsys,
    )
    assert "before" in out
    assert "after" in out


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
        ],
        capsys,
    )
    assert code == 2
    assert "missing.csv" in err or "not found" in err.lower()


def test_max_unequal_rows_counts_unique_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    with left.open("w", newline="") as f:
        csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"], ["3", "c"]])
    right.write_text("id,val\n1,a\n")
    # unequal_rows=0 but df1_unique=2, so total_diff=2 > 0 → exit 1
    code, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--max-unequal-rows",
            "0",
        ],
        capsys,
    )
    assert code == 1


def test_ignore_unique_rows_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    with left.open("w", newline="") as f:
        csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"], ["3", "c"]])
    right.write_text("id,val\n1,a\n")
    # unequal_rows=0, unique rows excluded → total_diff=0 ≤ 0 → exit 0
    code, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--max-unequal-rows",
            "0",
            "--ignore-unique-rows",
        ],
        capsys,
    )
    assert code == 0


def test_max_unequal_rows_with_ignore_extra_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,val\n1,a\n2,b\n")
    right.write_text("id,val,extra\n1,a,x\n2,b,y\n")
    # extra column in right, values match, --ignore-extra-columns → exit 0
    code, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--max-unequal-rows",
            "0",
            "--ignore-extra-columns",
        ],
        capsys,
    )
    assert code == 0


def test_df_names_default_to_stem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "before_snapshot.csv"
    right = tmp_path / "after_snapshot.csv"
    for p in (left, right):
        with p.open("w", newline="") as f:
            csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"]])
    _, out, _ = run(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id"],
        capsys,
    )
    assert "before_snapshot" in out
    assert "after_snapshot" in out


def test_ignore_unique_rows_without_max_unequal_rows_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
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
            "--ignore-unique-rows",
        ],
        capsys,
    )
    assert code == 2
    assert "--max-unequal-rows" in err


# ---------------------------------------------------------------------------
# --abs-tol
# ---------------------------------------------------------------------------


def test_abs_tol_allows_small_numeric_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,val\n1,1.0\n2,2.0\n")
    right.write_text("id,val\n1,1.0\n2,2.005\n")
    # without tolerance the values differ
    code_no_tol, _, _ = run(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id"],
        capsys,
    )
    assert code_no_tol == 1
    # with abs_tol=0.01 the difference of 0.005 is within tolerance
    code_tol, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--abs-tol",
            "0.01",
        ],
        capsys,
    )
    assert code_tol == 0


def test_abs_tol_still_fails_when_difference_exceeds_tolerance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,val\n1,1.0\n2,2.0\n")
    right.write_text("id,val\n1,1.0\n2,2.5\n")
    code, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--abs-tol",
            "0.01",
        ],
        capsys,
    )
    assert code == 1


# ---------------------------------------------------------------------------
# --rel-tol
# ---------------------------------------------------------------------------


def test_rel_tol_allows_small_relative_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    # 100 vs 101 is a 1% relative difference
    left.write_text("id,val\n1,100.0\n")
    right.write_text("id,val\n1,101.0\n")
    code_no_tol, _, _ = run(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id"],
        capsys,
    )
    assert code_no_tol == 1
    code_tol, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--rel-tol",
            "0.02",
        ],
        capsys,
    )
    assert code_tol == 0


def test_rel_tol_still_fails_when_relative_difference_exceeds_tolerance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    # 100 vs 120 is a 20% relative difference
    left.write_text("id,val\n1,100.0\n")
    right.write_text("id,val\n1,120.0\n")
    code, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--rel-tol",
            "0.02",
        ],
        capsys,
    )
    assert code == 1


# ---------------------------------------------------------------------------
# --ignore-spaces
# ---------------------------------------------------------------------------


def test_ignore_spaces_matches_values_differing_only_in_whitespace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,val\n1,hello\n2,world\n")
    right.write_text("id,val\n1,  hello  \n2,  world  \n")
    code_no_flag, _, _ = run(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id"],
        capsys,
    )
    assert code_no_flag == 1
    code_flag, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--ignore-spaces",
        ],
        capsys,
    )
    assert code_flag == 0


def test_ignore_spaces_still_fails_on_non_whitespace_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,val\n1,hello\n")
    right.write_text("id,val\n1,world\n")
    code, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--ignore-spaces",
        ],
        capsys,
    )
    assert code == 1


# ---------------------------------------------------------------------------
# --ignore-case
# ---------------------------------------------------------------------------


def test_ignore_case_matches_values_differing_only_in_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,val\n1,Hello\n2,World\n")
    right.write_text("id,val\n1,HELLO\n2,WORLD\n")
    code_no_flag, _, _ = run(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id"],
        capsys,
    )
    assert code_no_flag == 1
    code_flag, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--ignore-case",
        ],
        capsys,
    )
    assert code_flag == 0


def test_ignore_case_still_fails_on_non_case_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,val\n1,hello\n")
    right.write_text("id,val\n1,world\n")
    code, _, _ = run(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--ignore-case",
        ],
        capsys,
    )
    assert code == 1
