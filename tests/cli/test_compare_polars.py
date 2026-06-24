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
from collections.abc import Callable
from pathlib import Path

import pytest


def test_match_exits_0_default_backend(
    csv_match: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    """Polars is the default backend -- no --backend flag needed."""
    left, right = csv_match
    code, out, _ = cli(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id"]
    )
    assert code == 0
    assert "DataComPy Comparison" in out


def test_mismatch_exits_1(
    csv_mismatch: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_mismatch
    code, _, _ = cli(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id"]
    )
    assert code == 1


def test_json_output(
    csv_mismatch: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_mismatch
    code, out, _ = cli(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id", "--json"]
    )
    assert code == 1
    data = json.loads(out)
    assert data["row_summary"]["unequal_rows"] == 1


def test_missing_on_exits_2(
    csv_match: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_match
    code, _, err = cli(["compare", "--left", str(left), "--right", str(right)])
    assert code == 2
    assert "--on" in err


def test_df_names_appear_in_output(
    csv_mismatch: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_mismatch
    _, out, _ = cli(
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
        ]
    )
    assert "before" in out
    assert "after" in out


def test_missing_left_exits_2(
    tmp_path: Path,
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    right = tmp_path / "right.csv"
    right.write_text("id,val\n1,a\n")
    code, _, err = cli(
        [
            "compare",
            "--left",
            str(tmp_path / "missing.csv"),
            "--right",
            str(right),
            "--on",
            "id",
        ]
    )
    assert code == 2
    assert "missing.csv" in err or "not found" in err.lower()


def test_max_unequal_rows_counts_unique_rows(
    tmp_path: Path,
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    with left.open("w", newline="") as f:
        csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"], ["3", "c"]])
    right.write_text("id,val\n1,a\n")
    # unequal_rows=0 but df1_unique=2, so total_diff=2 > 0 → exit 1
    code, _, _ = cli(
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
        ]
    )
    assert code == 1


def test_ignore_unique_rows_flag(
    tmp_path: Path,
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    with left.open("w", newline="") as f:
        csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"], ["3", "c"]])
    right.write_text("id,val\n1,a\n")
    # unequal_rows=0, unique rows excluded → total_diff=0 ≤ 0 → exit 0
    code, _, _ = cli(
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
        ]
    )
    assert code == 0


def test_max_unequal_rows_with_ignore_extra_columns(
    csv_pair: Callable[[str, str], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_pair("id,val\n1,a\n2,b\n", "id,val,extra\n1,a,x\n2,b,y\n")
    # extra column in right, values match, --ignore-extra-columns → exit 0
    code, _, _ = cli(
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
        ]
    )
    assert code == 0


def test_df_names_default_to_stem(
    tmp_path: Path,
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left = tmp_path / "before_snapshot.csv"
    right = tmp_path / "after_snapshot.csv"
    for p in (left, right):
        with p.open("w", newline="") as f:
            csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"]])
    _, out, _ = cli(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id"]
    )
    assert "before_snapshot" in out
    assert "after_snapshot" in out


def test_ignore_unique_rows_without_max_unequal_rows_exits_2(
    csv_pair: Callable[[str, str], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_pair("id,val\n1,a\n", "id,val\n1,a\n")
    code, _, err = cli(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--ignore-unique-rows",
        ]
    )
    assert code == 2
    assert "--max-unequal-rows" in err


# ---------------------------------------------------------------------------
# Tolerance and string-normalisation flags
#
# Each case: without the flag the comparison fails (exit 1); with it, passes.
# A separate "still fails" case verifies the flag does not mask real diffs.
# ---------------------------------------------------------------------------

_FLAG_PASS_CASES = [
    pytest.param(
        "id,val\n1,1.0\n2,2.0\n",
        "id,val\n1,1.0\n2,2.005\n",
        ["--abs-tol", "0.01"],
        id="abs_tol_within_tolerance",
    ),
    pytest.param(
        "id,val\n1,100.0\n",
        "id,val\n1,101.0\n",
        ["--rel-tol", "0.02"],
        id="rel_tol_within_tolerance",
    ),
    pytest.param(
        "id,val\n1,hello\n2,world\n",
        "id,val\n1,  hello  \n2,  world  \n",
        ["--ignore-spaces"],
        id="ignore_spaces_whitespace_only_diff",
    ),
    pytest.param(
        "id,val\n1,Hello\n2,World\n",
        "id,val\n1,HELLO\n2,WORLD\n",
        ["--ignore-case"],
        id="ignore_case_case_only_diff",
    ),
]


@pytest.mark.parametrize("left_csv,right_csv,flag", _FLAG_PASS_CASES)
def test_flag_turns_failing_comparison_into_pass(
    left_csv: str,
    right_csv: str,
    flag: list[str],
    csv_pair: Callable[[str, str], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_pair(left_csv, right_csv)
    base_argv = ["compare", "--left", str(left), "--right", str(right), "--on", "id"]
    code_no_flag, _, _ = cli(base_argv)
    assert code_no_flag == 1
    code_flag, _, _ = cli([*base_argv, *flag])
    assert code_flag == 0


_FLAG_STILL_FAIL_CASES = [
    pytest.param(
        "id,val\n1,1.0\n2,2.0\n",
        "id,val\n1,1.0\n2,2.5\n",
        ["--abs-tol", "0.01"],
        id="abs_tol_exceeds_tolerance",
    ),
    pytest.param(
        "id,val\n1,100.0\n",
        "id,val\n1,120.0\n",
        ["--rel-tol", "0.02"],
        id="rel_tol_exceeds_tolerance",
    ),
    pytest.param(
        "id,val\n1,hello\n",
        "id,val\n1,world\n",
        ["--ignore-spaces"],
        id="ignore_spaces_non_whitespace_diff",
    ),
    pytest.param(
        "id,val\n1,hello\n",
        "id,val\n1,world\n",
        ["--ignore-case"],
        id="ignore_case_non_case_diff",
    ),
]


@pytest.mark.parametrize("left_csv,right_csv,flag", _FLAG_STILL_FAIL_CASES)
def test_flag_does_not_mask_real_difference(
    left_csv: str,
    right_csv: str,
    flag: list[str],
    csv_pair: Callable[[str, str], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_pair(left_csv, right_csv)
    code, _, _ = cli(
        ["compare", "--left", str(left), "--right", str(right), "--on", "id", *flag]
    )
    assert code == 1
