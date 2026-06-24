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
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def test_match_exits_0(
    csv_match: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_match
    code, out, _ = cli(
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
        ]
    )
    assert code == 0
    assert "DataComPy Comparison" in out


def test_mismatch_exits_1(
    csv_mismatch: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_mismatch
    code, _, _ = cli(
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
        ]
    )
    assert code == 1


def test_json_output_is_valid(
    csv_mismatch: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_mismatch
    code, out, _ = cli(
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
        ]
    )
    assert code == 1
    data = json.loads(out)
    assert "row_summary" in data
    assert data["row_summary"]["unequal_rows"] == 1


def test_quiet_suppresses_output(
    csv_match: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_match
    code, out, _ = cli(
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
        ]
    )
    assert code == 0
    assert out == ""


def test_max_unequal_rows_threshold_fail(
    csv_mismatch: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_mismatch
    code, _, _ = cli(
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
        ]
    )
    assert code == 1


def test_max_unequal_rows_threshold_pass(
    csv_mismatch: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_mismatch
    code, _, _ = cli(
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
        ]
    )
    assert code == 0


def test_missing_left_exits_2(
    csv_pair: Callable[[str, str], tuple[Path, Path]],
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
            "--backend",
            "pandas",
        ]
    )
    assert code == 2
    assert "missing.csv" in err or "not found" in err.lower()


def test_on_index_with_polars_exits_2(
    csv_match: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_match
    code, _, err = cli(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on-index",
            "--backend",
            "polars",
        ]
    )
    assert code == 2
    assert "--on-index" in err


def test_on_index_and_on_mutually_exclusive_exits_2(
    csv_match: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_match
    code, _, _ = cli(
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
        ]
    )
    assert code == 2


def test_on_index_pandas_match(
    csv_pair: Callable[[str, str], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    content = pd.DataFrame({"val": ["a", "b"]}).to_csv(index=False)
    left, right = csv_pair(content, content)
    code, _, _ = cli(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on-index",
            "--backend",
            "pandas",
        ]
    )
    assert code == 0


def test_parquet_input(
    tmp_path: Path,
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    table = pa.table({"id": [1, 2], "val": ["a", "b"]})
    pq.write_table(table, left)
    pq.write_table(table, right)
    code, _, _ = cli(
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
        ]
    )
    assert code == 0


def test_unknown_extension_without_format_exits_2(
    csv_pair: Callable[[str, str], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_pair("id,val\n1,a\n", "id,val\n1,a\n")
    left = left.rename(left.parent / "data.xyz")
    right = right.rename(right.parent / "data2.xyz")
    code, _, err = cli(
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
        ]
    )
    assert code == 2
    assert ".xyz" in err


def test_pandas_no_join_key_exits_2(
    csv_match: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_match
    code, _, err = cli(
        ["compare", "--left", str(left), "--right", str(right), "--backend", "pandas"]
    )
    assert code == 2
    assert "--on" in err


def test_negative_max_unequal_rows_exits_2(
    csv_match: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_match
    code, _, err = cli(
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
        ]
    )
    assert code == 2
    assert "non-negative" in err


def test_csv_delimiter_semicolon(
    csv_pair: Callable[[str, str], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_pair("id;val\n1;a\n2;b\n", "id;val\n1;a\n2;b\n")
    code, _, _ = cli(
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
        ]
    )
    assert code == 0


def test_txt_extension_exits_2(
    csv_pair: Callable[[str, str], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_pair("id,val\n1,a\n", "id,val\n1,a\n")
    left = left.rename(left.parent / "data.txt")
    right = right.rename(right.parent / "data2.txt")
    code, _, err = cli(
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
        ]
    )
    assert code == 2
    assert ".txt" in err


def test_df_names_default_to_stem(
    tmp_path: Path,
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left = tmp_path / "sales_before.csv"
    right = tmp_path / "sales_after.csv"
    for p in (left, right):
        with p.open("w", newline="") as f:
            csv.writer(f).writerows([["id", "val"], ["1", "a"], ["2", "b"]])
    _, out, _ = cli(
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
        ]
    )
    assert "sales_before" in out
    assert "sales_after" in out


# ---------------------------------------------------------------------------
# --csv-delimiter validation
# ---------------------------------------------------------------------------


def test_multi_char_delimiter_exits_2(
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
            "--backend",
            "pandas",
            "--csv-delimiter",
            "||",
        ]
    )
    assert code == 2
    assert "csv-delimiter" in err.lower() or "delimiter" in err.lower()


def test_empty_delimiter_exits_2(
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
            "--backend",
            "pandas",
            "--csv-delimiter",
            "",
        ]
    )
    assert code == 2
    assert "csv-delimiter" in err.lower() or "delimiter" in err.lower()


def test_single_char_delimiter_is_accepted(
    csv_pair: Callable[[str, str], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_pair("id|val\n1|a\n2|b\n", "id|val\n1|a\n2|b\n")
    code, _, _ = cli(
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
            "|",
        ]
    )
    assert code == 0


# ---------------------------------------------------------------------------
# --debug flag
# ---------------------------------------------------------------------------


def test_debug_flag_re_raises_cli_error(
    tmp_path: Path,
) -> None:
    """--debug must cause CLIError subclasses to propagate as real exceptions
    rather than being swallowed into a friendly exit-2 message."""
    from datacompy.cli import main
    from datacompy.cli.errors import BadArgsError

    with pytest.raises(BadArgsError):
        main(
            [
                "--debug",
                "compare",
                "--left",
                str(tmp_path / "a.csv"),
                "--right",
                str(tmp_path / "b.csv"),
                "--on",
                "id",
                "--on-index",  # mutually exclusive with --on -- triggers BadArgsError
                "--backend",
                "pandas",
            ]
        )
