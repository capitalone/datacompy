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

"""Tests for the CLI argument parser shape and defaults."""

import pytest
from datacompy.cli.parser import build_parser


def test_compare_defaults() -> None:
    p = build_parser()
    args = p.parse_args(
        ["compare", "--left", "a.csv", "--right", "b.csv", "--on", "id"]
    )
    assert args.backend == "polars"
    assert args.abs_tol == 0.0
    assert args.rel_tol == 0.0
    assert args.ignore_spaces is False
    assert args.ignore_case is False
    assert args.ignore_extra_columns is False
    assert args.cast_column_names_lower is True
    assert args.df1_name is None  # resolved to stem in to_compare_args
    assert args.df2_name is None  # resolved to stem in to_compare_args
    assert args.ignore_unique_rows is False
    assert args.csv_delimiter == ","
    assert args.sample_count == 10
    assert args.column_count == 10
    assert args.max_unequal_rows is None
    assert args.json is False
    assert args.quiet is False


def test_compare_backend_choices() -> None:
    p = build_parser()
    for backend in ("pandas", "polars", "spark", "snowflake"):
        args = p.parse_args(
            [
                "compare",
                "--left",
                "a.csv",
                "--right",
                "b.csv",
                "--on",
                "id",
                "--backend",
                backend,
            ]
        )
        assert args.backend == backend


def test_compare_invalid_backend_exits() -> None:
    p = build_parser()
    with pytest.raises(SystemExit) as exc:
        p.parse_args(
            [
                "compare",
                "--left",
                "a.csv",
                "--right",
                "b.csv",
                "--on",
                "id",
                "--backend",
                "duckdb",
            ]
        )
    assert exc.value.code == 2


def test_compare_format_choices() -> None:
    p = build_parser()
    for fmt in ("csv", "parquet", "json"):
        args = p.parse_args(
            [
                "compare",
                "--left",
                "a.csv",
                "--right",
                "b.csv",
                "--on",
                "id",
                "--format",
                fmt,
            ]
        )
        assert args.format == fmt


def test_compare_multi_on_columns() -> None:
    p = build_parser()
    args = p.parse_args(
        ["compare", "--left", "a.csv", "--right", "b.csv", "--on", "id", "--on", "date"]
    )
    assert args.on == ["id", "date"]


def test_compare_on_index_flag() -> None:
    p = build_parser()
    args = p.parse_args(
        [
            "compare",
            "--left",
            "a.csv",
            "--right",
            "b.csv",
            "--on-index",
            "--backend",
            "pandas",
        ]
    )
    assert args.on_index is True
    assert args.on is None


def test_compare_version(capsys: pytest.CaptureFixture[str]) -> None:
    p = build_parser()
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "datacompy" in out
