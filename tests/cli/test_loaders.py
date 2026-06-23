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

"""Unit tests for datacompy/cli/loaders.py and backends._default_name / _unescape_delimiter.

Covers:
  - Issue 1: _default_name produces correct labels for file paths and Snowflake table refs
  - Issue 2: load_pandas wraps all I/O errors (including corrupt files) as LoadError
  - Issue 3: _unescape_delimiter converts \\t / \\n / \\r escape sequences
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from datacompy.cli.backends import _default_name, _unescape_delimiter
from datacompy.cli.errors import LoadError
from datacompy.cli.loaders import (
    is_snowflake_ref,
    load_pandas,
    load_polars,
)

# ---------------------------------------------------------------------------
# is_snowflake_ref — the shared predicate used by both loaders and backends
# ---------------------------------------------------------------------------


class TestIsSnowflakeRef:
    def test_three_part_ref(self) -> None:
        assert is_snowflake_ref("PROD.ANALYTICS.SALES_FACT") is True

    def test_three_part_lowercase(self) -> None:
        assert is_snowflake_ref("mydb.reporting.orders") is True

    def test_two_part_ref(self) -> None:
        assert is_snowflake_ref("ANALYTICS.SALES_FACT") is True

    def test_csv_file_is_not_a_ref(self) -> None:
        assert is_snowflake_ref("sales_data.csv") is False

    def test_parquet_file_is_not_a_ref(self) -> None:
        assert is_snowflake_ref("data.parquet") is False

    def test_json_file_is_not_a_ref(self) -> None:
        assert is_snowflake_ref("events.json") is False

    def test_txt_file_is_not_a_ref(self) -> None:
        assert is_snowflake_ref("data.txt") is False

    def test_gz_file_is_not_a_ref(self) -> None:
        assert is_snowflake_ref("data.csv.gz") is False

    def test_zip_file_is_not_a_ref(self) -> None:
        assert is_snowflake_ref("archive.zip") is False

    def test_s3_uri_is_not_a_ref(self) -> None:
        assert is_snowflake_ref("s3://bucket/prefix/file.csv") is False

    def test_relative_path_is_not_a_ref(self) -> None:
        assert is_snowflake_ref("path/to/file.csv") is False

    def test_windows_path_is_not_a_ref(self) -> None:
        assert is_snowflake_ref("C:\\data\\file.csv") is False

    def test_single_word_is_not_a_ref(self) -> None:
        assert is_snowflake_ref("mytable") is False


# ---------------------------------------------------------------------------
# Issue 1 — _default_name: correct labels for files and Snowflake table refs
# ---------------------------------------------------------------------------


class TestDefaultName:
    def test_csv_file_returns_stem(self) -> None:
        assert _default_name("sales_data.csv") == "sales_data"

    def test_parquet_file_returns_stem(self) -> None:
        assert _default_name("archive/orders_2024.parquet") == "orders_2024"

    def test_s3_uri_returns_stem(self) -> None:
        assert _default_name("s3://bucket/prefix/snapshot.csv") == "snapshot"

    def test_three_part_snowflake_ref_returns_table_name(self) -> None:
        assert _default_name("PROD.ANALYTICS.SALES_FACT") == "SALES_FACT"

    def test_three_part_lowercase_snowflake_ref(self) -> None:
        assert _default_name("mydb.reporting.orders_2024") == "orders_2024"

    def test_two_part_snowflake_ref_returns_table_name(self) -> None:
        assert _default_name("ANALYTICS.SALES_FACT") == "SALES_FACT"

    def test_same_schema_different_tables_produce_distinct_names(self) -> None:
        """Regression: Path.stem produced 'PROD.ANALYTICS' for both sides."""
        assert _default_name("PROD.ANALYTICS.SALES_FACT") != _default_name(
            "PROD.ANALYTICS.SALES_PREV"
        )
        assert _default_name("PROD.ANALYTICS.SALES_FACT") == "SALES_FACT"
        assert _default_name("PROD.ANALYTICS.SALES_PREV") == "SALES_PREV"

    def test_explicit_name_overrides_default(self, tmp_path: Path) -> None:
        """to_compare_args respects --df1-name / --df2-name when provided."""
        import argparse

        from datacompy.cli.backends import to_compare_args

        ns = argparse.Namespace(
            left="PROD.ANALYTICS.SALES_FACT",
            right="PROD.ANALYTICS.SALES_PREV",
            df1_name="before",
            df2_name="after",
            format=None,
            on=["ID"],
            on_index=False,
            backend="snowflake",
            abs_tol=0.0,
            rel_tol=0.0,
            ignore_spaces=False,
            ignore_case=False,
            ignore_extra_columns=False,
            ignore_unique_rows=False,
            cast_column_names_lower=True,
            csv_delimiter=",",
            sample_count=10,
            column_count=10,
            max_unequal_rows=None,
            json=False,
            quiet=False,
            spark_app_name="datacompy-cli",
            snowflake_config=None,
        )
        args = to_compare_args(ns)
        assert args.df1_name == "before"
        assert args.df2_name == "after"


# ---------------------------------------------------------------------------
# Issue 1 (continued) — integration: Snowflake report uses correct table names
# ---------------------------------------------------------------------------


class TestSnowflakeDfNameIntegration:
    """Verify the label reaches the report, not just _default_name in isolation."""

    def test_three_part_refs_produce_distinct_df_names_in_report(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from unittest.mock import patch

        from tests.cli.conftest import run

        mock_report = MagicMock()
        mock_report.render.return_value = (
            "DataComPy Comparison\nSALES_FACT vs SALES_PREV"
        )
        mock_report.row_summary = MagicMock(unequal_rows=0)
        mock_compare = MagicMock()
        mock_compare.build_report_data.return_value = mock_report
        mock_compare.matches.return_value = True

        with (
            patch(
                "datacompy.cli.sessions.get_snowflake_session",
                return_value=MagicMock(),
            ),
            patch(
                "datacompy.cli.commands.compare.load_snowflake",
                side_effect=lambda s, ref, fmt: ref,
            ),
            patch(
                "datacompy.cli.commands.compare.make_snowflake_compare",
                return_value=mock_compare,
            ) as mock_make,
        ):
            run(
                [
                    "compare",
                    "--left",
                    "PROD.ANALYTICS.SALES_FACT",
                    "--right",
                    "PROD.ANALYTICS.SALES_PREV",
                    "--on",
                    "ID",
                    "--backend",
                    "snowflake",
                ],
                capsys,
            )

        call_args = mock_make.call_args
        built_args = call_args[0][0]
        assert built_args.df1_name == "SALES_FACT"
        assert built_args.df2_name == "SALES_PREV"


# ---------------------------------------------------------------------------
# Issue 2 — load_pandas wraps all errors as LoadError (not just FileNotFoundError)
# ---------------------------------------------------------------------------


class TestLoadPandasErrorHandling:
    def test_missing_file_raises_load_error(self, tmp_path: Path) -> None:
        with pytest.raises(LoadError, match="not found"):
            load_pandas(str(tmp_path / "nonexistent.csv"), "csv")

    def test_corrupt_parquet_raises_load_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.parquet"
        bad.write_bytes(b"this is not a valid parquet file")
        with pytest.raises(LoadError, match=r"bad\.parquet"):
            load_pandas(str(bad), "parquet")

    def test_corrupt_json_raises_load_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json{{{{")
        with pytest.raises(LoadError, match=r"bad\.json"):
            load_pandas(str(bad), "json")

    def test_corrupt_csv_with_bad_delimiter_raises_load_error(
        self, tmp_path: Path
    ) -> None:
        """A delimiter that causes a pandas parse failure should be a LoadError."""
        bad = tmp_path / "data.csv"
        bad.write_text("id,val\n1,a\n2,b\n")
        # Reading a comma-delimited file with a pipe delimiter produces a
        # single-column frame — not an error. Use a clearly unreadable encoding
        # scenario instead: a binary file masquerading as CSV.
        binary = tmp_path / "binary.csv"
        binary.write_bytes(bytes(range(256)))
        with pytest.raises(LoadError, match=r"binary\.csv"):
            load_pandas(str(binary), "csv")

    def test_error_message_contains_path(self, tmp_path: Path) -> None:
        bad = tmp_path / "corrupt.parquet"
        bad.write_bytes(b"garbage")
        with pytest.raises(LoadError) as exc_info:
            load_pandas(str(bad), "parquet")
        assert "corrupt.parquet" in str(exc_info.value)

    def test_valid_csv_loads_successfully(self, tmp_path: Path) -> None:
        p = tmp_path / "good.csv"
        p.write_text("id,val\n1,a\n2,b\n")
        df = load_pandas(str(p), "csv")
        assert list(df.columns) == ["id", "val"]
        assert len(df) == 2


class TestLoadPolarsErrorHandlingConsistency:
    """Polars already caught Exception broadly; confirm it still does and matches pandas."""

    def test_corrupt_parquet_raises_load_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.parquet"
        bad.write_bytes(b"garbage")
        with pytest.raises(LoadError, match=r"bad\.parquet"):
            load_polars(str(bad), "parquet")

    def test_missing_file_raises_load_error(self, tmp_path: Path) -> None:
        with pytest.raises(LoadError, match="not found"):
            load_polars(str(tmp_path / "nope.csv"), "csv")


# ---------------------------------------------------------------------------
# Issue 3 — _unescape_delimiter converts escape sequences
# ---------------------------------------------------------------------------


class TestUnescapeDelimiter:
    def test_tab_escape_sequence(self) -> None:
        assert _unescape_delimiter("\\t") == "\t"

    def test_newline_escape_sequence(self) -> None:
        assert _unescape_delimiter("\\n") == "\n"

    def test_carriage_return_escape_sequence(self) -> None:
        assert _unescape_delimiter("\\r") == "\r"

    def test_real_tab_unchanged(self) -> None:
        assert _unescape_delimiter("\t") == "\t"

    def test_semicolon_unchanged(self) -> None:
        assert _unescape_delimiter(";") == ";"

    def test_comma_unchanged(self) -> None:
        assert _unescape_delimiter(",") == ","

    def test_pipe_unchanged(self) -> None:
        assert _unescape_delimiter("|") == "|"


class TestCsvDelimiterIntegration:
    """End-to-end: --csv-delimiter '\\t' parses TSV files correctly via the CLI."""

    def _write_tsv(self, path: Path, rows: list[list[str]]) -> None:
        with path.open("w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(rows)

    def test_backslash_t_string_parses_tsv_pandas(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tests.cli.conftest import run

        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        for p in (left, right):
            self._write_tsv(p, [["id", "val"], ["1", "a"], ["2", "b"]])

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
                "--csv-delimiter",
                "\\t",  # literal backslash-t as a user would type it
            ],
            capsys,
        )
        assert code == 0, f"expected match, stderr: {err}"

    def test_backslash_t_string_parses_tsv_polars(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tests.cli.conftest import run

        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        for p in (left, right):
            self._write_tsv(p, [["id", "val"], ["1", "a"], ["2", "b"]])

        code, _, err = run(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--csv-delimiter",
                "\\t",
            ],
            capsys,
        )
        assert code == 0, f"expected match, stderr: {err}"

    def test_backslash_t_detects_mismatch(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tests.cli.conftest import run

        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        self._write_tsv(left, [["id", "val"], ["1", "a"], ["2", "b"]])
        self._write_tsv(right, [["id", "val"], ["1", "a"], ["2", "X"]])

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
                "\\t",
            ],
            capsys,
        )
        assert code == 1

    def test_real_tab_also_works(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A real tab character passed programmatically should still work."""
        from tests.cli.conftest import run

        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        for p in (left, right):
            self._write_tsv(p, [["id", "val"], ["1", "a"], ["2", "b"]])

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
                "--csv-delimiter",
                "\t",  # actual tab character
            ],
            capsys,
        )
        assert code == 0, f"expected match, stderr: {err}"
