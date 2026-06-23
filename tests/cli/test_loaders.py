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

"""Unit tests for datacompy/cli/loaders.py and backends._default_name / _unescape_delimiter."""

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
# is_snowflake_ref
# ---------------------------------------------------------------------------


def test_is_snowflake_ref_three_part() -> None:
    assert is_snowflake_ref("PROD.ANALYTICS.SALES_FACT") is True


def test_is_snowflake_ref_three_part_lowercase() -> None:
    assert is_snowflake_ref("mydb.reporting.orders") is True


def test_is_snowflake_ref_two_part() -> None:
    assert is_snowflake_ref("ANALYTICS.SALES_FACT") is True


def test_is_snowflake_ref_csv_file_is_not_a_ref() -> None:
    assert is_snowflake_ref("sales_data.csv") is False


def test_is_snowflake_ref_parquet_file_is_not_a_ref() -> None:
    assert is_snowflake_ref("data.parquet") is False


def test_is_snowflake_ref_json_file_is_not_a_ref() -> None:
    assert is_snowflake_ref("events.json") is False


def test_is_snowflake_ref_txt_file_is_not_a_ref() -> None:
    assert is_snowflake_ref("data.txt") is False


def test_is_snowflake_ref_gz_file_is_not_a_ref() -> None:
    assert is_snowflake_ref("data.csv.gz") is False


def test_is_snowflake_ref_zip_file_is_not_a_ref() -> None:
    assert is_snowflake_ref("archive.zip") is False


def test_is_snowflake_ref_s3_uri_is_not_a_ref() -> None:
    assert is_snowflake_ref("s3://bucket/prefix/file.csv") is False


def test_is_snowflake_ref_relative_path_is_not_a_ref() -> None:
    assert is_snowflake_ref("path/to/file.csv") is False


def test_is_snowflake_ref_windows_path_is_not_a_ref() -> None:
    assert is_snowflake_ref("C:\\data\\file.csv") is False


def test_is_snowflake_ref_single_word_is_not_a_ref() -> None:
    assert is_snowflake_ref("mytable") is False


# Valid identifiers: numbers and special characters inside segments
def test_is_snowflake_ref_numbers_inside_segment_are_valid() -> None:
    assert is_snowflake_ref("db1.schema2.table3") is True


def test_is_snowflake_ref_dollar_sign_in_segment_is_valid() -> None:
    assert is_snowflake_ref("MY$DB.MY$SCHEMA.MY$TABLE") is True


def test_is_snowflake_ref_underscore_led_segment_is_valid() -> None:
    assert is_snowflake_ref("_db._schema._table") is True


def test_is_snowflake_ref_mixed_case_with_numbers_is_valid() -> None:
    assert is_snowflake_ref("PROD_2024.Analytics.SALES_FACT_V2") is True


# Digit-leading segments — not valid Snowflake identifiers; the current
# regex accepts these because \w includes [0-9].
def test_is_snowflake_ref_digit_led_first_segment_is_not_a_ref() -> None:
    assert is_snowflake_ref("1db.schema.table") is False


def test_is_snowflake_ref_digit_led_second_segment_is_not_a_ref() -> None:
    assert is_snowflake_ref("db.2schema.table") is False


def test_is_snowflake_ref_digit_led_third_segment_is_not_a_ref() -> None:
    assert is_snowflake_ref("db.schema.3table") is False


def test_is_snowflake_ref_all_numeric_two_part_is_not_a_ref() -> None:
    # "1.5" looks like a floating-point literal, not a table ref
    assert is_snowflake_ref("1.5") is False


def test_is_snowflake_ref_all_numeric_three_part_is_not_a_ref() -> None:
    assert is_snowflake_ref("1.2.3") is False


# Version strings and other dot-separated values without extensions
def test_is_snowflake_ref_version_string_without_extension_is_not_a_ref() -> None:
    # e.g. a file named "v1.5" with no extension
    assert is_snowflake_ref("v1.5") is False


# Structure: wrong number of parts
def test_is_snowflake_ref_four_part_is_not_a_ref() -> None:
    assert is_snowflake_ref("a.b.c.d") is False


def test_is_snowflake_ref_empty_string_is_not_a_ref() -> None:
    assert is_snowflake_ref("") is False


def test_is_snowflake_ref_empty_segment_is_not_a_ref() -> None:
    assert is_snowflake_ref("db..table") is False


def test_is_snowflake_ref_leading_dot_is_not_a_ref() -> None:
    assert is_snowflake_ref(".schema.table") is False


def test_is_snowflake_ref_trailing_dot_is_not_a_ref() -> None:
    assert is_snowflake_ref("schema.table.") is False


# Quoted identifiers are not handled by the CLI (the regex does not
# recognise double-quote delimited names).
def test_is_snowflake_ref_quoted_identifier_is_not_a_ref() -> None:
    assert is_snowflake_ref('"My DB"."My Schema"."My Table"') is False


# ---------------------------------------------------------------------------
# _default_name
# ---------------------------------------------------------------------------


def test_default_name_csv_file_returns_stem() -> None:
    assert _default_name("sales_data.csv") == "sales_data"


def test_default_name_parquet_file_returns_stem() -> None:
    assert _default_name("archive/orders_2024.parquet") == "orders_2024"


def test_default_name_s3_uri_returns_stem() -> None:
    assert _default_name("s3://bucket/prefix/snapshot.csv") == "snapshot"


def test_default_name_three_part_snowflake_ref_returns_table_name() -> None:
    assert _default_name("PROD.ANALYTICS.SALES_FACT") == "SALES_FACT"


def test_default_name_three_part_lowercase_snowflake_ref() -> None:
    assert _default_name("mydb.reporting.orders_2024") == "orders_2024"


def test_default_name_two_part_snowflake_ref_returns_table_name() -> None:
    assert _default_name("ANALYTICS.SALES_FACT") == "SALES_FACT"


def test_default_name_same_schema_different_tables_produce_distinct_names() -> None:
    """Regression: Path.stem produced 'PROD.ANALYTICS' for both sides."""
    assert _default_name("PROD.ANALYTICS.SALES_FACT") != _default_name(
        "PROD.ANALYTICS.SALES_PREV"
    )
    assert _default_name("PROD.ANALYTICS.SALES_FACT") == "SALES_FACT"
    assert _default_name("PROD.ANALYTICS.SALES_PREV") == "SALES_PREV"


def test_default_name_explicit_name_overrides_default(tmp_path: Path) -> None:
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
# _default_name integration: Snowflake report uses correct table names
# ---------------------------------------------------------------------------


def test_snowflake_three_part_refs_produce_distinct_df_names_in_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify the label reaches the report, not just _default_name in isolation."""
    from unittest.mock import patch

    from tests.cli.conftest import run

    mock_report = MagicMock()
    mock_report.render.return_value = "DataComPy Comparison\nSALES_FACT vs SALES_PREV"
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
            side_effect=lambda s, ref, fmt, **kw: ref,
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
# load_pandas error handling
# ---------------------------------------------------------------------------


def test_load_pandas_missing_file_raises_load_error(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="not found"):
        load_pandas(str(tmp_path / "nonexistent.csv"), "csv")


def test_load_pandas_corrupt_parquet_raises_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"this is not a valid parquet file")
    with pytest.raises(LoadError, match=r"bad\.parquet"):
        load_pandas(str(bad), "parquet")


def test_load_pandas_corrupt_json_raises_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json{{{{")
    with pytest.raises(LoadError, match=r"bad\.json"):
        load_pandas(str(bad), "json")


def test_load_pandas_binary_csv_raises_load_error(tmp_path: Path) -> None:
    binary = tmp_path / "binary.csv"
    binary.write_bytes(bytes(range(256)))
    with pytest.raises(LoadError, match=r"binary\.csv"):
        load_pandas(str(binary), "csv")


def test_load_pandas_error_message_contains_path(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.parquet"
    bad.write_bytes(b"garbage")
    with pytest.raises(LoadError) as exc_info:
        load_pandas(str(bad), "parquet")
    assert "corrupt.parquet" in str(exc_info.value)


def test_load_pandas_valid_csv_loads_successfully(tmp_path: Path) -> None:
    p = tmp_path / "good.csv"
    p.write_text("id,val\n1,a\n2,b\n")
    df = load_pandas(str(p), "csv")
    assert list(df.columns) == ["id", "val"]
    assert len(df) == 2


# ---------------------------------------------------------------------------
# load_polars error handling
# ---------------------------------------------------------------------------


def test_load_polars_corrupt_parquet_raises_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"garbage")
    with pytest.raises(LoadError, match=r"bad\.parquet"):
        load_polars(str(bad), "parquet")


def test_load_polars_missing_file_raises_load_error(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="not found"):
        load_polars(str(tmp_path / "nope.csv"), "csv")


# ---------------------------------------------------------------------------
# _unescape_delimiter
# ---------------------------------------------------------------------------


def test_unescape_delimiter_tab_escape_sequence() -> None:
    assert _unescape_delimiter("\\t") == "\t"


def test_unescape_delimiter_newline_escape_sequence() -> None:
    assert _unescape_delimiter("\\n") == "\n"


def test_unescape_delimiter_carriage_return_escape_sequence() -> None:
    assert _unescape_delimiter("\\r") == "\r"


def test_unescape_delimiter_real_tab_unchanged() -> None:
    assert _unescape_delimiter("\t") == "\t"


def test_unescape_delimiter_semicolon_unchanged() -> None:
    assert _unescape_delimiter(";") == ";"


def test_unescape_delimiter_comma_unchanged() -> None:
    assert _unescape_delimiter(",") == ","


def test_unescape_delimiter_pipe_unchanged() -> None:
    assert _unescape_delimiter("|") == "|"


# ---------------------------------------------------------------------------
# CSV delimiter integration (end-to-end via CLI)
# ---------------------------------------------------------------------------


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(rows)


def test_csv_delimiter_backslash_t_parses_tsv_pandas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.cli.conftest import run

    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    for p in (left, right):
        _write_tsv(p, [["id", "val"], ["1", "a"], ["2", "b"]])

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
            "\\t",
        ],
        capsys,
    )
    assert code == 0, f"expected match, stderr: {err}"


def test_csv_delimiter_backslash_t_parses_tsv_polars(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.cli.conftest import run

    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    for p in (left, right):
        _write_tsv(p, [["id", "val"], ["1", "a"], ["2", "b"]])

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


def test_csv_delimiter_backslash_t_detects_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.cli.conftest import run

    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_tsv(left, [["id", "val"], ["1", "a"], ["2", "b"]])
    _write_tsv(right, [["id", "val"], ["1", "a"], ["2", "X"]])

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


def test_csv_delimiter_real_tab_also_works(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real tab character passed programmatically should still work."""
    from tests.cli.conftest import run

    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    for p in (left, right):
        _write_tsv(p, [["id", "val"], ["1", "a"], ["2", "b"]])

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
            "\t",
        ],
        capsys,
    )
    assert code == 0, f"expected match, stderr: {err}"


def test_load_snowflake_staging_raises_error_when_no_database(
    tmp_path: Path,
) -> None:
    """load_snowflake with a local CSV and session.get_current_database()=None
    must raise BadArgsError, not return a broken '.' -prefixed table ref."""
    from datacompy.cli.errors import BadArgsError
    from datacompy.cli.loaders import load_snowflake

    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,val\n1,a\n2,b\n")

    mock_session = MagicMock()
    mock_session.get_current_database.return_value = None
    mock_session.get_current_schema.return_value = "PUBLIC"
    mock_session.write_pandas.return_value = None

    with pytest.raises(BadArgsError, match=r"SNOWFLAKE_DATABASE|database"):
        load_snowflake(mock_session, str(csv_file), "csv")


def test_load_snowflake_staging_raises_error_when_no_schema(
    tmp_path: Path,
) -> None:
    """load_snowflake with a local CSV and session.get_current_schema()=None
    must raise BadArgsError, not return a broken ref like 'PROD..DATACOMPY_TMP_...'."""
    from datacompy.cli.errors import BadArgsError
    from datacompy.cli.loaders import load_snowflake

    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,val\n1,a\n2,b\n")

    mock_session = MagicMock()
    mock_session.get_current_database.return_value = "PROD"
    mock_session.get_current_schema.return_value = None
    mock_session.write_pandas.return_value = None

    with pytest.raises(BadArgsError, match=r"SNOWFLAKE_SCHEMA|schema"):
        load_snowflake(mock_session, str(csv_file), "csv")


def test_load_snowflake_staging_returns_valid_ref_when_both_set(
    tmp_path: Path,
) -> None:
    """When db and schema are both available, the returned ref must be
    a fully-qualified 'db.schema.table' string."""
    from datacompy.cli.loaders import load_snowflake

    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,val\n1,a\n2,b\n")

    mock_session = MagicMock()
    mock_session.get_current_database.return_value = "PROD"
    mock_session.get_current_schema.return_value = "PUBLIC"
    mock_session.write_pandas.return_value = None

    ref = load_snowflake(mock_session, str(csv_file), "csv")

    parts = ref.split(".")
    assert len(parts) == 3, f"expected db.schema.table, got {ref!r}"
    assert parts[0] == "PROD"
    assert parts[1] == "PUBLIC"
    assert parts[2].startswith("DATACOMPY_TMP_")


def test_load_snowflake_csv_delimiter_is_honoured_when_staging(
    tmp_path: Path,
) -> None:
    """When a semicolon-delimited CSV is staged, write_pandas must receive
    a DataFrame with two separate columns, not a single column whose name
    contains the delimiter."""
    import pandas as pd
    from datacompy.cli.loaders import load_snowflake

    semi_csv = tmp_path / "data.csv"
    semi_csv.write_text("id;val\n1;a\n2;b\n")

    captured_df: list[pd.DataFrame] = []

    mock_session = MagicMock()
    mock_session.get_current_database.return_value = "PROD"
    mock_session.get_current_schema.return_value = "PUBLIC"

    def _capture_write_pandas(df: pd.DataFrame, **kwargs: object) -> None:
        captured_df.append(df)

    mock_session.write_pandas.side_effect = _capture_write_pandas

    load_snowflake(mock_session, str(semi_csv), "csv", csv_delimiter=";")

    assert len(captured_df) == 1, "write_pandas should have been called once"
    df = captured_df[0]
    assert list(df.columns) == [
        "id",
        "val",
    ], f"Expected ['id', 'val'], got {list(df.columns)!r} — delimiter was ignored"
