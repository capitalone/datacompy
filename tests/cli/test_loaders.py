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

import argparse
import importlib.util
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from datacompy.cli.backends import _default_name, to_compare_args
from datacompy.cli.errors import BadArgsError, LoadError
from datacompy.cli.loaders import (
    is_snowflake_ref,
    load_pandas,
    load_polars,
    load_snowflake,
)
from datacompy.cli.parser import _single_char_delimiter

# ---------------------------------------------------------------------------
# is_snowflake_ref
# ---------------------------------------------------------------------------

_VALID_SNOWFLAKE_REFS = [
    pytest.param("PROD.ANALYTICS.SALES_FACT", id="three_part_upper"),
    pytest.param("mydb.reporting.orders", id="three_part_lowercase"),
    pytest.param("ANALYTICS.SALES_FACT", id="two_part"),
    pytest.param("MY$DB.MY$SCHEMA.MY$TABLE", id="dollar_sign_in_segment"),
    pytest.param("_db._schema._table", id="underscore_led_segment"),
    pytest.param("PROD_2024.Analytics.SALES_FACT_V2", id="mixed_case_with_numbers"),
]


@pytest.mark.parametrize("ref", _VALID_SNOWFLAKE_REFS)
def test_is_snowflake_ref_valid(ref: str) -> None:
    assert is_snowflake_ref(ref) is True


_INVALID_SNOWFLAKE_REFS = [
    pytest.param("sales_data.csv", id="csv_file"),
    pytest.param("s3://bucket/prefix/file.csv", id="s3_uri"),
    pytest.param("path/to/file.csv", id="slash_path"),
    pytest.param('"My DB"."My Schema"."My Table"', id="quoted_identifier"),
    pytest.param("mytable", id="single_segment"),
    pytest.param("a.b.c.d", id="four_part"),
]


@pytest.mark.parametrize("ref", _INVALID_SNOWFLAKE_REFS)
def test_is_snowflake_ref_invalid(ref: str) -> None:
    assert is_snowflake_ref(ref) is False


# ---------------------------------------------------------------------------
# _default_name
# ---------------------------------------------------------------------------

_DEFAULT_NAME_CASES = [
    pytest.param("sales_data.csv", "sales_data", id="csv_file"),
    pytest.param("archive/orders_2024.parquet", "orders_2024", id="parquet_file"),
    pytest.param("s3://bucket/prefix/snapshot.csv", "snapshot", id="s3_uri"),
    pytest.param("PROD.ANALYTICS.SALES_FACT", "SALES_FACT", id="three_part_ref"),
    pytest.param(
        "mydb.reporting.orders_2024", "orders_2024", id="three_part_lowercase_ref"
    ),
    pytest.param("ANALYTICS.SALES_FACT", "SALES_FACT", id="two_part_ref"),
]


@pytest.mark.parametrize("ref,expected", _DEFAULT_NAME_CASES)
def test_default_name(ref: str, expected: str) -> None:
    assert _default_name(ref) == expected


def test_default_name_same_schema_different_tables_produce_distinct_names() -> None:
    """Regression: Path.stem produced 'PROD.ANALYTICS' for both sides."""
    assert _default_name("PROD.ANALYTICS.SALES_FACT") != _default_name(
        "PROD.ANALYTICS.SALES_PREV"
    )
    assert _default_name("PROD.ANALYTICS.SALES_FACT") == "SALES_FACT"
    assert _default_name("PROD.ANALYTICS.SALES_PREV") == "SALES_PREV"


def test_default_name_explicit_name_overrides_default(tmp_path: Path) -> None:
    """to_compare_args respects --df1-name / --df2-name when provided."""
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
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    """Verify the label reaches the report, not just _default_name in isolation."""
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
            "datacompy.cli.compare.load_snowflake",
            side_effect=lambda s, ref, fmt, **kw: ref,
        ),
        patch(
            "datacompy.cli.compare.make_snowflake_compare",
            return_value=mock_compare,
        ) as mock_make,
    ):
        cli(
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
            ]
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
# _single_char_delimiter (parser-level argparse type= callable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        pytest.param("\\t", "\t", id="backslash_t_escape"),
        pytest.param("\t", "\t", id="real_tab_unchanged"),
        pytest.param(";", ";", id="semicolon_unchanged"),
        pytest.param(",", ",", id="comma_unchanged"),
        pytest.param("|", "|", id="pipe_unchanged"),
    ],
)
def test_single_char_delimiter_valid(raw: str, expected: str) -> None:
    assert _single_char_delimiter(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("ab", id="two_chars"),
        pytest.param("abc", id="three_chars"),
        pytest.param("\\n", id="backslash_n_not_a_single_char"),
        pytest.param("\\r", id="backslash_r_not_a_single_char"),
    ],
)
def test_single_char_delimiter_invalid_raises(raw: str) -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _single_char_delimiter(raw)


# ---------------------------------------------------------------------------
# CSV delimiter integration (end-to-end via CLI)
# ---------------------------------------------------------------------------


def test_csv_delimiter_backslash_t_parses_tsv_pandas(
    tsv_pair: Callable[[list[list[str]], list[list[str]]], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    rows = [["id", "val"], ["1", "a"], ["2", "b"]]
    left, right = tsv_pair(rows, rows)
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
            "\\t",
        ]
    )
    assert code == 0, f"expected match, stderr: {err}"


def test_csv_delimiter_backslash_t_parses_tsv_polars(
    tsv_pair: Callable[[list[list[str]], list[list[str]]], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    rows = [["id", "val"], ["1", "a"], ["2", "b"]]
    left, right = tsv_pair(rows, rows)
    code, _, err = cli(
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
        ]
    )
    assert code == 0, f"expected match, stderr: {err}"


def test_csv_delimiter_backslash_t_detects_mismatch(
    tsv_pair: Callable[[list[list[str]], list[list[str]]], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left_rows = [["id", "val"], ["1", "a"], ["2", "b"]]
    right_rows = [["id", "val"], ["1", "a"], ["2", "X"]]
    left, right = tsv_pair(left_rows, right_rows)
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
            "\\t",
        ]
    )
    assert code == 1


def test_csv_delimiter_real_tab_also_works(
    tsv_pair: Callable[[list[list[str]], list[list[str]]], tuple[Path, Path]],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    """A real tab character passed programmatically should still work."""
    rows = [["id", "val"], ["1", "a"], ["2", "b"]]
    left, right = tsv_pair(rows, rows)
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
            "\t",
        ]
    )
    assert code == 0, f"expected match, stderr: {err}"


# ---------------------------------------------------------------------------
# load_snowflake staging
# ---------------------------------------------------------------------------

_snowflake_missing = importlib.util.find_spec("snowflake") is None
_skip_no_snowflake = pytest.mark.skipif(
    _snowflake_missing, reason="snowflake.snowpark not installed"
)


@_skip_no_snowflake
def test_load_snowflake_staging_raises_error_when_no_database(
    mock_snowflake_session: MagicMock,
    tmp_path: Path,
) -> None:
    """load_snowflake with a local CSV and session.get_current_database()=None
    must raise BadArgsError, not return a broken '.' -prefixed table ref."""
    mock_snowflake_session.get_current_database.return_value = None
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,val\n1,a\n2,b\n")
    with pytest.raises(BadArgsError, match=r"SNOWFLAKE_DATABASE|database"):
        load_snowflake(mock_snowflake_session, str(csv_file), "csv")


@_skip_no_snowflake
def test_load_snowflake_staging_raises_error_when_no_schema(
    mock_snowflake_session: MagicMock,
    tmp_path: Path,
) -> None:
    """load_snowflake with a local CSV and session.get_current_schema()=None
    must raise BadArgsError, not return a broken ref like 'PROD..DATACOMPY_TMP_...'."""
    mock_snowflake_session.get_current_schema.return_value = None
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,val\n1,a\n2,b\n")
    with pytest.raises(BadArgsError, match=r"SNOWFLAKE_SCHEMA|schema"):
        load_snowflake(mock_snowflake_session, str(csv_file), "csv")


@_skip_no_snowflake
def test_load_snowflake_staging_returns_valid_ref_when_both_set(
    mock_snowflake_session: MagicMock,
    tmp_path: Path,
) -> None:
    """When db and schema are both available, the returned ref must be
    a fully-qualified 'db.schema.table' string."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,val\n1,a\n2,b\n")
    ref = load_snowflake(mock_snowflake_session, str(csv_file), "csv")
    parts = ref.split(".")
    assert len(parts) == 3, f"expected db.schema.table, got {ref!r}"
    assert parts[0] == "PROD"
    assert parts[1] == "PUBLIC"
    assert parts[2].startswith("DATACOMPY_TMP_")


@_skip_no_snowflake
def test_load_snowflake_csv_delimiter_is_honoured_when_staging(
    mock_snowflake_session: MagicMock,
    tmp_path: Path,
) -> None:
    """When a semicolon-delimited CSV is staged, write_pandas must receive
    a DataFrame with two separate columns, not a single column whose name
    contains the delimiter."""
    semi_csv = tmp_path / "data.csv"
    semi_csv.write_text("id;val\n1;a\n2;b\n")

    captured_df: list[pd.DataFrame] = []

    def _capture_write_pandas(df: pd.DataFrame, **kwargs: object) -> None:
        captured_df.append(df)

    mock_snowflake_session.write_pandas.side_effect = _capture_write_pandas

    load_snowflake(mock_snowflake_session, str(semi_csv), "csv", csv_delimiter=";")

    assert len(captured_df) == 1, "write_pandas should have been called once"
    df = captured_df[0]
    assert list(df.columns) == [
        "id",
        "val",
    ], f"Expected ['id', 'val'], got {list(df.columns)!r} -- delimiter was ignored"
