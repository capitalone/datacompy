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

"""End to end tests for ``datacompy compare`` on the in memory backends.

Every test drives :func:`datacompy.cli.main`, which returns the process exit
code, so the assertions are on the contract a CI pipeline actually sees.
"""

import sys

import pytest
from datacompy.cli import main

from tests.cli.conftest import TOTAL_DIFFERING_ROWS, UNEQUAL_ROWS

MATCH = 0
MISMATCH = 1
ERROR = 2

IN_MEMORY_BACKENDS = ["pandas", "polars"]


@pytest.fixture(params=IN_MEMORY_BACKENDS)
def backend(request):
    """Run the test once per in memory backend."""
    return request.param


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


def test_identical_datasets_exit_zero(cli, left_csv, backend, capsys):
    assert cli("--right", str(left_csv), "--backend", backend) == MATCH


def test_differing_datasets_exit_one(cli, backend, capsys):
    assert cli("--backend", backend) == MISMATCH


def test_missing_input_file_exits_two(cli, tmp_path, backend, capsys):
    assert cli("--left", str(tmp_path / "nope.csv"), "--backend", backend) == ERROR
    assert "file not found" in capsys.readouterr().err


def test_unreadable_input_exits_two(cli, tmp_path, backend, capsys):
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_text("this is not parquet")
    assert cli("--left", str(corrupt), "--backend", backend) == ERROR
    assert "cannot read" in capsys.readouterr().err


def test_keyboard_interrupt_exits_130(cli, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("datacompy.cli.COMMANDS", {"compare": boom})
    assert cli() == 130
    assert "interrupted" in capsys.readouterr().err


def test_debug_reraises_instead_of_printing(cli, tmp_path):
    from datacompy.cli.errors import LoadError

    with pytest.raises(LoadError):
        cli("--left", str(tmp_path / "nope.csv"), "--debug")


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_join_columns_are_required(left_csv, right_csv, capsys):
    exit_code = main(["compare", "--left", str(left_csv), "--right", str(right_csv)])
    assert exit_code == ERROR
    assert "--on is required" in capsys.readouterr().err


def test_on_and_on_index_are_mutually_exclusive(cli, capsys):
    assert cli("--on", "id", "--on-index", "--backend", "pandas") == ERROR
    assert "mutually exclusive" in capsys.readouterr().err


def test_on_index_is_rejected_for_non_pandas_backends(cli, capsys):
    assert cli("--on-index", "--backend", "polars") == ERROR
    message = capsys.readouterr().err
    assert "--on-index is not supported with --backend polars" in message
    assert "pandas" in message


def test_backend_specific_flags_are_rejected_elsewhere(cli, capsys):
    assert cli("--spark-app-name", "x", "--backend", "polars") == ERROR
    assert "--spark-app-name is not supported" in capsys.readouterr().err


def test_cache_intermediates_is_rejected_for_non_spark_backends(cli, capsys):
    assert cli("--no-cache-intermediates", "--backend", "polars") == ERROR
    assert "--cache-intermediates is not supported" in capsys.readouterr().err


def test_cast_column_names_lower_is_rejected_for_snowflake(cli, capsys):
    assert cli("--no-cast-column-names-lower", "--backend", "snowflake") == ERROR
    assert "--cast-column-names-lower is not supported" in capsys.readouterr().err


def test_ignore_unique_rows_needs_a_threshold(cli, backend, capsys):
    assert cli("--ignore-unique-rows", "--backend", backend) == ERROR
    assert "--max-unequal-rows" in capsys.readouterr().err


def test_unknown_extension_needs_an_explicit_format(
    cli, tmp_path, left_csv, backend, capsys
):
    renamed = tmp_path / "left.data"
    renamed.write_text(left_csv.read_text())

    assert cli("--left", str(renamed), "--backend", backend) == ERROR
    assert "--input-format" in capsys.readouterr().err

    assert (
        cli("--left", str(renamed), "--input-format", "csv", "--backend", backend)
        == MISMATCH
    )


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------


def test_on_index_joins_on_the_dataframe_index(cli, capsys):
    assert cli("--on-index", "--backend", "pandas") == MISMATCH


@pytest.mark.parametrize(
    "on_args",
    [["--on", "id,name"], ["--on", "id", "--on", "name"]],
)
def test_multi_column_joins(cli, backend, on_args, capsys):
    assert cli(*on_args, "--backend", backend) == MISMATCH


def test_unknown_join_column_exits_two(cli, backend, capsys):
    assert cli("--on", "not_a_column", "--backend", backend) == ERROR
    assert "not_a_column" in capsys.readouterr().err


def test_comparing_a_file_against_itself_gets_distinct_report_labels(
    cli, left_csv, backend, capsys
):
    # Identical dataset labels would make the report ambiguous, and the pandas
    # backend cannot merge two frames that share a name.
    assert cli("--right", str(left_csv), "--backend", backend) == MATCH
    out = capsys.readouterr().out
    assert "left_1" in out
    assert "left_2" in out


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


def test_global_absolute_tolerance_absorbs_the_difference(cli, backend, capsys):
    assert (
        cli(
            "--abs-tol",
            "0.01",
            "--max-unequal-rows",
            "0",
            "--ignore-unique-rows",
            "--backend",
            backend,
        )
        == MATCH
    )


def test_per_column_tolerance_absorbs_the_difference(cli, backend, capsys):
    assert (
        cli(
            "--abs-tol",
            "amount=0.01",
            "--max-unequal-rows",
            "0",
            "--ignore-unique-rows",
            "--backend",
            backend,
        )
        == MATCH
    )


def test_per_column_tolerance_on_another_column_does_not(cli, backend, capsys):
    assert (
        cli(
            "--abs-tol",
            "name=99",
            "--max-unequal-rows",
            "0",
            "--ignore-unique-rows",
            "--backend",
            backend,
        )
        == MISMATCH
    )


def test_relative_tolerance(cli, backend, capsys):
    assert (
        cli(
            "--rel-tol",
            "0.001",
            "--max-unequal-rows",
            "0",
            "--ignore-unique-rows",
            "--backend",
            backend,
        )
        == MATCH
    )


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (0, MISMATCH),
        (TOTAL_DIFFERING_ROWS - 1, MISMATCH),
        (TOTAL_DIFFERING_ROWS, MATCH),
        (TOTAL_DIFFERING_ROWS + 1, MATCH),
    ],
)
def test_max_unequal_rows_counts_unique_rows_by_default(
    cli, backend, threshold, expected, capsys
):
    assert cli("--max-unequal-rows", str(threshold), "--backend", backend) == expected


@pytest.mark.parametrize(
    "threshold, expected",
    [(UNEQUAL_ROWS - 1, MISMATCH), (UNEQUAL_ROWS, MATCH)],
)
def test_ignore_unique_rows_counts_only_value_mismatches(
    cli, backend, threshold, expected, capsys
):
    assert (
        cli(
            "--max-unequal-rows",
            str(threshold),
            "--ignore-unique-rows",
            "--backend",
            backend,
        )
        == expected
    )


def test_extra_columns_fail_the_threshold_unless_ignored(
    cli, tmp_path, left_frame, backend, capsys
):
    wider = tmp_path / "wider.csv"
    left_frame.assign(extra=1).to_csv(wider, index=False)

    threshold = ["--max-unequal-rows", str(TOTAL_DIFFERING_ROWS)]
    assert cli("--left", str(wider), *threshold, "--backend", backend) == MISMATCH
    assert (
        cli(
            "--left",
            str(wider),
            *threshold,
            "--ignore-extra-columns",
            "--backend",
            backend,
        )
        == MATCH
    )


def test_ignore_extra_columns_without_a_threshold(
    cli, tmp_path, left_frame, left_csv, backend, capsys
):
    wider = tmp_path / "wider.csv"
    left_frame.assign(extra=1).to_csv(wider, index=False)

    assert (
        cli("--left", str(wider), "--right", str(left_csv), "--backend", backend)
        == MISMATCH
    )
    assert (
        cli(
            "--left",
            str(wider),
            "--right",
            str(left_csv),
            "--ignore-extra-columns",
            "--backend",
            backend,
        )
        == MATCH
    )


# ---------------------------------------------------------------------------
# Empty and non-overlapping inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_csv(tmp_path, left_frame):
    """The same columns as the fixtures, with no rows."""
    path = tmp_path / "empty.csv"
    left_frame.iloc[:0].to_csv(path, index=False)
    return path


def test_two_empty_datasets_are_not_a_match(cli, empty_csv, backend, capsys):
    """An empty intersection is a non-match, not a match with nothing to report.

    Every backend's ``intersect_rows_match`` returns False when no rows overlap,
    so "no rows differ" and "the datasets match" are different answers here.
    ``within_threshold`` derives its verdict from the report data rather than
    calling ``matches()``, and this pins the one case where the shortest reading
    of those counts would disagree with the library.
    """
    assert (
        cli("--left", str(empty_csv), "--right", str(empty_csv), "--backend", backend)
        == MISMATCH
    )


def test_datasets_with_no_overlapping_join_keys_are_not_a_match(
    cli, tmp_path, left_frame, backend, capsys
):
    """Every row is unique to one side, so nothing intersects."""
    disjoint = tmp_path / "disjoint.csv"
    left_frame.assign(id=left_frame["id"] + 100).to_csv(disjoint, index=False)

    assert cli("--right", str(disjoint), "--backend", backend) == MISMATCH


def test_empty_datasets_satisfy_an_explicit_zero_threshold(
    cli, empty_csv, backend, capsys
):
    """``--max-unequal-rows`` asks a different question, and gets a different answer.

    The threshold is a bound on how many rows differ, and zero rows differ, so an
    empty comparison clears it even though it is not a match. Keeping the two
    branches distinct is deliberate.
    """
    assert (
        cli(
            "--left",
            str(empty_csv),
            "--right",
            str(empty_csv),
            "--max-unequal-rows",
            "0",
            "--backend",
            backend,
        )
        == MATCH
    )


# ---------------------------------------------------------------------------
# Normalisation flags
# ---------------------------------------------------------------------------


def test_ignore_case(cli, tmp_path, left_frame, left_csv, backend, capsys):
    upper = tmp_path / "upper.csv"
    left_frame.assign(name=left_frame["name"].str.upper()).to_csv(upper, index=False)

    assert (
        cli("--left", str(upper), "--right", str(left_csv), "--backend", backend)
        == MISMATCH
    )
    assert (
        cli(
            "--left",
            str(upper),
            "--right",
            str(left_csv),
            "--ignore-case",
            "--backend",
            backend,
        )
        == MATCH
    )


def test_ignore_spaces(cli, tmp_path, left_frame, left_csv, backend, capsys):
    padded = tmp_path / "padded.csv"
    left_frame.assign(name="  " + left_frame["name"] + "  ").to_csv(padded, index=False)

    assert (
        cli("--left", str(padded), "--right", str(left_csv), "--backend", backend)
        == MISMATCH
    )
    assert (
        cli(
            "--left",
            str(padded),
            "--right",
            str(left_csv),
            "--ignore-spaces",
            "--backend",
            backend,
        )
        == MATCH
    )


def test_cast_column_names_lower(cli, tmp_path, left_frame, left_csv, backend, capsys):
    # Only the non-join column is uppercased, so --on id resolves either way and
    # the flag alone decides whether the two frames line up.
    shouty = tmp_path / "shouty.csv"
    left_frame.rename(columns={"name": "NAME"}).to_csv(shouty, index=False)

    assert (
        cli("--left", str(shouty), "--right", str(left_csv), "--backend", backend)
        == MATCH
    )
    assert (
        cli(
            "--left",
            str(shouty),
            "--right",
            str(left_csv),
            "--no-cast-column-names-lower",
            "--backend",
            backend,
        )
        == MISMATCH
    )


# ---------------------------------------------------------------------------
# Input formats
# ---------------------------------------------------------------------------


def test_parquet_input(tmp_path, left_frame, right_frame, backend, capsys):
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    left_frame.to_parquet(left)
    right_frame.to_parquet(right)

    assert (
        main(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--backend",
                backend,
            ]
        )
        == MISMATCH
    )


def test_json_input(tmp_path, left_frame, right_frame, backend, capsys):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left_frame.to_json(left, orient="records")
    right_frame.to_json(right, orient="records")

    assert (
        main(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--backend",
                backend,
            ]
        )
        == MISMATCH
    )


def test_newline_delimited_json_input(
    tmp_path, left_frame, right_frame, backend, capsys
):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    left_frame.to_json(left, orient="records", lines=True)
    right_frame.to_json(right, orient="records", lines=True)

    assert (
        main(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--backend",
                backend,
            ]
        )
        == MISMATCH
    )


def test_mixed_input_formats_are_inferred_per_file(
    tmp_path, left_frame, backend, capsys
):
    csv_path = tmp_path / "left.csv"
    parquet_path = tmp_path / "right.parquet"
    left_frame.to_csv(csv_path, index=False)
    left_frame.to_parquet(parquet_path)

    assert (
        main(
            [
                "compare",
                "--left",
                str(csv_path),
                "--right",
                str(parquet_path),
                "--on",
                "id",
                "--backend",
                backend,
            ]
        )
        == MATCH
    )


def test_custom_csv_delimiter(tmp_path, left_frame, backend, capsys):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left_frame.to_csv(left, index=False, sep="\t")
    left_frame.to_csv(right, index=False, sep="\t")

    assert (
        main(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--csv-delimiter",
                r"\t",
                "--backend",
                backend,
            ]
        )
        == MATCH
    )


def test_tsv_extension_is_inferred(tmp_path, left_frame, backend, capsys):
    left = tmp_path / "left.tsv"
    right = tmp_path / "right.tsv"
    left_frame.to_csv(left, index=False, sep="\t")
    left_frame.to_csv(right, index=False, sep="\t")

    assert (
        main(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--backend",
                backend,
            ]
        )
        == MATCH
    )


def test_mixed_csv_and_tsv_delimiters_are_inferred_per_file(
    tmp_path, left_frame, backend, capsys
):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.tsv"
    left_frame.to_csv(left, index=False)
    left_frame.to_csv(right, index=False, sep="\t")

    assert (
        main(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--backend",
                backend,
            ]
        )
        == MATCH
    )


def test_tab_extension_is_not_inferred(tmp_path, left_frame, backend, capsys):
    """``.tab`` is deliberately unmapped: some tools mean "tabular" by it."""
    left = tmp_path / "left.tab"
    right = tmp_path / "right.tab"
    left_frame.to_csv(left, index=False, sep="\t")
    left_frame.to_csv(right, index=False, sep="\t")

    assert (
        main(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--backend",
                backend,
            ]
        )
        == ERROR
    )
    assert "cannot infer the format" in capsys.readouterr().err


def test_explicit_delimiter_overrides_the_extension(
    tmp_path, left_frame, backend, capsys
):
    """The escape hatch for a comma separated file that carries a ``.tsv`` name.

    Without this the two return statements in ``infer_delimiter`` could be
    reordered and every other delimiter test would still pass.
    """
    left = tmp_path / "left.tsv"
    right = tmp_path / "right.tsv"
    left_frame.to_csv(left, index=False)
    left_frame.to_csv(right, index=False)

    argv = [
        "compare",
        "--left",
        str(left),
        "--right",
        str(right),
        "--on",
        "id",
        "--backend",
        backend,
    ]

    assert main([*argv, "--csv-delimiter", ","]) == MATCH
    # The extension is wrong about this file, so inference alone cannot read it.
    assert main(argv) == ERROR


def test_wrong_delimiter_warns_before_the_join_column_error(
    tmp_path, left_frame, backend, capsys
):
    left = tmp_path / "left.tsv"
    right = tmp_path / "right.tsv"
    left_frame.to_csv(left, index=False)
    left_frame.to_csv(right, index=False)

    assert (
        main(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--backend",
                backend,
            ]
        )
        == ERROR
    )

    stderr = capsys.readouterr().err
    assert "parsed into a single column" in stderr
    assert "--csv-delimiter" in stderr
    assert str(left) in stderr
    assert "must have all columns from join_columns" in stderr


def test_wrong_delimiter_warns_on_the_on_index_path(tmp_path, left_frame, capsys):
    """``--on-index`` raises nothing, so the warning is the only signal."""
    left = tmp_path / "left.tsv"
    right = tmp_path / "right.tsv"
    left_frame.to_csv(left, index=False)
    left_frame.to_csv(right, index=False)

    main(
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

    assert "parsed into a single column" in capsys.readouterr().err


def test_a_correctly_parsed_input_is_not_warned_about(cli, backend, capsys):
    assert cli("--backend", backend) == MISMATCH
    assert "warning" not in capsys.readouterr().err


def test_a_genuine_single_column_file_is_not_warned_about(
    tmp_path, left_frame, backend, capsys
):
    """One column and no delimiter in its name is an ordinary file, not a misparse."""
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left_frame[["id"]].to_csv(left, index=False)
    left_frame[["id"]].to_csv(right, index=False)

    assert (
        main(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--on",
                "id",
                "--backend",
                backend,
            ]
        )
        == MATCH
    )
    assert "warning" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Entry point wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["datacompy", "datacompy.cli"])
def test_module_entry_points_report_the_package_version(module):
    import subprocess

    from datacompy import __version__

    result = subprocess.run(
        [sys.executable, "-m", module, "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in result.stdout


def test_module_entry_point_propagates_the_exit_code(left_csv, right_csv):
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "datacompy",
            "compare",
            "--left",
            str(left_csv),
            "--right",
            str(right_csv),
            "--on",
            "id",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == MISMATCH
    assert result.stdout == ""
