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

"""Argument parser for the DataComPy CLI."""

import argparse
from pathlib import Path

import datacompy
from datacompy.cli.compare import run_compare


def _single_char_delimiter(value: str) -> str:
    r"""Argparse ``type=`` that accepts a single-character delimiter.

    Translates ``\t`` to a real tab so both the shell literal ``$'\t'`` and
    the string ``\t`` work identically.  Rejects anything that is not exactly
    one character after translation.
    """
    translated = value.replace("\\t", "\t")
    if len(translated) != 1:
        raise argparse.ArgumentTypeError(
            f"--csv-delimiter must be a single character, got {value!r}"
        )
    return translated


def _non_negative_int(value: str) -> int:
    """Argparse ``type=`` that accepts a non-negative integer."""
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--max-unequal-rows must be a non-negative integer, got {value!r}"
        ) from exc
    if n < 0:
        raise argparse.ArgumentTypeError(
            f"--max-unequal-rows must be a non-negative integer, got {n}"
        )
    return n


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="datacompy",
        description="Compare two datasets across multiple backends.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {datacompy.__version__}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Re-raise unexpected exceptions instead of printing a friendly message. "
        "Useful when filing bug reports.",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    _add_compare_subparser(sub)
    return parser


def _add_compare_subparser(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Register the ``compare`` subcommand on *sub*."""
    cmp = sub.add_parser(
        "compare",
        help="Compare two datasets and report differences.",
        description=(
            "Load --left and --right files, compare them using --backend, "
            "and exit 0 (match), 1 (mismatch / threshold violated), or 2 (error)."
        ),
    )

    # ---- inputs ----------------------------------------------------------------
    inp = cmp.add_argument_group("input")
    inp.add_argument(
        "--left", required=True, metavar="PATH", help="Path or URI to the left dataset."
    )
    inp.add_argument(
        "--right",
        required=True,
        metavar="PATH",
        help="Path or URI to the right dataset.",
    )
    inp.add_argument(
        "--format",
        choices=["csv", "parquet", "json"],
        default=None,
        help="Input file format.  Inferred from extension when omitted.",
    )
    inp.add_argument(
        "--csv-delimiter",
        type=_single_char_delimiter,
        default=",",
        metavar="CHAR",
        help=(
            "Field delimiter for CSV files (default: comma). "
            "Use '\\t' for tab-separated files (the shell escape $'\\t' also works)."
        ),
    )

    # ---- join keys -------------------------------------------------------------
    keys = cmp.add_argument_group("join keys")
    keys.add_argument(
        "--on",
        action="append",
        dest="on",
        default=None,
        metavar="COL",
        help="Join column name (required unless --on-index is used).  Repeat for composite keys: --on id --on date.",
    )
    keys.add_argument(
        "--on-index",
        action="store_true",
        default=False,
        help="Join on DataFrame index (Pandas backend only). Mutually exclusive with --on.",
    )

    # ---- backend ---------------------------------------------------------------
    backend = cmp.add_argument_group("backend")
    backend.add_argument(
        "--backend",
        choices=["pandas", "polars", "spark", "snowflake"],
        default="polars",
        help="Comparison backend.  Polars is fast and the default; use Pandas for index-based joins or wider ecosystem compatibility.  Default: polars.",
    )

    # ---- tolerances & flags ----------------------------------------------------
    tol = cmp.add_argument_group("tolerances and flags")
    tol.add_argument(
        "--abs-tol",
        type=float,
        default=0.0,
        metavar="N",
        help="Absolute tolerance for numeric comparisons (default 0.0).",
    )
    tol.add_argument(
        "--rel-tol",
        type=float,
        default=0.0,
        metavar="N",
        help="Relative tolerance for numeric comparisons (default 0.0).",
    )
    tol.add_argument(
        "--ignore-spaces",
        action="store_true",
        default=False,
        help="Ignore leading/trailing whitespace in string columns.",
    )
    tol.add_argument(
        "--ignore-case",
        action="store_true",
        default=False,
        help="Ignore case in string columns.",
    )
    tol.add_argument(
        "--ignore-extra-columns",
        action="store_true",
        default=False,
        help="Treat comparisons as matching even if one side has extra columns.",
    )
    tol.add_argument(
        "--cast-column-names-lower",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cast column names to lowercase before comparing (default: enabled).",
    )

    # ---- naming ----------------------------------------------------------------
    names = cmp.add_argument_group("naming")
    names.add_argument(
        "--df1-name",
        default=None,
        metavar="NAME",
        help="Label for the left dataset in the report (default: left filename stem).",
    )
    names.add_argument(
        "--df2-name",
        default=None,
        metavar="NAME",
        help="Label for the right dataset in the report (default: right filename stem).",
    )

    # ---- report shape ----------------------------------------------------------
    report = cmp.add_argument_group("report")
    report.add_argument(
        "--sample-count",
        type=int,
        default=10,
        metavar="N",
        help="Max mismatch rows to sample per column (default 10).",
    )
    report.add_argument(
        "--column-count",
        type=int,
        default=10,
        metavar="N",
        help="Max columns to display in unique-row samples (default 10).",
    )
    report.add_argument(
        "--max-unequal-rows",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help=(
            "Exit 0 if the number of differing rows is at most N; exit 1 otherwise. "
            "By default counts both value mismatches AND rows that exist only in one "
            "dataset. Pass --ignore-unique-rows to count value mismatches only. "
            "Must be a non-negative integer."
        ),
    )
    report.add_argument(
        "--ignore-unique-rows",
        action="store_true",
        default=False,
        help=(
            "With --max-unequal-rows: exclude rows that exist only in one dataset "
            "from the difference count. Only value mismatches in common rows are counted."
        ),
    )

    # ---- output ----------------------------------------------------------------
    out = cmp.add_argument_group("output")
    out.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit JSON report to stdout instead of text.",
    )
    out.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help=(
            "Suppress the text report. Has no effect when --json is also given. "
            "Exit code still reflects match/mismatch."
        ),
    )

    # ---- backend-specific ------------------------------------------------------
    bspec = cmp.add_argument_group("backend-specific options")
    bspec.add_argument(
        "--spark-app-name",
        default="datacompy-cli",
        metavar="NAME",
        help="Spark application name (Spark backend only).",
    )
    bspec.add_argument(
        "--snowflake-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a JSON file with Snowflake connection parameters. "
        "Overrides environment variables: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, "
        "SNOWFLAKE_PASSWORD (or SNOWFLAKE_AUTHENTICATOR for SSO), and optionally "
        "SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA "
        "(Snowflake backend only).",
    )

    cmp.set_defaults(func=run_compare)
