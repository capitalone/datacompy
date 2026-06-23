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

"""Implementation of the ``datacompy compare`` subcommand."""

from typing import Any

from datacompy.cli.backends import (
    CompareArgs,
    make_pandas_compare,
    make_polars_compare,
    make_snowflake_compare,
    make_spark_compare,
    to_compare_args,
)
from datacompy.cli.errors import BadArgsError
from datacompy.cli.loaders import (
    infer_format,
    load_pandas,
    load_polars,
    load_snowflake,
    load_spark,
)
from datacompy.cli.output import emit


def run_compare(ns: Any) -> int:
    """Execute a comparison and return the appropriate exit code.

    Parameters
    ----------
    ns:
        Parsed :class:`argparse.Namespace` from the ``compare`` subparser.

    Returns
    -------
    int
        ``0`` — datasets match (within any specified thresholds).
        ``1`` — datasets differ or a threshold was violated.
        Raises :class:`~datacompy.cli.errors.CLIError` (→ exit ``2``) on
        invalid arguments, missing files, or unexpected exceptions.
    """
    args = to_compare_args(ns)
    _validate_args(args)

    compare, session = _build_compare(args)
    try:
        report_data = compare.build_report_data(
            sample_count=args.sample_count,
            column_count=args.column_count,
        )

        emit(report_data, as_json=args.json, quiet=args.quiet)

        if args.max_unequal_rows is not None:
            total_diff = report_data.row_summary.unequal_rows
            if not args.ignore_unique_rows:
                total_diff += (
                    report_data.row_summary.df1_unique
                    + report_data.row_summary.df2_unique
                )
            column_ok = args.ignore_extra_columns or (
                report_data.column_summary.df1_unique == 0
                and report_data.column_summary.df2_unique == 0
            )
            matched = column_ok and total_diff <= args.max_unequal_rows
        else:
            matched = compare.matches(ignore_extra_columns=args.ignore_extra_columns)

        return 0 if matched else 1
    finally:
        if session is not None:
            session.close()


def _validate_args(args: CompareArgs) -> None:
    """Raise :class:`~datacompy.cli.errors.BadArgsError` on invalid combos."""
    if args.on_index and args.backend != "pandas":
        raise BadArgsError("--on-index is only supported with --backend pandas.")
    if args.on_index and args.on:
        raise BadArgsError("--on and --on-index are mutually exclusive.")
    if not args.on_index and not args.on:
        raise BadArgsError(
            "--on is required (or --on-index for the pandas backend). "
            "Specify at least one join column with --on COL."
        )
    if len(args.csv_delimiter) != 1:
        raise BadArgsError(
            f"--csv-delimiter must be a single character, got {args.csv_delimiter!r}."
        )
    if args.max_unequal_rows is not None and args.max_unequal_rows < 0:
        raise BadArgsError("--max-unequal-rows must be a non-negative integer.")
    if args.ignore_unique_rows and args.max_unequal_rows is None:
        raise BadArgsError(
            "--ignore-unique-rows requires --max-unequal-rows to be set."
        )


def _build_compare(args: CompareArgs) -> tuple[Any, Any]:
    """Bootstrap the session (if needed), load files, and return (compare, session).

    The second element is the Snowflake session to close after use, or ``None``
    for backends that do not require an explicit session.
    """
    if args.backend == "pandas":
        fmt_l = infer_format(args.left, args.format)
        fmt_r = infer_format(args.right, args.format)
        pd1 = load_pandas(args.left, fmt_l, csv_delimiter=args.csv_delimiter)
        pd2 = load_pandas(args.right, fmt_r, csv_delimiter=args.csv_delimiter)
        return make_pandas_compare(args, pd1, pd2), None

    if args.backend == "polars":
        fmt_l = infer_format(args.left, args.format)
        fmt_r = infer_format(args.right, args.format)
        pl1 = load_polars(args.left, fmt_l, csv_delimiter=args.csv_delimiter)
        pl2 = load_polars(args.right, fmt_r, csv_delimiter=args.csv_delimiter)
        return make_polars_compare(args, pl1, pl2), None

    if args.backend == "spark":
        from datacompy.cli.sessions import get_spark_session

        spark = get_spark_session(args.spark_app_name)
        fmt_l = infer_format(args.left, args.format)
        fmt_r = infer_format(args.right, args.format)
        sp1 = load_spark(spark, args.left, fmt_l, csv_delimiter=args.csv_delimiter)
        sp2 = load_spark(spark, args.right, fmt_r, csv_delimiter=args.csv_delimiter)
        return make_spark_compare(args, spark, sp1, sp2), None

    if args.backend == "snowflake":
        from datacompy.cli.sessions import get_snowflake_session

        session = get_snowflake_session(args.snowflake_config)
        # infer_format is NOT called here — load_snowflake handles table refs
        # (e.g. DB.SCHEMA.MY_TABLE) that have no file extension.
        ref1 = load_snowflake(
            session, args.left, args.format, csv_delimiter=args.csv_delimiter
        )
        ref2 = load_snowflake(
            session, args.right, args.format, csv_delimiter=args.csv_delimiter
        )
        return make_snowflake_compare(args, session, ref1, ref2), session

    raise BadArgsError(f"Unknown backend: {args.backend!r}")
