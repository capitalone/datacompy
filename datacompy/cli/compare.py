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

import argparse
from contextlib import ExitStack

from datacompy.base import BaseCompare
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
from datacompy.report import ReportData


def run_compare(ns: argparse.Namespace) -> int:
    """Execute a comparison and return the appropriate exit code.

    Parameters
    ----------
    ns:
        Parsed :class:`argparse.Namespace` from the ``compare`` subparser.

    Returns
    -------
    int
        ``0`` -- datasets match (within any specified thresholds).
        ``1`` -- datasets differ or a threshold was violated.
        Raises :class:`~datacompy.cli.errors.CLIError` (exit ``2``) on
        invalid arguments, missing files, or unsupported backends.
    """
    args = to_compare_args(ns)
    _validate_arg_combinations(args)

    with ExitStack() as stack:
        compare = _build_backend(args, stack)
        report_data = compare.build_report_data(
            sample_count=args.sample_count,
            column_count=args.column_count,
        )
        emit(report_data, as_json=args.json, quiet=args.quiet)
        return 0 if _matched(args, report_data, compare) else 1


def _validate_arg_combinations(args: CompareArgs) -> None:
    """Raise :class:`~datacompy.cli.errors.BadArgsError` on invalid argument combinations.

    Single-value constraints (e.g. delimiter length, non-negative counts) are
    enforced earlier by argparse ``type=`` callables in ``parser.py``.  This
    function handles rules that require comparing two or more arguments.
    """
    if args.on_index and args.backend != "pandas":
        raise BadArgsError("--on-index is only supported with --backend pandas.")
    if args.on_index and args.on:
        raise BadArgsError("--on and --on-index are mutually exclusive.")
    if not args.on_index and not args.on:
        raise BadArgsError(
            "--on is required (or --on-index for the pandas backend). "
            "Specify at least one join column with --on COL."
        )
    if args.ignore_unique_rows and args.max_unequal_rows is None:
        raise BadArgsError(
            "--ignore-unique-rows requires --max-unequal-rows to be set."
        )


def _build_backend(args: CompareArgs, stack: ExitStack) -> BaseCompare:
    """Load both datasets and return the appropriate ``*Compare`` instance.

    *stack* is an :class:`~contextlib.ExitStack` that owns session cleanup for
    Spark and Snowflake.  Registering sessions on the stack ensures they are
    closed even when loading or comparison raises an exception.
    """
    if args.backend == "pandas":
        return _build_pandas(args)
    if args.backend == "polars":
        return _build_polars(args)
    if args.backend == "spark":
        return _build_spark(args, stack)
    if args.backend == "snowflake":
        return _build_snowflake(args, stack)
    raise BadArgsError(
        f"Unknown backend: {args.backend!r}"
    )  # unreachable (argparse choices=)


def _build_pandas(args: CompareArgs) -> BaseCompare:
    fmt_l = infer_format(args.left, args.format)
    fmt_r = infer_format(args.right, args.format)
    df1 = load_pandas(args.left, fmt_l, csv_delimiter=args.csv_delimiter)
    df2 = load_pandas(args.right, fmt_r, csv_delimiter=args.csv_delimiter)
    return make_pandas_compare(args, df1, df2)


def _build_polars(args: CompareArgs) -> BaseCompare:
    fmt_l = infer_format(args.left, args.format)
    fmt_r = infer_format(args.right, args.format)
    df1 = load_polars(args.left, fmt_l, csv_delimiter=args.csv_delimiter)
    df2 = load_polars(args.right, fmt_r, csv_delimiter=args.csv_delimiter)
    return make_polars_compare(args, df1, df2)


def _build_spark(args: CompareArgs, stack: ExitStack) -> BaseCompare:
    from datacompy.cli.sessions import get_spark_session

    spark = get_spark_session(args.spark_app_name)
    stack.callback(spark.stop)
    fmt_l = infer_format(args.left, args.format)
    fmt_r = infer_format(args.right, args.format)
    df1 = load_spark(spark, args.left, fmt_l, csv_delimiter=args.csv_delimiter)
    df2 = load_spark(spark, args.right, fmt_r, csv_delimiter=args.csv_delimiter)
    return make_spark_compare(args, spark, df1, df2)  # type: ignore[no-any-return]


def _build_snowflake(args: CompareArgs, stack: ExitStack) -> BaseCompare:
    from datacompy.cli.sessions import get_snowflake_session

    session = get_snowflake_session(args.snowflake_config)
    stack.callback(session.close)
    ref1 = load_snowflake(
        session, args.left, args.format, csv_delimiter=args.csv_delimiter
    )
    ref2 = load_snowflake(
        session, args.right, args.format, csv_delimiter=args.csv_delimiter
    )
    return make_snowflake_compare(args, session, ref1, ref2)  # type: ignore[no-any-return]


def _matched(args: CompareArgs, report_data: ReportData, compare: BaseCompare) -> bool:
    """Return ``True`` when the comparison satisfies the configured threshold."""
    if args.max_unequal_rows is not None:
        total_diff = report_data.row_summary.unequal_rows
        if not args.ignore_unique_rows:
            total_diff += (
                report_data.row_summary.df1_unique + report_data.row_summary.df2_unique
            )
        column_ok = args.ignore_extra_columns or (
            report_data.column_summary.df1_unique == 0
            and report_data.column_summary.df2_unique == 0
        )
        return column_ok and total_diff <= args.max_unequal_rows
    return compare.matches(ignore_extra_columns=args.ignore_extra_columns)
