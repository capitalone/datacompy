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

"""The ``datacompy compare`` subcommand."""

import argparse
from contextlib import ExitStack

from datacompy.cli.backends import BACKENDS, suspect_delimiter
from datacompy.cli.errors import BadArgsError
from datacompy.cli.output import emit, print_warning
from datacompy.cli.parser import OPTIONS, OPTIONS_BY_FLAG, fill_defaults
from datacompy.report import ReportData


def run_compare(namespace: argparse.Namespace) -> int:
    """Run a comparison and return the process exit code.

    Parameters
    ----------
    namespace : argparse.Namespace
        Parsed arguments from the ``compare`` subparser.

    Returns
    -------
    int
        ``0`` when the datasets match or stay within ``--max-unequal-rows``,
        and ``1`` when they differ. Problems with the arguments, the input
        files, or the backend raise a
        :class:`~datacompy.cli.errors.CLIError`, which
        :func:`datacompy.cli.main` turns into exit code ``2``.
    """
    validate_arguments(namespace)
    fill_defaults(namespace)

    backend = BACKENDS[namespace.backend]
    with ExitStack() as stack:
        # The session is registered on the stack by the backend, so Spark and
        # Snowflake shut down on the failure path as well as the success path.
        session = backend.open_session(namespace, stack)
        left = backend.load(session, namespace.left, namespace)
        right = backend.load(session, namespace.right, namespace)

        # Warn before building, so a wrong delimiter is named ahead of the
        # missing join column it causes, and is still reported under
        # --on-index, where nothing fails at all.
        for ref, frame in ((namespace.left, left), (namespace.right, right)):
            suspect = suspect_delimiter(ref, namespace, frame)
            if suspect is not None:
                print_warning(suspect)

        try:
            comparison = backend.build(namespace, session, left, right)
        except ValueError as exc:
            # The comparison classes validate user supplied configuration such
            # as join columns and tolerances with a plain ValueError. That is a
            # bad argument, not a bug, so report it as one instead of dumping a
            # traceback on the user.
            raise BadArgsError(str(exc)) from exc

        report_data = comparison.build_report_data(
            sample_count=namespace.sample_count,
            column_count=namespace.column_count,
        )
        emit(
            report_data,
            namespace.report_format,
            namespace.output,
            quiet=namespace.quiet,
        )
        matched = within_threshold(namespace, report_data)
    return 0 if matched else 1


def validate_arguments(namespace: argparse.Namespace) -> None:
    """Reject argument combinations that argparse cannot express.

    Per value rules (delimiter length, non negative counts, tolerance syntax)
    are enforced by the ``type=`` callables in
    :mod:`datacompy.cli.parser`. This function covers the rules that need to
    look at more than one argument at a time.

    Must run before :func:`datacompy.cli.parser.fill_defaults`, which is what
    makes :meth:`~datacompy.cli.parser.Opt.was_given` distinguish an explicitly
    passed flag from an absent one.

    Raises
    ------
    BadArgsError
        On any invalid combination.
    """
    backend = OPTIONS_BY_FLAG["--backend"].value(namespace)

    for opt in OPTIONS:
        if opt.was_given(namespace) and backend not in opt.backends:
            applies_to = ", ".join(sorted(opt.backends))
            raise BadArgsError(
                f"{opt.flags[0]} is not supported with --backend {backend}. "
                f"It applies to: {applies_to}."
            )

    join_columns = OPTIONS_BY_FLAG["--on"].value(namespace)
    on_index = OPTIONS_BY_FLAG["--on-index"].value(namespace)
    if join_columns and on_index:
        raise BadArgsError("--on and --on-index are mutually exclusive.")
    if not join_columns and not on_index:
        raise BadArgsError(
            "--on is required. Specify at least one join column with --on COL, "
            "or use --on-index with --backend pandas to join on the index."
        )

    ignore_unique_rows = OPTIONS_BY_FLAG["--ignore-unique-rows"].value(namespace)
    max_unequal_rows = OPTIONS_BY_FLAG["--max-unequal-rows"].value(namespace)
    if ignore_unique_rows and max_unequal_rows is None:
        raise BadArgsError(
            "--ignore-unique-rows only has an effect together with "
            "--max-unequal-rows N."
        )


def within_threshold(
    namespace: argparse.Namespace,
    report_data: ReportData,
) -> bool:
    """Return ``True`` when the comparison counts as a pass.

    Without ``--max-unequal-rows`` this reproduces
    :meth:`~datacompy.base.BaseCompare.matches`. With it, the datasets pass
    while the number of differing rows stays at or below the threshold, and
    while the columns line up unless ``--ignore-extra-columns`` was given.

    Both branches read *report_data*, which ``build_report_data`` has already
    computed. Calling ``matches()`` here instead would recount straight from the
    DataFrames, and on Spark and Snowflake each of those counts is a distributed
    action, so the default invocation would scan the data a second time after
    the report had already been rendered.
    """
    rows = report_data.row_summary
    columns_ok = namespace.ignore_extra_columns or (
        report_data.column_summary.df1_unique == 0
        and report_data.column_summary.df2_unique == 0
    )

    if namespace.max_unequal_rows is None:
        rows_overlap = rows.df1_unique == 0 and rows.df2_unique == 0
        # An empty intersection is a non-match for every backend, so the
        # common_rows test is what keeps two empty datasets from passing.
        intersect_matches = rows.common_rows > 0 and rows.unequal_rows == 0
        return columns_ok and rows_overlap and intersect_matches

    differing_rows = rows.unequal_rows
    if not namespace.ignore_unique_rows:
        differing_rows += rows.df1_unique + rows.df2_unique
    return columns_ok and differing_rows <= namespace.max_unequal_rows
