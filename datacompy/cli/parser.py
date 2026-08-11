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

"""Declarative argument specification for the DataComPy CLI.

The module holds a single source of truth, :data:`OPTIONS`, describing every
option the ``compare`` subcommand accepts. Each :class:`Opt` records both how
argparse should register the flag and which ``*Compare`` constructor keyword it
maps to, so :func:`build_parser` and
:func:`datacompy.cli.backends.compare_kwargs` are generated from the same data
rather than maintained as two parallel hand-written lists.

Adding a library keyword argument to the CLI is therefore a single new row, and
``tests/cli/test_parser.py`` checks every :attr:`Opt.kwarg` against the real
constructor signature so the two can never silently drift apart.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from datacompy.cli.errors import BadArgsError

ALL_BACKENDS = frozenset({"pandas", "polars", "spark", "snowflake"})
FILE_BACKENDS = frozenset({"pandas", "polars", "spark"})
INPUT_FORMATS = ("csv", "parquet", "json")
REPORT_FORMATS = ("text", "json", "html")

#: Argument group headings, in the order they appear in ``--help``.
GROUP_INPUT = "input"
GROUP_JOIN = "join keys"
GROUP_BACKEND = "backend"
GROUP_COMPARISON = "comparison"
GROUP_NAMING = "naming"
GROUP_REPORT = "report"
GROUP_OUTPUT = "output"
GROUP_BACKEND_SPECIFIC = "backend specific"


@dataclass(frozen=True)
class Opt:
    """One command line option and its mapping onto a ``*Compare`` keyword.

    Attributes
    ----------
    flags : tuple of str
        Option strings passed to ``argparse.ArgumentParser.add_argument``.
    help : str
        Help text shown in ``--help``.
    group : str
        Argument group heading the option is listed under.
    kwarg : str, optional
        Name of the ``*Compare`` constructor keyword this option supplies.
        ``None`` marks the option as CLI only (for example ``--output``).
    backends : frozenset of str
        Backends that accept this option. Passing the flag with any other
        backend is rejected by
        :func:`datacompy.cli.compare.validate_arguments`.
    resolve : callable, optional
        Post-processing applied to the raw parsed value before it is handed to
        the constructor. Receives ``(raw_value, namespace)``. Used where the
        parsed shape differs from the library shape, such as flattening
        repeated ``--on`` groups into a single list.
    default : Any
        Value used when the flag is absent. Options are registered with
        ``argparse.SUPPRESS`` so that "left at the default" stays
        distinguishable from "explicitly passed".
    options : dict
        Extra keyword arguments forwarded to ``add_argument`` (``action``,
        ``type``, ``choices``, ``metavar``, ``required``).
    """

    flags: tuple[str, ...]
    help: str
    group: str
    kwarg: str | None = None
    backends: frozenset[str] = ALL_BACKENDS
    resolve: Callable[[Any, argparse.Namespace], Any] | None = None
    default: Any = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def dest(self) -> str:
        """The ``argparse`` destination attribute, derived from the first flag."""
        return self.flags[0].lstrip("-").replace("-", "_")

    def was_given(self, namespace: argparse.Namespace) -> bool:
        """Return ``True`` when the user actually passed this flag."""
        return hasattr(namespace, self.dest)

    def value(self, namespace: argparse.Namespace) -> Any:
        """Return the parsed value, falling back to :attr:`default`."""
        return getattr(namespace, self.dest, self.default)

    def resolved(self, namespace: argparse.Namespace) -> Any:
        """Return the value in the shape the library constructor expects."""
        raw = self.value(namespace)
        if self.resolve is None:
            return raw
        return self.resolve(raw, namespace)


# ---------------------------------------------------------------------------
# argparse ``type=`` callables
# ---------------------------------------------------------------------------


def join_column_group(value: str) -> list[str]:
    """Split a single ``--on`` value on commas.

    Combined with ``action="append"`` this makes ``--on id,date``,
    ``--on id --on date`` and any mix of the two equivalent. Column names that
    genuinely contain a comma must use the repeated form.
    """
    columns = [col.strip() for col in value.split(",") if col.strip()]
    if not columns:
        raise argparse.ArgumentTypeError("--on requires at least one column name")
    return columns


def tolerance(value: str) -> float | tuple[str, float]:
    """Parse a tolerance as either a bare number or a ``COLUMN=VALUE`` pair.

    A bare number applies to every numeric column. Repeated ``COLUMN=VALUE``
    pairs are collected into the per column dictionary that
    :func:`datacompy.base.validate_tolerance_parameter` accepts.
    """
    column, sep, raw = value.partition("=")
    text = raw if sep else value
    try:
        number = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a number or COLUMN=NUMBER, got {value!r}"
        ) from exc
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"tolerance must not be negative, got {number}"
        )
    if not sep:
        return number
    if not column.strip():
        raise argparse.ArgumentTypeError(f"missing column name in {value!r}")
    return column.strip(), number


def single_char(value: str) -> str:
    r"""Accept a one character delimiter, translating a literal ``\t`` to a tab."""
    translated = value.replace("\\t", "\t")
    if len(translated) != 1:
        raise argparse.ArgumentTypeError(
            f"expected a single character, got {value!r}. "
            r"Use '\t' (or the shell escape $'\t') for tab separated files."
        )
    return translated


def non_negative_int(value: str) -> int:
    """Accept a non negative integer."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a non negative integer, got {value!r}"
        ) from exc
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non negative integer, got {number}"
        )
    return number


# ---------------------------------------------------------------------------
# ``resolve=`` callables
# ---------------------------------------------------------------------------


def _flatten_join_columns(
    raw: list[list[str]] | None, namespace: argparse.Namespace
) -> list[str] | None:
    """Flatten repeated ``--on`` groups into a single ordered column list."""
    if not raw:
        return None
    return [column for group in raw for column in group]


def _combine_tolerances(
    raw: list[float | tuple[str, float]] | None,
    namespace: argparse.Namespace,
    *,
    flag: str,
) -> float | dict[str, float] | None:
    """Reduce repeated tolerance values to a single float or a per column dict.

    The library accepts ``float | Dict[str, float]`` but not a mixture, so
    combining a bare number with ``COLUMN=VALUE`` pairs is rejected here with a
    message naming the flag.
    """
    if not raw:
        return None
    pairs = [item for item in raw if isinstance(item, tuple)]
    scalars = [item for item in raw if not isinstance(item, tuple)]
    if pairs and scalars:
        raise BadArgsError(
            f"{flag} takes either a single number or one or more COLUMN=VALUE "
            "pairs, not both."
        )
    if scalars:
        if len(scalars) > 1:
            raise BadArgsError(
                f"{flag} was given a bare number more than once. Use "
                f"{flag} COLUMN=VALUE to set per column tolerances."
            )
        return scalars[0]
    return dict(pairs)


def default_dataset_name(ref: str, backend: str) -> str:
    """Derive a report label from a file path or Snowflake table reference.

    File paths use the stem, so ``sales_data.parquet`` becomes ``sales_data``.
    Snowflake references use the final segment, so ``PROD.ANALYTICS.SALES``
    becomes ``SALES`` rather than the misleading ``PROD.ANALYTICS`` that
    ``Path.stem`` would produce.
    """
    if backend == "snowflake":
        return ref.rsplit(".", 1)[-1]
    return Path(ref).stem


def _resolve_dataset_name(
    raw: str | None, namespace: argparse.Namespace, *, side: str
) -> str:
    """Return an explicit dataset label, or one derived from the input reference.

    When both sides derive the same label, which happens when a file is compared
    against itself or against a file of the same name in another directory, the
    labels are suffixed so the two columns of the report stay distinguishable.
    """
    if raw is not None:
        return raw
    left = default_dataset_name(namespace.left, namespace.backend)
    right = default_dataset_name(namespace.right, namespace.backend)
    if left != right:
        return left if side == "left" else right
    return f"{left}_1" if side == "left" else f"{right}_2"


# ---------------------------------------------------------------------------
# The specification
# ---------------------------------------------------------------------------

OPTIONS: tuple[Opt, ...] = (
    Opt(
        flags=("--left",),
        help="Path, URI, or Snowflake table reference for the left dataset.",
        group=GROUP_INPUT,
        options={"required": True, "metavar": "REF"},
    ),
    Opt(
        flags=("--right",),
        help="Path, URI, or Snowflake table reference for the right dataset.",
        group=GROUP_INPUT,
        options={"required": True, "metavar": "REF"},
    ),
    Opt(
        flags=("--input-format",),
        help=(
            "Force the input file format for both datasets. Omit to infer it "
            "from each file extension, which also handles mixed format inputs."
        ),
        group=GROUP_INPUT,
        backends=FILE_BACKENDS,
        options={"choices": list(INPUT_FORMATS)},
    ),
    Opt(
        flags=("--csv-delimiter",),
        help=(
            "Field delimiter for CSV input. Inferred from each file extension, "
            "a tab for .tsv and a comma otherwise. This flag overrides "
            r"inference for both inputs: use '\t' for a tab separated file "
            "with an unusual extension, or ',' to force a comma for a comma "
            "separated file named .tsv."
        ),
        group=GROUP_INPUT,
        backends=FILE_BACKENDS,
        options={"type": single_char, "metavar": "CHAR"},
    ),
    Opt(
        flags=("--on",),
        help=(
            "Join column. Accepts a comma separated list (--on id,date) or "
            "repeated flags (--on id --on date). Use the repeated form for "
            "column names that contain a comma."
        ),
        group=GROUP_JOIN,
        kwarg="join_columns",
        resolve=_flatten_join_columns,
        options={
            "action": "append",
            "type": join_column_group,
            "metavar": "COL[,COL...]",
        },
    ),
    Opt(
        flags=("--on-index",),
        help="Join on the DataFrame index instead of columns. Pandas backend only.",
        group=GROUP_JOIN,
        kwarg="on_index",
        backends=frozenset({"pandas"}),
        default=False,
        options={"action": "store_true"},
    ),
    Opt(
        flags=("--backend",),
        help=(
            "Comparison backend. Polars is fast and the default. Use pandas for "
            "index based joins, spark for distributed data, or snowflake to "
            "compare tables in place."
        ),
        group=GROUP_BACKEND,
        default="polars",
        options={"choices": sorted(ALL_BACKENDS)},
    ),
    Opt(
        flags=("--abs-tol",),
        help=(
            "Absolute tolerance for numeric comparisons (default 0). Accepts a "
            "single number applied to every column, or repeated COLUMN=VALUE "
            "pairs for per column tolerances."
        ),
        group=GROUP_COMPARISON,
        kwarg="abs_tol",
        resolve=partial(_combine_tolerances, flag="--abs-tol"),
        options={"action": "append", "type": tolerance, "metavar": "N|COL=N"},
    ),
    Opt(
        flags=("--rel-tol",),
        help=(
            "Relative tolerance for numeric comparisons (default 0). Accepts a "
            "single number or repeated COLUMN=VALUE pairs."
        ),
        group=GROUP_COMPARISON,
        kwarg="rel_tol",
        resolve=partial(_combine_tolerances, flag="--rel-tol"),
        options={"action": "append", "type": tolerance, "metavar": "N|COL=N"},
    ),
    Opt(
        flags=("--ignore-spaces",),
        help="Ignore leading and trailing whitespace in string columns.",
        group=GROUP_COMPARISON,
        kwarg="ignore_spaces",
        default=False,
        options={"action": "store_true"},
    ),
    Opt(
        flags=("--ignore-case",),
        help="Ignore case in string columns.",
        group=GROUP_COMPARISON,
        kwarg="ignore_case",
        default=False,
        options={"action": "store_true"},
    ),
    Opt(
        flags=("--cast-column-names-lower",),
        help=(
            "Cast column names to lowercase before comparing (default: enabled). "
            "Not applicable to snowflake, which normalises identifiers to uppercase."
        ),
        group=GROUP_COMPARISON,
        kwarg="cast_column_names_lower",
        backends=FILE_BACKENDS,
        default=True,
        options={"action": argparse.BooleanOptionalAction},
    ),
    Opt(
        flags=("--ignore-extra-columns",),
        help=(
            "Treat the datasets as matching even when one side has columns the "
            "other does not."
        ),
        group=GROUP_COMPARISON,
        default=False,
        options={"action": "store_true"},
    ),
    Opt(
        flags=("--df1-name",),
        help="Label for the left dataset in the report (default: derived from --left).",
        group=GROUP_NAMING,
        kwarg="df1_name",
        resolve=partial(_resolve_dataset_name, side="left"),
        options={"metavar": "NAME"},
    ),
    Opt(
        flags=("--df2-name",),
        help="Label for the right dataset in the report (default: derived from --right).",
        group=GROUP_NAMING,
        kwarg="df2_name",
        resolve=partial(_resolve_dataset_name, side="right"),
        options={"metavar": "NAME"},
    ),
    Opt(
        flags=("--sample-count",),
        help="Maximum number of sample mismatch rows to show per column (default 10).",
        group=GROUP_REPORT,
        default=10,
        options={"type": non_negative_int, "metavar": "N"},
    ),
    Opt(
        flags=("--column-count",),
        help="Maximum number of columns to show in unique row samples (default 10).",
        group=GROUP_REPORT,
        default=10,
        options={"type": non_negative_int, "metavar": "N"},
    ),
    Opt(
        flags=("--max-unequal-rows",),
        help=(
            "Exit 0 when the number of differing rows is at most N, and 1 "
            "otherwise. Counts value mismatches plus rows present in only one "
            "dataset; pass --ignore-unique-rows to count value mismatches only."
        ),
        group=GROUP_REPORT,
        options={"type": non_negative_int, "metavar": "N"},
    ),
    Opt(
        flags=("--ignore-unique-rows",),
        help=(
            "With --max-unequal-rows, exclude rows that exist in only one "
            "dataset from the difference count."
        ),
        group=GROUP_REPORT,
        default=False,
        options={"action": "store_true"},
    ),
    Opt(
        flags=("--report-format",),
        help="Report rendering (default: text).",
        group=GROUP_OUTPUT,
        default="text",
        options={"choices": list(REPORT_FORMATS)},
    ),
    Opt(
        flags=("--output",),
        help=(
            "Write the report to this file instead of stdout. Parent "
            "directories are created as needed."
        ),
        group=GROUP_OUTPUT,
        options={"type": Path, "metavar": "PATH"},
    ),
    Opt(
        flags=("--quiet",),
        help=(
            "Do not print the report to stdout. A file named by --output is "
            "still written. The exit code still reflects the result."
        ),
        group=GROUP_OUTPUT,
        default=False,
        options={"action": "store_true"},
    ),
    Opt(
        flags=("--spark-app-name",),
        help="Spark application name.",
        group=GROUP_BACKEND_SPECIFIC,
        backends=frozenset({"spark"}),
        default="datacompy-cli",
        options={"metavar": "NAME"},
    ),
    Opt(
        flags=("--cache-intermediates",),
        help=(
            "Cache intermediate DataFrames (default: enabled). Pass "
            "--no-cache-intermediates on Databricks Serverless and other "
            "environments that do not support caching."
        ),
        group=GROUP_BACKEND_SPECIFIC,
        kwarg="cache_intermediates",
        backends=frozenset({"spark"}),
        default=True,
        options={"action": argparse.BooleanOptionalAction},
    ),
    Opt(
        flags=("--snowflake-config",),
        help=(
            "Path to a JSON file of Snowflake connection parameters. When "
            "omitted the session is built from SNOWFLAKE_ACCOUNT plus one of "
            "SNOWFLAKE_TOKEN (OAuth), SNOWFLAKE_AUTHENTICATOR, or "
            "SNOWFLAKE_PASSWORD. SNOWFLAKE_USER is required for everything "
            "except OAuth, and SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE, "
            "SNOWFLAKE_DATABASE, and SNOWFLAKE_SCHEMA are optional."
        ),
        group=GROUP_BACKEND_SPECIFIC,
        backends=frozenset({"snowflake"}),
        options={"type": Path, "metavar": "PATH"},
    ),
)


#: Lookup from primary flag to specification row, for targeted validation.
OPTIONS_BY_FLAG: dict[str, Opt] = {opt.flags[0]: opt for opt in OPTIONS}


def fill_defaults(namespace: argparse.Namespace) -> None:
    """Populate *namespace* in place with the default for every absent option.

    Options are registered with ``argparse.SUPPRESS`` so that an absent flag
    leaves no attribute behind, which is what makes :meth:`Opt.was_given`
    meaningful. Call :func:`datacompy.cli.compare.validate_arguments` before
    this function, because afterwards every option looks as though it was
    explicitly passed.
    """
    for opt in OPTIONS:
        if not hasattr(namespace, opt.dest):
            setattr(namespace, opt.dest, opt.default)
    if not hasattr(namespace, "debug"):
        namespace.debug = False


def package_version() -> str:
    """Return the datacompy version.

    Read from the package itself rather than from installed distribution
    metadata, because ``pyproject.toml`` derives the distribution version from
    ``datacompy.__version__`` and an editable install can carry stale metadata.
    """
    from datacompy import __version__

    return __version__


def _debug_parent() -> argparse.ArgumentParser:
    """Return a parent parser supplying ``--debug``.

    Sharing it between the top level parser and the subcommand means ``--debug``
    is accepted on either side of the subcommand name. ``SUPPRESS`` stops the
    subparser from overwriting a value already set at the top level.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Re-raise unexpected exceptions with a full traceback instead of a "
            "short message. Useful when filing a bug report."
        ),
    )
    return parent


def build_parser() -> argparse.ArgumentParser:
    """Build the top level ``datacompy`` argument parser."""
    debug_parent = _debug_parent()
    parser = argparse.ArgumentParser(
        prog="datacompy",
        description="Compare two datasets across pandas, polars, Spark, or Snowflake.",
        parents=[debug_parent],
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version()}",
    )

    subcommands = parser.add_subparsers(dest="command", required=True)
    compare = subcommands.add_parser(
        "compare",
        help="Compare two datasets and report the differences.",
        description=(
            "Load --left and --right, compare them with --backend, and exit 0 "
            "when they match, 1 when they differ or a threshold is exceeded, "
            "and 2 on error."
        ),
        parents=[debug_parent],
    )

    groups: dict[str, argparse._ArgumentGroup] = {}
    for opt in OPTIONS:
        if opt.group not in groups:
            groups[opt.group] = compare.add_argument_group(opt.group)
        groups[opt.group].add_argument(
            *opt.flags,
            help=opt.help,
            default=argparse.SUPPRESS,
            **opt.options,
        )
    return parser
