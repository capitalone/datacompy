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

"""Backend factory functions.

Each ``make_*_compare`` function accepts a typed ``CompareArgs`` and the
already-loaded DataFrames, then constructs and returns the appropriate
backend-specific ``Compare`` instance.  The factories centralise the
constructor-signature differences between backends (e.g. Pandas supports
``on_index``; Snowflake has no ``cast_column_names_lower``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from datacompy.cli.loaders import is_snowflake_ref
from datacompy.pandas import PandasCompare
from datacompy.polars import PolarsCompare


@dataclass(frozen=True)
class CompareArgs:
    """Typed mirror of the ``compare`` subcommand arguments."""

    left: str
    right: str
    format: str | None
    on: list[str] | None
    on_index: bool
    backend: str
    abs_tol: float
    rel_tol: float
    ignore_spaces: bool
    ignore_case: bool
    ignore_extra_columns: bool
    ignore_unique_rows: bool
    cast_column_names_lower: bool
    csv_delimiter: str
    df1_name: str
    df2_name: str
    sample_count: int
    column_count: int
    max_unequal_rows: int | None
    json: bool
    quiet: bool
    spark_app_name: str
    snowflake_config: Path | None


def _default_name(ref: str) -> str:
    """Derive a human-readable dataset label from a file path or Snowflake table ref.

    For file paths ``Path.stem`` is used (``"sales_data.parquet"`` → ``"sales_data"``).
    For Snowflake table refs the table name (last segment) is used so that
    ``"PROD.ANALYTICS.SALES_FACT"`` → ``"SALES_FACT"`` rather than the
    misleading ``"PROD.ANALYTICS"`` that ``Path.stem`` would produce.
    """
    if is_snowflake_ref(ref):
        return ref.rsplit(".", 1)[-1]
    return Path(ref).stem


def _unescape_delimiter(raw: str) -> str:
    r"""Translate common escape sequences in a CLI-supplied delimiter string.

    Argparse always delivers argv values as plain strings, so a user who
    types ``--csv-delimiter '\\t'`` gets the two-character string ``\\t``
    rather than a real tab.  This function maps the most common sequences
    to their single-character equivalents so both forms work identically.
    """
    return raw.replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\r")


def to_compare_args(ns: Any) -> CompareArgs:
    """Convert an :class:`argparse.Namespace` to a typed :class:`CompareArgs`."""
    return CompareArgs(
        left=ns.left,
        right=ns.right,
        format=ns.format,
        on=ns.on,
        on_index=ns.on_index,
        backend=ns.backend,
        abs_tol=ns.abs_tol,
        rel_tol=ns.rel_tol,
        ignore_spaces=ns.ignore_spaces,
        ignore_case=ns.ignore_case,
        ignore_extra_columns=ns.ignore_extra_columns,
        ignore_unique_rows=ns.ignore_unique_rows,
        cast_column_names_lower=ns.cast_column_names_lower,
        csv_delimiter=_unescape_delimiter(ns.csv_delimiter),
        df1_name=ns.df1_name if ns.df1_name is not None else _default_name(ns.left),
        df2_name=ns.df2_name if ns.df2_name is not None else _default_name(ns.right),
        sample_count=ns.sample_count,
        column_count=ns.column_count,
        max_unequal_rows=ns.max_unequal_rows,
        json=ns.json,
        quiet=ns.quiet,
        spark_app_name=ns.spark_app_name,
        snowflake_config=ns.snowflake_config,
    )


def make_pandas_compare(
    args: CompareArgs, df1: pd.DataFrame, df2: pd.DataFrame
) -> PandasCompare:
    """Construct a :class:`~datacompy.pandas.PandasCompare`."""
    if args.on_index:
        return PandasCompare(
            df1,
            df2,
            on_index=True,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            df1_name=args.df1_name,
            df2_name=args.df2_name,
            ignore_spaces=args.ignore_spaces,
            ignore_case=args.ignore_case,
            cast_column_names_lower=args.cast_column_names_lower,
        )
    return PandasCompare(
        df1,
        df2,
        join_columns=args.on,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        df1_name=args.df1_name,
        df2_name=args.df2_name,
        ignore_spaces=args.ignore_spaces,
        ignore_case=args.ignore_case,
        cast_column_names_lower=args.cast_column_names_lower,
    )


def make_polars_compare(
    args: CompareArgs, df1: pl.DataFrame, df2: pl.DataFrame
) -> PolarsCompare:
    """Construct a :class:`~datacompy.polars.PolarsCompare`."""
    return PolarsCompare(
        df1,
        df2,
        join_columns=args.on or [],
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        df1_name=args.df1_name,
        df2_name=args.df2_name,
        ignore_spaces=args.ignore_spaces,
        ignore_case=args.ignore_case,
        cast_column_names_lower=args.cast_column_names_lower,
    )


def make_spark_compare(args: CompareArgs, spark: Any, df1: Any, df2: Any) -> Any:
    """Construct a :class:`~datacompy.spark.SparkSQLCompare`."""
    try:
        from datacompy.spark import SparkSQLCompare
    except ImportError as exc:
        from datacompy.cli.errors import MissingExtraError

        raise MissingExtraError(
            "Spark backend requires 'datacompy[spark]'. "
            "Install it with: pip install datacompy[spark]"
        ) from exc

    return SparkSQLCompare(
        spark,
        df1,
        df2,
        join_columns=args.on or [],
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        df1_name=args.df1_name,
        df2_name=args.df2_name,
        ignore_spaces=args.ignore_spaces,
        ignore_case=args.ignore_case,
        cast_column_names_lower=args.cast_column_names_lower,
    )


def make_snowflake_compare(
    args: CompareArgs, session: Any, ref1: Any, ref2: Any
) -> Any:
    """Construct a :class:`~datacompy.snowflake.SnowflakeCompare`."""
    try:
        from datacompy.snowflake import SnowflakeCompare
    except ImportError as exc:
        from datacompy.cli.errors import MissingExtraError

        raise MissingExtraError(
            "Snowflake backend requires 'datacompy[snowflake]'. "
            "Install it with: pip install datacompy[snowflake]"
        ) from exc

    return SnowflakeCompare(
        session,
        ref1,
        ref2,
        join_columns=args.on,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        df1_name=args.df1_name,
        df2_name=args.df2_name,
        ignore_spaces=args.ignore_spaces,
        ignore_case=args.ignore_case,
    )
