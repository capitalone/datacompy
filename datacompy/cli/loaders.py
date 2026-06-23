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

"""File loaders for each supported backend.

Each ``load_*`` function accepts a path (or URI) string and a format
string (``"csv"``, ``"parquet"``, or ``"json"``) and returns a
backend-appropriate DataFrame.

Cloud URIs (``s3://``, ``gs://``, ``abfs://``) are passed through
as-is; they work when the user has installed the relevant optional
filesystem library (``s3fs``, ``gcsfs``, ``adlfs``).
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from datacompy.cli.errors import BadArgsError, LoadError, MissingExtraError

_TABLE_REF_RE = re.compile(r"^[\w$]+(\.[\w$]+){1,2}$")

_NON_TABLE_REF_EXTENSIONS = frozenset(
    {
        ".csv",
        ".tsv",
        ".parquet",
        ".pq",
        ".json",
        ".jsonl",
        ".ndjson",
        ".txt",
        ".gz",
        ".zip",
    }
)


def is_snowflake_ref(ref: str) -> bool:
    """Return ``True`` when *ref* looks like a Snowflake ``[db.]schema.table`` identifier.

    A ref is considered a table identifier when it:
    - contains no path separators (rules out file paths and URIs),
    - does not end with a recognised file-like extension (rules out
      ``data.csv``, ``archive.zip``, ``snapshot.parquet.gz``, etc.), and
    - matches the ``word.word[.word]`` pattern (2- or 3-part dotted identifier).

    Note: ``.gz`` and ``.zip`` are included in the extension guard so that
    compressed file paths are never mistaken for table refs.  They are not
    loadable by the CLI directly — use ``--format csv`` (etc.) with the
    underlying reader if it supports the compression format.
    """
    if "/" in ref or "\\" in ref:
        return False
    if Path(ref).suffix.lower() in _NON_TABLE_REF_EXTENSIONS:
        return False
    return bool(_TABLE_REF_RE.match(ref))


_FORMAT_MAP: dict[str, list[str]] = {
    "csv": [".csv", ".tsv"],
    "parquet": [".parquet", ".pq"],
    "json": [".json", ".jsonl", ".ndjson"],
}


def infer_format(path: str, override: str | None) -> str:
    """Return the file format string for *path*.

    Parameters
    ----------
    path:
        File path or URI.
    override:
        Explicit format string from ``--format``.  When provided it is
        returned directly without inspecting *path*.

    Returns
    -------
    str
        One of ``"csv"``, ``"parquet"``, or ``"json"``.

    Raises
    ------
    BadArgsError
        When the extension is not recognised and *override* is ``None``.
    """
    if override is not None:
        return override
    ext = Path(path).suffix.lower()
    for fmt, exts in _FORMAT_MAP.items():
        if ext in exts:
            return fmt
    raise BadArgsError(
        f"Cannot infer format from extension {ext!r}. "
        "Use --format csv|parquet|json to specify it explicitly."
    )


def load_pandas(path: str, fmt: str, csv_delimiter: str = ",") -> pd.DataFrame:
    """Load *path* into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path:
        Local path or cloud URI.
    fmt:
        One of ``"csv"``, ``"parquet"``, ``"json"``.
    csv_delimiter:
        Field delimiter used when *fmt* is ``"csv"`` (default: comma).

    Raises
    ------
    LoadError
        On any I/O or parse error (``FileNotFoundError``, ``OSError``,
        corrupt file, etc.).
    BadArgsError
        On unsupported *fmt*.
    """
    try:
        if fmt == "csv":
            return pd.read_csv(path, sep=csv_delimiter)
        if fmt == "parquet":
            return pd.read_parquet(path)
        if fmt == "json":
            return pd.read_json(path)
    except FileNotFoundError as exc:
        raise LoadError(f"File not found: {path}") from exc
    except Exception as exc:
        raise LoadError(f"Cannot read {path}: {exc}") from exc
    raise BadArgsError(f"Unsupported format for pandas loader: {fmt!r}")


def load_polars(path: str, fmt: str, csv_delimiter: str = ",") -> pl.DataFrame:
    """Load *path* into a :class:`polars.DataFrame`.

    Parameters
    ----------
    path:
        Local path or cloud URI.
    fmt:
        One of ``"csv"``, ``"parquet"``, ``"json"``.
    csv_delimiter:
        Field delimiter used when *fmt* is ``"csv"`` (default: comma).

    Raises
    ------
    LoadError
        On file-not-found or I/O errors.
    BadArgsError
        On unsupported *fmt*.
    """
    try:
        if fmt == "csv":
            return pl.read_csv(path, separator=csv_delimiter)
        if fmt == "parquet":
            return pl.read_parquet(path)
        if fmt == "json":
            return pl.read_json(path)
    except FileNotFoundError as exc:
        raise LoadError(f"File not found: {path}") from exc
    except Exception as exc:
        raise LoadError(f"Cannot read {path}: {exc}") from exc
    raise BadArgsError(f"Unsupported format for polars loader: {fmt!r}")


def load_spark(spark: Any, path: str, fmt: str, csv_delimiter: str = ",") -> Any:
    """Load *path* into a PySpark DataFrame.

    Parameters
    ----------
    spark:
        A live :class:`pyspark.sql.SparkSession`.
    path:
        Local path or cloud URI supported by the active Hadoop connectors.
    fmt:
        One of ``"csv"``, ``"parquet"``, ``"json"``.
    csv_delimiter:
        Field delimiter used when *fmt* is ``"csv"`` (default: comma).

    Raises
    ------
    LoadError
        On I/O errors reported by Spark.
    BadArgsError
        On unsupported *fmt*.
    """
    try:
        if fmt == "csv":
            return spark.read.csv(
                path, header=True, inferSchema=True, sep=csv_delimiter
            )
        if fmt == "parquet":
            return spark.read.parquet(path)
        if fmt == "json":
            return spark.read.json(path)
    except Exception as exc:
        raise LoadError(f"Spark cannot read {path}: {exc}") from exc
    raise BadArgsError(f"Unsupported format for Spark loader: {fmt!r}")


def _expand_table_ref(session: Any, ref: str) -> str:
    """Ensure *ref* is a fully-qualified ``db.schema.table`` identifier.

    A 3-part ref is returned unchanged.  A 2-part ``schema.table`` ref is
    expanded by prepending the session's current database.

    Raises
    ------
    BadArgsError
        When *ref* is 2-part and the session has no current database.  Users
        can fix this by passing the fully-qualified ``db.schema.table`` form
        or by setting ``SNOWFLAKE_DATABASE`` in their environment.
    """
    parts = ref.split(".")
    if len(parts) == 3:
        return ref
    db = session.get_current_database()
    if not db:
        raise BadArgsError(
            f"Cannot resolve {ref!r} to a fully-qualified table name: the "
            "Snowflake session has no current database. Either use the "
            "db.schema.table form or set SNOWFLAKE_DATABASE."
        )
    return f"{db}.{ref}"


def load_snowflake(
    session: Any, ref: str, fmt: str | None, csv_delimiter: str = ","
) -> Any:
    """Load *ref* for use with :class:`~datacompy.snowflake.SnowflakeCompare`.

    When *ref* looks like ``db.schema.table`` (3-part) or ``schema.table``
    (2-part) it is resolved to a fully-qualified table name and returned
    directly.  Otherwise *ref* is treated as a local file, loaded via
    Pandas, and staged to a temporary Snowflake table whose name is returned.

    Parameters
    ----------
    session:
        A live :class:`snowflake.snowpark.Session`.
    ref:
        A ``db.schema.table`` (3-part) or ``schema.table`` (2-part) identifier,
        or a local file path.  2-part refs are expanded using the session's
        current database.
    fmt:
        File format; only used when *ref* is a local file.
    csv_delimiter:
        Field delimiter used when *fmt* is ``"csv"`` (default: comma).

    Returns
    -------
    str
        A ``db.schema.table`` string usable by
        :class:`~datacompy.snowflake.SnowflakeCompare`.

    Raises
    ------
    MissingExtraError
        When ``snowflake.snowpark`` is not installed.
    BadArgsError
        When a 2-part ref cannot be expanded because the session has no
        current database, or when the session has no current database/schema
        and a local file needs to be staged.
    LoadError
        On file read or Snowflake staging errors.
    """
    try:
        import snowflake.snowpark  # noqa: F401
    except ImportError as exc:
        raise MissingExtraError(
            "Snowflake backend requires 'datacompy[snowflake]'. "
            "Install it with: pip install datacompy[snowflake]"
        ) from exc

    if is_snowflake_ref(ref):
        return _expand_table_ref(session, ref)

    # Local file → stage to a temporary Snowflake table.
    actual_fmt = fmt or infer_format(ref, None)
    try:
        if actual_fmt == "csv":
            df_pd = pd.read_csv(ref, sep=csv_delimiter)
        elif actual_fmt == "parquet":
            df_pd = pd.read_parquet(ref)
        elif actual_fmt == "json":
            df_pd = pd.read_json(ref)
        else:
            raise BadArgsError(
                f"Unsupported format for Snowflake loader: {actual_fmt!r}"
            )
    except FileNotFoundError as exc:
        raise LoadError(f"File not found: {ref}") from exc
    except OSError as exc:
        raise LoadError(f"Cannot read {ref}: {exc}") from exc

    from uuid import uuid4

    db = session.get_current_database()
    schema = session.get_current_schema()
    if not db or not schema:
        missing_vars = ", ".join(
            v
            for v, val in [("SNOWFLAKE_DATABASE", db), ("SNOWFLAKE_SCHEMA", schema)]
            if not val
        )
        raise BadArgsError(
            f"Cannot stage {ref!r} to Snowflake: session has no current "
            f"database/schema. Set {missing_vars} or use the db.schema.table "
            "form directly."
        )

    table_name = f"DATACOMPY_TMP_{uuid4().hex[:8].upper()}"
    try:
        session.write_pandas(
            df_pd,
            table_name=table_name,
            auto_create_table=True,
            table_type="temp",
            overwrite=True,
        )
    except Exception as exc:
        raise LoadError(
            f"Failed to stage {ref} to Snowflake temp table: {exc}"
        ) from exc

    return f"{db}.{schema}.{table_name}"
