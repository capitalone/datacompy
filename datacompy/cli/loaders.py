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

from datacompy.cli.errors import BadArgsError, LoadError

_TABLE_REF_RE = re.compile(r"^[a-zA-Z_$][\w$]*(\.[a-zA-Z_$][\w$]*){1,2}$")

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
    - does not exist as a local file (an existing path is never a table ref),
    - contains no path separators (rules out file paths and URIs),
    - does not end with a recognised file-like extension (rules out
      ``data.csv``, ``archive.zip``, ``snapshot.parquet.gz``, etc.), and
    - matches the ``word.word[.word]`` pattern (2- or 3-part dotted identifier).

    The local-file existence check is performed first so that files with
    unusual or unlisted extensions (e.g. ``data.backup``, ``model.v2``) are
    never misclassified as table refs.

    Note: ``.gz`` and ``.zip`` are included in the extension guard so that
    compressed file paths with known suffixes are never mistaken for table
    refs.  They are not loadable by the CLI directly; use ``--format csv``
    (etc.) with the underlying reader if it supports the compression format.
    """
    if Path(ref).exists():
        return False
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

_NDJSON_EXTENSIONS = frozenset({".jsonl", ".ndjson"})


def _is_ndjson(path: str) -> bool:
    """Return ``True`` when *path* has a newline-delimited JSON extension."""
    return Path(path).suffix.lower() in _NDJSON_EXTENSIONS


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
            return pd.read_json(path, lines=_is_ndjson(path))
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
            return pl.read_ndjson(path) if _is_ndjson(path) else pl.read_json(path)
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
            return spark.read.json(path, multiLine=not _is_ndjson(path))
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


def load_snowflake(session: Any, ref: str) -> str:
    """Resolve *ref* to a fully-qualified ``db.schema.table`` name.

    Only Snowflake table references (``db.schema.table`` or ``schema.table``)
    are accepted.  Local file paths are not supported for the Snowflake backend;
    use the ``--backend pandas`` or ``--backend polars`` loader to read local
    files, or stage the data to a Snowflake table before comparing.

    Parameters
    ----------
    session:
        A live :class:`snowflake.snowpark.Session`.
    ref:
        A ``db.schema.table`` (3-part) or ``schema.table`` (2-part) identifier.
        2-part refs are expanded using the session's current database.

    Returns
    -------
    str
        A ``db.schema.table`` string usable by
        :class:`~datacompy.snowflake.SnowflakeCompare`.

    Raises
    ------
    BadArgsError
        When *ref* is not a recognised table reference, or when a 2-part ref
        cannot be resolved because the session has no current database.
    """
    if not is_snowflake_ref(ref):
        raise BadArgsError(
            f"{ref!r} does not look like a Snowflake table reference "
            "(expected db.schema.table or schema.table). "
            "Local file loading is not supported for the Snowflake backend. "
            "Stage your data to a Snowflake table first, then pass the table reference."
        )
    return _expand_table_ref(session, ref)
