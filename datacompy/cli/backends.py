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

"""Backend strategies for the DataComPy CLI.

Each :class:`CLIBackend` owns the three things that vary between backends:
opening a session (Spark and Snowflake only), turning a ``--left`` or
``--right`` reference into something the comparison accepts, and calling the
right ``*Compare`` constructor.

The constructor keyword arguments themselves are not written out here. They are
derived from :data:`datacompy.cli.parser.OPTIONS` by :func:`compare_kwargs`, so
the parser and the constructor call site cannot drift apart.

Imports of ``pyspark`` and ``snowflake.snowpark`` are deferred into methods,
following the same pattern as ``datacompy/__init__.py``.
"""

import argparse
import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import polars as pl

from datacompy.base import BaseCompare
from datacompy.cli.errors import BadArgsError, LoadError, MissingExtraError
from datacompy.cli.parser import OPTIONS

#: File extension to canonical format name and field delimiter.
#:
#: Format and delimiter live in one table on purpose. Mapping an extension to
#: ``csv`` without deciding its delimiter picks the right reader but not the
#: right separator, so the file is recognised and then misparsed into a single
#: column. Keeping the two in separate tables makes that mistake possible again
#: every time an extension is added; here the second element cannot be
#: forgotten. It is ``None`` for formats that have no delimiter.
_EXTENSIONS: dict[str, tuple[str, str | None]] = {
    ".csv": ("csv", ","),
    ".tsv": ("csv", "\t"),
    ".parquet": ("parquet", None),
    ".pq": ("parquet", None),
    ".json": ("json", None),
    ".jsonl": ("json", None),
    ".ndjson": ("json", None),
}

#: Delimiter for CSV input whose extension does not imply one.
_DEFAULT_DELIMITER = ","

#: Delimiters worth naming when an input parses into a single column.
_PLAUSIBLE_DELIMITERS = (",", "\t", ";", "|")

_NDJSON_EXTENSIONS = frozenset({".jsonl", ".ndjson"})

#: A two or three part dotted Snowflake identifier, e.g. ``DB.SCHEMA.TABLE``.
_TABLE_REF = re.compile(r"^[A-Za-z_$][\w$]*(\.[A-Za-z_$][\w$]*){1,2}$")


def infer_format(ref: str, override: str | None) -> str:
    """Return the input format for *ref*.

    Parameters
    ----------
    ref : str
        File path or URI.
    override : str, optional
        Explicit ``--input-format`` value. Returned as is when provided.

    Returns
    -------
    str
        One of ``"csv"``, ``"parquet"``, or ``"json"``.

    Raises
    ------
    BadArgsError
        When the extension is unrecognised and no override was given.
    """
    if override is not None:
        return override
    extension = Path(ref).suffix.lower()
    try:
        return _EXTENSIONS[extension][0]
    except KeyError:
        raise BadArgsError(
            f"cannot infer the format of {ref!r} from its extension "
            f"{extension or '(none)'!r}. Pass --input-format csv|parquet|json."
        ) from None


def infer_delimiter(ref: str, override: str | None) -> str:
    """Return the field delimiter for *ref*.

    Parameters
    ----------
    ref : str
        File path or URI.
    override : str, optional
        Explicit ``--csv-delimiter`` value. Returned as is when provided, and
        applied to both inputs. This is also how a comma is forced for a comma
        delimited file that happens to be named ``.tsv``, because
        ``--input-format`` selects the reader and says nothing about the
        delimiter.

    Returns
    -------
    str
        The delimiter implied by the extension, or a comma for anything else.
    """
    if override is not None:
        return override
    _, delimiter = _EXTENSIONS.get(Path(ref).suffix.lower(), ("", None))
    return delimiter or _DEFAULT_DELIMITER


def suspect_delimiter(
    ref: str, namespace: argparse.Namespace, frame: Any
) -> str | None:
    """Return a warning when *frame* looks like it was read with the wrong delimiter.

    A CSV input read with the wrong delimiter collapses every row into one
    column whose name is the entire header line. Nothing fails at that point:
    the symptom surfaces later as a missing join column, or not at all under
    ``--on-index``, where two mangled strings are compared instead. This is the
    last place where the reference, the resolved format and the resolved
    delimiter are all still in hand, so the guess is made here.

    Parameters
    ----------
    ref : str
        The ``--left`` or ``--right`` reference *frame* was read from.
    namespace : argparse.Namespace
        Parsed arguments, for the format and delimiter overrides.
    frame : Any
        The loaded DataFrame. Only ``columns`` is read, which every file
        backend exposes without touching the data.

    Returns
    -------
    str or None
        A warning message, or ``None`` when the parse looks fine. Only a single
        column CSV whose one column name *repeats* a delimiter other than the
        one used is reported, because that is what a collapsed multi-column
        header looks like. A genuine single column file, a name that merely
        contains one such delimiter (for example ``a;b``), and a plain typo in
        ``--on`` all stay quiet.
    """
    try:
        if infer_format(ref, namespace.input_format) != "csv":
            return None
    except BadArgsError:
        # Not a file this module recognises, for example a Snowflake table
        # reference. Nothing to say about its delimiter.
        return None

    # A pandas Index has no usable truth value, so test for None explicitly.
    frame_columns = getattr(frame, "columns", None)
    if frame_columns is None:
        return None
    columns = list(frame_columns)
    if len(columns) != 1:
        return None

    name = str(columns[0])
    used = infer_delimiter(ref, namespace.csv_delimiter)
    # A collapsed multi-column header repeats the real delimiter once per lost
    # column, so a plausible delimiter appearing more than once is the mark of a
    # misparse. A single occurrence is just as easily a genuine column name such
    # as "a;b", so it is left alone rather than warned about on a correct run.
    if not any(char != used and name.count(char) > 1 for char in _PLAUSIBLE_DELIMITERS):
        return None
    return (
        f"{ref} parsed into a single column named {name!r}. It was read with "
        f"{used!r}, which may be the wrong delimiter. Pass --csv-delimiter to "
        "override the extension based default."
    )


def _is_ndjson(ref: str) -> bool:
    """Return ``True`` when *ref* looks like newline delimited JSON."""
    return Path(ref).suffix.lower() in _NDJSON_EXTENSIONS


def compare_kwargs(namespace: argparse.Namespace, backend: str) -> dict[str, Any]:
    """Build the ``*Compare`` constructor keyword arguments for *backend*.

    Walks :data:`datacompy.cli.parser.OPTIONS`, keeping the rows that name a
    constructor keyword and that apply to this backend. Values that resolve to
    ``None`` are omitted so the library default applies.
    """
    kwargs: dict[str, Any] = {}
    for opt in OPTIONS:
        if opt.kwarg is None or backend not in opt.backends:
            continue
        value = opt.resolved(namespace)
        if value is None:
            continue
        kwargs[opt.kwarg] = value
    return kwargs


def _missing_extra(backend: str, extra: str | None) -> MissingExtraError:
    """Return the error raised when *backend* cannot be imported.

    Kept in one place because the same wording is needed wherever an optional
    dependency is imported. *extra* is ``None`` for backends with no optional
    dependency, where a failed import is not something ``pip install`` fixes.
    """
    if extra is None:
        return MissingExtraError(f"the {backend} backend could not be imported.")
    return MissingExtraError(
        f"the {backend} backend requires 'datacompy[{extra}]'. "
        f"Install it with: pip install 'datacompy[{extra}]'"
    )


class CLIBackend(ABC):
    """Strategy describing how the CLI drives one comparison backend."""

    #: Value accepted by ``--backend``.
    name: ClassVar[str]
    #: Module holding the ``*Compare`` class, imported on demand.
    module: ClassVar[str]
    #: Name of the ``*Compare`` class within :attr:`module`.
    class_name: ClassVar[str]
    #: Optional dependency extra required by this backend, if any.
    extra: ClassVar[str | None] = None

    @property
    def compare_cls(self) -> Callable[..., BaseCompare]:
        """Import and return the backend's ``*Compare`` class.

        Typed as a callable rather than ``type[BaseCompare]`` because the
        constructor signatures differ between backends, which is exactly what
        :func:`compare_kwargs` and the drift guard in ``tests/cli`` exist to
        reconcile.

        Raises
        ------
        MissingExtraError
            When the backend needs an optional dependency that is not installed.
        """
        try:
            module = importlib.import_module(self.module)
        except ImportError as exc:
            raise _missing_extra(self.name, self.extra) from exc
        return getattr(module, self.class_name)  # type: ignore[no-any-return]

    def open_session(self, namespace: argparse.Namespace, stack: ExitStack) -> Any:
        """Return a session for this backend, or ``None`` when none is needed.

        Implementations register teardown on *stack* so the session is closed on
        both the success and the failure path.
        """
        return None

    @abstractmethod
    def load(self, session: Any, ref: str, namespace: argparse.Namespace) -> Any:
        """Turn *ref* into whatever the backend's comparison accepts."""

    def build(
        self, namespace: argparse.Namespace, session: Any, left: Any, right: Any
    ) -> BaseCompare:
        """Construct the ``*Compare`` instance."""
        return self.compare_cls(left, right, **compare_kwargs(namespace, self.name))


class PandasBackend(CLIBackend):
    """Compare two files in memory with pandas."""

    name = "pandas"
    module = "datacompy.pandas"
    class_name = "PandasCompare"

    def load(
        self, session: Any, ref: str, namespace: argparse.Namespace
    ) -> pd.DataFrame:
        """Read *ref* into a :class:`pandas.DataFrame`."""
        fmt = infer_format(ref, namespace.input_format)
        try:
            if fmt == "csv":
                return pd.read_csv(
                    ref, sep=infer_delimiter(ref, namespace.csv_delimiter)
                )
            if fmt == "parquet":
                return pd.read_parquet(ref)
            return pd.read_json(ref, lines=_is_ndjson(ref))
        except FileNotFoundError as exc:
            raise LoadError(f"file not found: {ref}") from exc
        except Exception as exc:
            raise LoadError(f"cannot read {ref}: {exc}") from exc


class PolarsBackend(CLIBackend):
    """Compare two files in memory with polars."""

    name = "polars"
    module = "datacompy.polars"
    class_name = "PolarsCompare"

    def load(
        self, session: Any, ref: str, namespace: argparse.Namespace
    ) -> pl.DataFrame:
        """Read *ref* into a :class:`polars.DataFrame`."""
        fmt = infer_format(ref, namespace.input_format)
        try:
            if fmt == "csv":
                return pl.read_csv(
                    ref, separator=infer_delimiter(ref, namespace.csv_delimiter)
                )
            if fmt == "parquet":
                return pl.read_parquet(ref)
            if _is_ndjson(ref):
                return pl.read_ndjson(ref)
            return pl.read_json(ref)
        except FileNotFoundError as exc:
            raise LoadError(f"file not found: {ref}") from exc
        except Exception as exc:
            raise LoadError(f"cannot read {ref}: {exc}") from exc


class SparkBackend(CLIBackend):
    """Compare two files with Spark SQL."""

    name = "spark"
    module = "datacompy.spark"
    class_name = "SparkSQLCompare"
    extra = "spark"

    def open_session(self, namespace: argparse.Namespace, stack: ExitStack) -> Any:
        """Return a :class:`pyspark.sql.SparkSession`, stopping it only if we made it.

        A SparkSession is process wide, so ``getOrCreate`` returns the caller's
        existing session when there is one. :func:`datacompy.cli.main` is a
        public function that can be called in process from a notebook or an
        Airflow task, and stopping a session the CLI did not create would kill
        the caller's ``SparkContext`` along with it. Teardown is therefore
        registered only when this call is the one that created the session.

        For the same reason ``--spark-app-name`` has no effect when a session
        already exists: an application name cannot be changed once the
        ``SparkContext`` is running.

        The log level defaults to ``ERROR`` so PySpark's INFO and WARN chatter
        stays out of the CLI's own output. Override it with
        ``DATACOMPY_SPARK_LOG_LEVEL``.
        """
        try:
            from pyspark.sql import SparkSession
        except ImportError as exc:
            raise _missing_extra(self.name, self.extra) from exc

        # Read before getOrCreate, which installs an active session as a side
        # effect. This is thread local, so a session created on another thread
        # and never activated on this one is not detected.
        borrowed = SparkSession.getActiveSession()
        spark = SparkSession.builder.appName(namespace.spark_app_name).getOrCreate()
        if borrowed is None:
            stack.callback(spark.stop)
        try:
            spark.sparkContext.setLogLevel(
                os.environ.get("DATACOMPY_SPARK_LOG_LEVEL", "ERROR")
            )
        except Exception:
            # Log level is cosmetic and is not available on every deployment,
            # so never let it prevent the session from being returned.
            pass
        return spark

    def load(self, session: Any, ref: str, namespace: argparse.Namespace) -> Any:
        """Read *ref* into a PySpark DataFrame."""
        fmt = infer_format(ref, namespace.input_format)
        try:
            if fmt == "csv":
                return session.read.csv(
                    ref,
                    header=True,
                    inferSchema=True,
                    sep=infer_delimiter(ref, namespace.csv_delimiter),
                )
            if fmt == "parquet":
                return session.read.parquet(ref)
            return session.read.json(ref, multiLine=not _is_ndjson(ref))
        except Exception as exc:
            raise LoadError(f"Spark cannot read {ref}: {exc}") from exc

    def build(
        self, namespace: argparse.Namespace, session: Any, left: Any, right: Any
    ) -> BaseCompare:
        """Construct a :class:`~datacompy.spark.SparkSQLCompare`."""
        return self.compare_cls(
            session, left, right, **compare_kwargs(namespace, self.name)
        )


class SnowflakeBackend(CLIBackend):
    """Compare two Snowflake tables in place.

    ``--left`` and ``--right`` are always table references for this backend,
    either ``db.schema.table`` or ``schema.table``. Local files are read with
    ``--backend pandas`` or ``--backend polars`` instead.
    """

    name = "snowflake"
    module = "datacompy.snowflake"
    class_name = "SnowflakeCompare"
    extra = "snowflake"

    def open_session(self, namespace: argparse.Namespace, stack: ExitStack) -> Any:
        """Return a Snowpark session built from ``--snowflake-config`` or the environment."""
        try:
            from snowflake.snowpark.session import Session
        except ImportError as exc:
            raise _missing_extra(self.name, self.extra) from exc

        params = _snowflake_params(namespace.snowflake_config)
        session = Session.builder.configs(params).create()
        stack.callback(session.close)
        return session

    def load(self, session: Any, ref: str, namespace: argparse.Namespace) -> str:
        """Validate *ref* and return it as a fully qualified table name.

        Raises
        ------
        BadArgsError
            When *ref* is not a two or three part dotted identifier, or when a
            two part reference cannot be qualified because the session has no
            current database.
        """
        if not _TABLE_REF.match(ref):
            raise BadArgsError(
                f"{ref!r} is not a Snowflake table reference. Expected "
                "db.schema.table or schema.table. The snowflake backend "
                "compares tables in place; use --backend pandas or "
                "--backend polars to compare local files."
            )
        if ref.count(".") == 2:
            return ref
        database = session.get_current_database()
        if not database:
            raise BadArgsError(
                f"cannot qualify {ref!r}: the Snowflake session has no current "
                "database. Use the db.schema.table form or set SNOWFLAKE_DATABASE."
            )
        return f"{database}.{ref}"

    def build(
        self, namespace: argparse.Namespace, session: Any, left: Any, right: Any
    ) -> BaseCompare:
        """Construct a :class:`~datacompy.snowflake.SnowflakeCompare`."""
        return self.compare_cls(
            session, left, right, **compare_kwargs(namespace, self.name)
        )


def _snowflake_params(config_path: Path | None) -> dict[str, str]:
    """Build Snowpark connection parameters from a JSON file or the environment.

    Raises
    ------
    BadArgsError
        When the config file is missing, unreadable, or not a JSON object, or
        when the environment is missing a required variable.
    """
    if config_path is not None:
        try:
            raw = json.loads(config_path.read_text())
        except FileNotFoundError as exc:
            raise BadArgsError(
                f"--snowflake-config file not found: {config_path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise BadArgsError(f"--snowflake-config {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise BadArgsError(
                "--snowflake-config must contain a JSON object of connection "
                f"parameters, got {type(raw).__name__}."
            )
        return raw

    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    user = os.environ.get("SNOWFLAKE_USER")
    password = os.environ.get("SNOWFLAKE_PASSWORD")
    authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR")
    token = os.environ.get("SNOWFLAKE_TOKEN")

    # The connector only reads ``token`` as an OAuth access token when the
    # authenticator says so, so a bare SNOWFLAKE_TOKEN implies OAuth.
    if token and not authenticator:
        authenticator = "oauth"
    oauth = authenticator is not None and authenticator.lower() == "oauth"

    required = [("SNOWFLAKE_ACCOUNT", account)]
    if not oauth:
        # Under OAuth the access token carries the identity, so a user name is
        # optional. Every other authenticator still needs one.
        required.append(("SNOWFLAKE_USER", user))
    missing = [name for name, value in required if not value]
    if missing:
        raise BadArgsError(
            f"missing required environment variable(s): {', '.join(missing)}. "
            "Set them or pass --snowflake-config path/to/connection.json."
        )
    if oauth and not token:
        raise BadArgsError(
            "SNOWFLAKE_AUTHENTICATOR=oauth requires an access token in "
            "SNOWFLAKE_TOKEN, or pass --snowflake-config for full control over "
            "the connection parameters."
        )
    if not password and not authenticator:
        raise BadArgsError(
            "set SNOWFLAKE_PASSWORD, SNOWFLAKE_TOKEN, or SNOWFLAKE_AUTHENTICATOR, "
            "or pass --snowflake-config for full control over the connection "
            "parameters."
        )

    params: dict[str, str] = {"account": str(account)}
    if user:
        params["user"] = user
    if password:
        params["password"] = password
    if token:
        params["token"] = token
    if authenticator:
        params["authenticator"] = authenticator
    for variable, key in (
        ("SNOWFLAKE_ROLE", "role"),
        ("SNOWFLAKE_WAREHOUSE", "warehouse"),
        ("SNOWFLAKE_DATABASE", "database"),
        ("SNOWFLAKE_SCHEMA", "schema"),
    ):
        value = os.environ.get(variable)
        if value:
            params[key] = value
    return params


BACKENDS: dict[str, CLIBackend] = {
    backend.name: backend
    for backend in (
        PandasBackend(),
        PolarsBackend(),
        SparkBackend(),
        SnowflakeBackend(),
    )
}
