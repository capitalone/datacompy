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

#: File extension to canonical format name.
_EXTENSION_FORMATS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
}

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
        return _EXTENSION_FORMATS[extension]
    except KeyError:
        raise BadArgsError(
            f"cannot infer the format of {ref!r} from its extension "
            f"{extension or '(none)'!r}. Pass --input-format csv|parquet|json."
        ) from None


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
            raise MissingExtraError(
                f"the {self.name} backend requires 'datacompy[{self.extra}]'. "
                f"Install it with: pip install 'datacompy[{self.extra}]'"
            ) from exc
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
                return pd.read_csv(ref, sep=namespace.csv_delimiter)
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
                return pl.read_csv(ref, separator=namespace.csv_delimiter)
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
        """Return a :class:`pyspark.sql.SparkSession`, stopping it on exit.

        The log level defaults to ``ERROR`` so PySpark's INFO and WARN chatter
        stays out of the CLI's own output. Override it with
        ``DATACOMPY_SPARK_LOG_LEVEL``.
        """
        try:
            from pyspark.sql import SparkSession
        except ImportError as exc:
            raise MissingExtraError(
                "the spark backend requires 'datacompy[spark]'. "
                "Install it with: pip install 'datacompy[spark]'"
            ) from exc

        spark = SparkSession.builder.appName(namespace.spark_app_name).getOrCreate()
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
                    sep=namespace.csv_delimiter,
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
            raise MissingExtraError(
                "the snowflake backend requires 'datacompy[snowflake]'. "
                "Install it with: pip install 'datacompy[snowflake]'"
            ) from exc

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

    missing = [
        name
        for name, value in (("SNOWFLAKE_ACCOUNT", account), ("SNOWFLAKE_USER", user))
        if not value
    ]
    if missing:
        raise BadArgsError(
            f"missing required environment variable(s): {', '.join(missing)}. "
            "Set them or pass --snowflake-config path/to/connection.json."
        )
    if not password and not authenticator:
        raise BadArgsError(
            "set SNOWFLAKE_PASSWORD or SNOWFLAKE_AUTHENTICATOR, or pass "
            "--snowflake-config for full control over the connection parameters."
        )

    params: dict[str, str] = {"account": str(account), "user": str(user)}
    if password:
        params["password"] = password
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
