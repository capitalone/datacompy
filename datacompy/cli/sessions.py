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

"""Session factories for optional backends (Spark, Snowflake).

Imports for ``pyspark`` and ``snowflake.snowpark`` are quarantined here
so they are never loaded at package import time — the same pattern used
in ``datacompy/__init__.py``.
"""

import json
import os
from pathlib import Path
from typing import Any

from datacompy.cli.errors import BadArgsError, MissingExtraError


def get_spark_session(app_name: str = "datacompy-cli") -> Any:
    """Return a :class:`pyspark.sql.SparkSession`, creating one if needed.

    Sets log level to ``ERROR`` by default to suppress PySpark's verbose
    INFO/WARN output from the CLI stdout stream.  Override by setting the
    ``DATACOMPY_SPARK_LOG_LEVEL`` environment variable (e.g. ``INFO``).

    Parameters
    ----------
    app_name:
        Spark application name passed to ``SparkSession.builder.appName``.

    Raises
    ------
    MissingExtraError
        When ``pyspark`` is not installed.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise MissingExtraError(
            "Spark backend requires 'datacompy[spark]'. "
            "Install it with: pip install datacompy[spark]"
        ) from exc

    spark = SparkSession.builder.appName(app_name).getOrCreate()
    log_level = os.environ.get("DATACOMPY_SPARK_LOG_LEVEL", "ERROR")
    spark.sparkContext.setLogLevel(log_level)
    return spark


def get_snowflake_session(config_path: Path | None = None) -> Any:
    """Return a :class:`snowflake.snowpark.Session`.

    Parameters
    ----------
    config_path:
        Optional path to a JSON file whose top-level keys are Snowflake
        connection parameters (``account``, ``user``, ``password``, etc.).
        When ``None`` the session is built from environment variables:

        - ``SNOWFLAKE_ACCOUNT`` (required)
        - ``SNOWFLAKE_USER`` (required)
        - ``SNOWFLAKE_PASSWORD`` (required unless ``SNOWFLAKE_AUTHENTICATOR``
          is set to ``externalbrowser`` or another SSO authenticator)
        - ``SNOWFLAKE_ROLE`` (optional)
        - ``SNOWFLAKE_WAREHOUSE`` (optional)
        - ``SNOWFLAKE_DATABASE`` (optional)
        - ``SNOWFLAKE_SCHEMA`` (optional)
        - ``SNOWFLAKE_AUTHENTICATOR`` (optional; e.g. ``externalbrowser``)

    Raises
    ------
    MissingExtraError
        When ``snowflake.snowpark`` is not installed.
    BadArgsError
        When required environment variables are missing.
    """
    try:
        from snowflake.snowpark.session import Session
    except ImportError as exc:
        raise MissingExtraError(
            "Snowflake backend requires 'datacompy[snowflake]'. "
            "Install it with: pip install datacompy[snowflake]"
        ) from exc

    if config_path is not None:
        params: dict[str, str] = json.loads(config_path.read_text())
        return Session.builder.configs(params).create()

    # Build from environment variables.
    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    user = os.environ.get("SNOWFLAKE_USER")
    password = os.environ.get("SNOWFLAKE_PASSWORD")
    authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR")

    missing = [
        name
        for name, val in [("SNOWFLAKE_ACCOUNT", account), ("SNOWFLAKE_USER", user)]
        if not val
    ]
    if missing:
        raise BadArgsError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them or pass --snowflake-config path/to/conn.json."
        )

    if not password and not authenticator:
        raise BadArgsError(
            "Either SNOWFLAKE_PASSWORD or SNOWFLAKE_AUTHENTICATOR must be set. "
            "Pass --snowflake-config for full connection parameter control."
        )

    params = {"account": account, "user": user}  # type: ignore[dict-item]
    if password:
        params["password"] = password
    if authenticator:
        params["authenticator"] = authenticator
    for env_var, key in [
        ("SNOWFLAKE_ROLE", "role"),
        ("SNOWFLAKE_WAREHOUSE", "warehouse"),
        ("SNOWFLAKE_DATABASE", "database"),
        ("SNOWFLAKE_SCHEMA", "schema"),
    ]:
        val = os.environ.get(env_var)
        if val:
            params[key] = val

    return Session.builder.configs(params).create()
