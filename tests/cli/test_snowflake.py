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

"""Tests for the Snowflake backend's reference handling and session parameters.

None of these need Snowpark installed. Reference resolution only needs
``get_current_database``, so a stub session stands in, and the connection
parameter builder is pure. The comparison itself is exercised by the library's
own Snowflake tests.
"""

import json

import pytest
from datacompy.cli.backends import SnowflakeBackend, _snowflake_params
from datacompy.cli.errors import BadArgsError
from datacompy.cli.parser import build_parser, fill_defaults

SNOWFLAKE_ENV = (
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_AUTHENTICATOR",
    "SNOWFLAKE_TOKEN",
    "SNOWFLAKE_ROLE",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
)


class StubSession:
    """Minimal stand in exposing only what reference resolution touches."""

    def __init__(self, database=None):
        self._database = database

    def get_current_database(self):
        return self._database


@pytest.fixture
def namespace():
    """A parsed Snowflake invocation with defaults filled in."""
    parsed = build_parser().parse_args(
        [
            "compare",
            "--left",
            "PROD.ANALYTICS.SALES",
            "--right",
            "STAGE.ANALYTICS.SALES",
            "--on",
            "id",
            "--backend",
            "snowflake",
        ]
    )
    fill_defaults(parsed)
    return parsed


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every Snowflake environment variable so tests start from nothing."""
    for name in SNOWFLAKE_ENV:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------


def test_three_part_reference_is_used_as_is(namespace):
    backend = SnowflakeBackend()
    session = StubSession(database="OTHER")
    assert backend.load(session, "PROD.ANALYTICS.SALES", namespace) == (
        "PROD.ANALYTICS.SALES"
    )


def test_two_part_reference_is_qualified_with_the_current_database(namespace):
    backend = SnowflakeBackend()
    session = StubSession(database="PROD")
    assert backend.load(session, "ANALYTICS.SALES", namespace) == "PROD.ANALYTICS.SALES"


def test_two_part_reference_without_a_current_database_is_rejected(namespace):
    backend = SnowflakeBackend()
    with pytest.raises(BadArgsError, match="no current database"):
        backend.load(StubSession(database=None), "ANALYTICS.SALES", namespace)


@pytest.mark.parametrize(
    "ref",
    [
        "/data/sales.parquet",
        "s3://bucket/sales.parquet",
        "relative/sales.parquet",
        "SALES",
        "A.B.C.D",
        "1BAD.TABLE",
        "@stage/file.parquet",
        "",
    ],
)
def test_non_table_references_are_rejected(namespace, ref):
    """The backend never guesses whether a reference is a file or a table.

    With ``--backend snowflake`` a reference is always a table, so anything that
    is not a two or three part identifier is a clear argument error rather than
    something to fall back on.
    """
    backend = SnowflakeBackend()
    with pytest.raises(BadArgsError, match="not a Snowflake table reference"):
        backend.load(StubSession(database="PROD"), ref, namespace)


def test_a_bare_filename_is_read_as_a_table_reference(namespace):
    """``data.csv`` is a syntactically valid ``schema.table``, and is treated as one.

    This is the deliberate consequence of dropping file versus table guessing.
    Snowflake reports the missing table, which is clearer than a heuristic that
    is sometimes wrong in the other direction. Path-like and URI references are
    still rejected here, because they can never be identifiers.
    """
    backend = SnowflakeBackend()
    assert backend.load(StubSession(database="PROD"), "data.csv", namespace) == (
        "PROD.data.csv"
    )


@pytest.mark.parametrize(
    "ref", ["DB.SCHEMA.TABLE", "db.schema.table", "_DB.$SCHEMA.T1", "SCHEMA.TABLE"]
)
def test_valid_table_reference_shapes(namespace, ref):
    backend = SnowflakeBackend()
    assert backend.load(StubSession(database="PROD"), ref, namespace)


# ---------------------------------------------------------------------------
# Connection parameters
# ---------------------------------------------------------------------------


def test_config_file_is_used_verbatim(tmp_path):
    config = tmp_path / "connection.json"
    config.write_text(json.dumps({"account": "acct", "user": "u", "password": "p"}))
    assert _snowflake_params(config) == {
        "account": "acct",
        "user": "u",
        "password": "p",
    }


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(BadArgsError, match="file not found"):
        _snowflake_params(tmp_path / "absent.json")


def test_malformed_config_file_is_reported(tmp_path):
    config = tmp_path / "connection.json"
    config.write_text("{not json")
    with pytest.raises(BadArgsError, match="snowflake-config"):
        _snowflake_params(config)


def test_config_file_must_hold_an_object(tmp_path):
    config = tmp_path / "connection.json"
    config.write_text(json.dumps(["account", "user"]))
    with pytest.raises(BadArgsError, match="JSON object"):
        _snowflake_params(config)


def test_parameters_are_built_from_the_environment(monkeypatch, clean_env):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "u")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "p")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "wh")

    assert _snowflake_params(None) == {
        "account": "acct",
        "user": "u",
        "password": "p",
        "warehouse": "wh",
    }


def test_sso_authenticator_replaces_the_password(monkeypatch, clean_env):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "u")
    monkeypatch.setenv("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")

    params = _snowflake_params(None)
    assert params["authenticator"] == "externalbrowser"
    assert "password" not in params


def test_a_bare_token_selects_oauth_without_a_user(monkeypatch, clean_env):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_TOKEN", "tok")

    assert _snowflake_params(None) == {
        "account": "acct",
        "token": "tok",
        "authenticator": "oauth",
    }


def test_an_explicit_oauth_authenticator_is_left_alone(monkeypatch, clean_env):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_AUTHENTICATOR", "OAuth")
    monkeypatch.setenv("SNOWFLAKE_TOKEN", "tok")

    params = _snowflake_params(None)
    assert params["authenticator"] == "OAuth"
    assert params["token"] == "tok"


def test_a_user_is_still_passed_through_under_oauth(monkeypatch, clean_env):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_TOKEN", "tok")
    monkeypatch.setenv("SNOWFLAKE_USER", "u")

    assert _snowflake_params(None)["user"] == "u"


def test_oauth_without_a_token_is_rejected(monkeypatch, clean_env):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "u")
    monkeypatch.setenv("SNOWFLAKE_AUTHENTICATOR", "oauth")

    with pytest.raises(BadArgsError, match="SNOWFLAKE_TOKEN"):
        _snowflake_params(None)


def test_missing_required_environment_variables_are_named(monkeypatch, clean_env):
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "p")
    with pytest.raises(BadArgsError, match="SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER"):
        _snowflake_params(None)


def test_a_user_is_still_required_outside_oauth(monkeypatch, clean_env):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")

    with pytest.raises(BadArgsError, match="SNOWFLAKE_USER"):
        _snowflake_params(None)


def test_missing_credentials_are_reported(monkeypatch, clean_env):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "u")
    with pytest.raises(BadArgsError, match="SNOWFLAKE_PASSWORD"):
        _snowflake_params(None)
