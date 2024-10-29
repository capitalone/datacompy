"""Testing configuration file, currently used for generating a Snowpark local session for testing."""

import os

import pytest
from snowflake.snowpark.session import Session

CONNECTION_PARAMETERS = {
    "account": os.environ.get("SF_ACCOUNT"),
    "user": os.environ.get("SF_UID"),
    "password": os.environ.get("SF_PWD"),
    "warehouse": os.environ.get("SF_WAREHOUSE"),
    "database": os.environ.get("SF_DATABASE"),
    "schema": os.environ.get("SF_SCHEMA"),
}


def pytest_addoption(parser):
    parser.addoption("--snowflake-session", action="store", default="live")


@pytest.fixture(scope="module")
def snowpark_session() -> Session:
    return Session.builder.configs(CONNECTION_PARAMETERS).create()
