"""Testing configuration file, currently used for generating a Snowpark session for testing."""

import os

import pytest
from snowflake.snowpark.session import Session


@pytest.fixture(scope="module")
def snowpark_session() -> Session:
    connection_parameters = {
        "account": os.environ.get("SF_ACCOUNT"),
        "user": os.environ.get("SF_UID"),
        "password": os.environ.get("SF_PWD"),
        "warehouse": os.environ.get("SF_WAREHOUSE"),  # optional
        "database": os.environ.get("SF_DATABASE"),
        "schema": os.environ.get("SF_SCHEMA"),
    }
    return Session.builder.configs(connection_parameters).create()
