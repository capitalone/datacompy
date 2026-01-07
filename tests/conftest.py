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

"""Testing configuration file, currently used for generating a Snowpark local session for testing."""

import os

import pytest

try:
    from snowflake.snowpark.session import Session
except ModuleNotFoundError:
    Session = None  # type: ignore

CONNECTION_PARAMETERS = {
    "account": os.environ.get("SF_ACCOUNT"),
    "user": os.environ.get("SF_UID"),
    "warehouse": os.environ.get("SF_WAREHOUSE"),
    "database": os.environ.get("SF_DATABASE"),
    "schema": os.environ.get("SF_SCHEMA"),
    "authenticator": "externalbrowser",
}


@pytest.fixture(scope="module")
def snowflake_session(request) -> Session:  # type: ignore
    if request.config.getoption("--snowflake-session") == "local":
        return Session.builder.config("local_testing", True).create()
    else:
        return Session.builder.configs(CONNECTION_PARAMETERS).create()


def pytest_addoption(parser):
    parser.addoption("--snowflake-session", action="store", default="integration")
