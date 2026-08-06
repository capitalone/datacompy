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

"""End-to-end coverage of SparkSQLCompare against a Spark Connect session.

These tests must run in their own pytest process: starting a local Spark
Connect server sets ``SPARK_LOCAL_REMOTE``, after which every later
``SparkSession.builder.getOrCreate()`` in the process returns the Connect
session. They are therefore marked ``spark_connect`` and deselected from the
default run; see ``pytest-connect.ini`` and the CI workflow.

Nothing here may import ``pyspark.sql.functions``. Those helpers dispatch on
the same global flag this module deliberately clears, so using them to build
fixtures would break in the test rather than exercise the library.
"""

import os

import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytest.importorskip("grpc")

from datacompy.spark import SparkSQLCompare

pytestmark = pytest.mark.spark_connect


@pytest.fixture(scope="module")
def connect_session():
    """A Spark Connect session with PySpark's global Connect flag cleared.

    ``SparkSession.builder.remote(...)`` sets ``SPARK_CONNECT_MODE_ENABLED``,
    and while it is set ``pyspark.sql.functions`` transparently forwards to the
    Spark Connect implementations. Real Connect users -- a notebook runtime, a
    serverless runtime, a session handed over by another framework -- do not
    have that variable set, which is the situation that broke datacompy in
    issue #535. Clearing it is what makes these tests able to fail; leaving it
    set would make them pass against the unfixed library.
    """
    from pyspark.sql import SparkSession

    try:
        session = (
            SparkSession.builder.remote("local[2]")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.sql.adaptive.enabled", "false")
            .getOrCreate()
        )
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"could not start a local Spark Connect server: {exc}")

    os.environ.pop("SPARK_CONNECT_MODE_ENABLED", None)
    yield session
    # Restored before teardown: PySpark's own shutdown path consults is_remote().
    os.environ["SPARK_CONNECT_MODE_ENABLED"] = "1"
    session.stop()


def test_session_is_connect_without_the_global_flag(connect_session):
    """Guard: without this holding, every other test in the file is vacuous."""
    from pyspark.sql.utils import is_remote

    assert type(connect_session).__module__.startswith("pyspark.sql.connect")
    assert "SPARK_CONNECT_MODE_ENABLED" not in os.environ
    assert not is_remote()


def test_compare_and_report(connect_session):
    """The core path from issue #535, plus report() rendering."""
    df1 = connect_session.createDataFrame(
        pd.DataFrame(
            {
                "acct_id": [1, 2, 3],
                "name": ["george", "mike", "bob"],
                "amount": [100.0, 200.0, 300.0],
                "active": [True, False, True],
            }
        )
    )
    df2 = connect_session.createDataFrame(
        pd.DataFrame(
            {
                "acct_id": [1, 2, 4],
                "name": ["george", "MIKE", "sue"],
                "amount": [100.0, 200.5, 400.0],
                "active": [True, False, False],
            }
        )
    )
    compare = SparkSQLCompare(connect_session, df1, df2, join_columns="acct_id")

    assert not compare.matches()
    assert compare.count_matching_rows() == 1
    assert compare.intersect_rows.count() == 2

    report = compare.report()
    assert "DataComPy Comparison" in report
    assert "acct_id" in report


def test_compare_ignore_case_and_spaces(connect_session):
    """Reaches spark_normalize_string_column, which resolves from a Column."""
    df1 = connect_session.createDataFrame(
        pd.DataFrame({"acct_id": [1, 2], "name": ["  george  ", "mike"]})
    )
    df2 = connect_session.createDataFrame(
        pd.DataFrame({"acct_id": [1, 2], "name": ["GEORGE", "MIKE"]})
    )
    normalized = SparkSQLCompare(
        connect_session,
        df1,
        df2,
        join_columns="acct_id",
        ignore_spaces=True,
        ignore_case=True,
    )
    exact = SparkSQLCompare(connect_session, df1, df2, join_columns="acct_id")

    assert normalized.matches()
    assert not exact.matches()


def test_duplicate_join_keys(connect_session):
    """Exercises _generate_id_within_group, the only Window user."""
    df1 = connect_session.createDataFrame(
        pd.DataFrame({"acct_id": [1, 1, 2], "name": ["a", "a", "b"]})
    )
    df2 = connect_session.createDataFrame(
        pd.DataFrame({"acct_id": [1, 1, 2], "name": ["a", "z", "b"]})
    )
    compare = SparkSQLCompare(connect_session, df1, df2, join_columns="acct_id")

    assert not compare.matches()
    assert compare.count_matching_rows() == 2
    assert compare.report()


def test_array_column(connect_session):
    """Exercises SparkArrayLikeComparator."""
    df1 = connect_session.createDataFrame(
        [(1, [1, 2]), (2, [3])], "acct_id int, tags array<int>"
    )
    df2 = connect_session.createDataFrame(
        [(1, [1, 2]), (2, [9])], "acct_id int, tags array<int>"
    )
    compare = SparkSQLCompare(connect_session, df1, df2, join_columns="acct_id")

    assert not compare.matches()
    assert compare.count_matching_rows() == 1
    assert compare.report()


def test_numeric_tolerance(connect_session):
    """Exercises SparkNumericComparator, including the tolerance branch."""
    df1 = connect_session.createDataFrame(
        pd.DataFrame({"acct_id": [1, 2], "amount": [100.0, 200.0]})
    )
    df2 = connect_session.createDataFrame(
        pd.DataFrame({"acct_id": [1, 2], "amount": [100.01, 250.0]})
    )
    exact = SparkSQLCompare(connect_session, df1, df2, join_columns="acct_id")
    assert not exact.matches()
    assert exact.count_matching_rows() == 0

    # 0.01 is inside the tolerance, 50.0 is not.
    toleranced = SparkSQLCompare(
        connect_session, df1, df2, join_columns="acct_id", abs_tol=0.1
    )
    assert not toleranced.matches()
    assert toleranced.count_matching_rows() == 1


def test_mismatch_helpers(connect_session):
    """Exercises all_mismatch and sample_mismatch."""
    df1 = connect_session.createDataFrame(
        pd.DataFrame({"acct_id": [1, 2, 3], "amount": [100.0, 200.0, 300.0]})
    )
    df2 = connect_session.createDataFrame(
        pd.DataFrame({"acct_id": [1, 2, 3], "amount": [100.0, 999.0, 300.0]})
    )
    compare = SparkSQLCompare(connect_session, df1, df2, join_columns="acct_id")

    assert compare.all_mismatch().count() == 1
    assert compare.sample_mismatch("amount").count() == 1


def test_hide_sensitive_columns(connect_session):
    """Exercises hide_sensitive_columns."""
    df1 = connect_session.createDataFrame(
        pd.DataFrame({"acct_id": [1, 2], "ssn": ["111", "222"]})
    )
    df2 = connect_session.createDataFrame(
        pd.DataFrame({"acct_id": [1, 2], "ssn": ["111", "333"]})
    )
    compare = SparkSQLCompare(connect_session, df1, df2, join_columns="acct_id")
    compare.hide_sensitive_columns(["ssn"])

    report = compare.report()
    assert "*******" in report
    assert "222" not in report
    assert "333" not in report


def test_validate_dataframe_accepts_connect_dataframe(connect_session):
    """A Connect DataFrame must not be rejected by the type check."""
    df = connect_session.createDataFrame(pd.DataFrame({"acct_id": [1]}))

    # Would raise TypeError if the Connect DataFrame were not recognised.
    SparkSQLCompare(connect_session, df, df, join_columns="acct_id")

    with pytest.raises(TypeError):
        SparkSQLCompare(connect_session, "not a dataframe", df, join_columns="acct_id")
