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

"""Tests for the Spark backend of the CLI.

These need ``datacompy[spark]`` and Java 17. The session is created and stopped
by the CLI itself, which is part of what is under test, so no session fixture is
shared with the rest of the suite.
"""

import pytest
from datacompy.cli import main

pytest.importorskip("pyspark", reason="requires datacompy[spark]")

MATCH = 0
MISMATCH = 1
ERROR = 2


@pytest.fixture
def no_borrowed_session():
    """Assert the CLI will be the one creating the session.

    A SparkSession is process wide, so the ownership tests below can only say
    anything if nothing else in the run has one open. ``pytest-spark`` supplies
    a session scoped ``spark_session`` fixture that would be exactly that, and
    today it is simply not instantiated yet when ``tests/cli`` runs. Skipping
    makes that dependency visible instead of turning it into a confusing
    failure if the collection order ever changes.
    """
    from pyspark.sql import SparkSession

    if SparkSession.getActiveSession() is not None:
        pytest.skip("a SparkSession is already active, so ownership cannot be asserted")
    yield


@pytest.fixture
def borrowed_session():
    """A session the CLI did not create, standing in for a notebook or Airflow task."""
    from pyspark.sql import SparkSession

    session = SparkSession.builder.appName("datacompy-cli-borrowed").getOrCreate()
    try:
        yield session
    finally:
        session.stop()


def test_spark_compares_csv_files(left_csv, right_csv, capsys):
    exit_code = main(
        [
            "compare",
            "--left",
            str(left_csv),
            "--right",
            str(right_csv),
            "--on",
            "id",
            "--backend",
            "spark",
        ]
    )
    assert exit_code == MISMATCH
    assert "DataComPy Comparison" in capsys.readouterr().out


def test_spark_matches_identical_files(left_csv, tmp_path, left_frame, capsys):
    copy = tmp_path / "copy.csv"
    left_frame.to_csv(copy, index=False)

    exit_code = main(
        [
            "compare",
            "--left",
            str(left_csv),
            "--right",
            str(copy),
            "--on",
            "id",
            "--backend",
            "spark",
        ]
    )
    assert exit_code == MATCH


def test_spark_session_is_stopped_even_when_loading_fails(
    no_borrowed_session, tmp_path, capsys
):
    """The session is registered on an ExitStack, so it closes on the error path too."""
    from pyspark.sql import SparkSession

    exit_code = main(
        [
            "compare",
            "--left",
            str(tmp_path / "absent.csv"),
            "--right",
            str(tmp_path / "absent.csv"),
            "--on",
            "id",
            "--backend",
            "spark",
        ]
    )
    assert exit_code == ERROR
    assert SparkSession.getActiveSession() is None


def test_spark_app_name_is_accepted(left_csv, right_csv, capsys):
    exit_code = main(
        [
            "compare",
            "--left",
            str(left_csv),
            "--right",
            str(right_csv),
            "--on",
            "id",
            "--backend",
            "spark",
            "--spark-app-name",
            "datacompy-cli-test",
        ]
    )
    assert exit_code == MISMATCH


def test_a_borrowed_session_survives_the_comparison(
    borrowed_session, left_csv, right_csv, capsys
):
    """``main`` must not stop a session it did not create.

    ``getOrCreate`` hands back the caller's session, and a SparkContext is
    process wide, so stopping it here would break a notebook or an Airflow task
    that called ``main`` in process and carried on afterwards.
    """
    from pyspark.sql import SparkSession

    exit_code = main(
        [
            "compare",
            "--left",
            str(left_csv),
            "--right",
            str(right_csv),
            "--on",
            "id",
            "--backend",
            "spark",
        ]
    )

    assert exit_code == MISMATCH
    assert SparkSession.getActiveSession() is not None
    # Still usable, not merely still referenced.
    assert borrowed_session.createDataFrame([(1,)], ["x"]).count() == 1


def test_cache_intermediates_changes_what_spark_does(
    borrowed_session, left_csv, right_csv, monkeypatch, capsys
):
    """The flag has to reach Spark, not just survive parsing.

    ``SparkSQLCompare`` caches ``intersect_rows`` and unpersists it again before
    the comparison returns, so nothing is left to inspect once the CLI exits.
    Counting ``cache()`` calls is what separates the two runs.

    The class to patch is taken from a DataFrame the session actually produces.
    ``cache`` is overridden on the concrete class, so patching the
    ``pyspark.sql.DataFrame`` base would never intercept, and the concrete
    class is not at the same import path across PySpark versions.
    """
    df_cls = type(borrowed_session.createDataFrame([(1,)], ["x"]))
    original = df_cls.cache
    calls = []

    def spy(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(df_cls, "cache", spy)

    argv = [
        "compare",
        "--left",
        str(left_csv),
        "--right",
        str(right_csv),
        "--on",
        "id",
        "--backend",
        "spark",
        "--quiet",
    ]

    assert main(argv) == MISMATCH
    cached_by_default = len(calls)

    calls.clear()
    assert main([*argv, "--no-cache-intermediates"]) == MISMATCH
    cached_when_disabled = len(calls)

    assert cached_by_default > 0, "expected caching to be enabled by default"
    assert cached_when_disabled == 0


def test_a_borrowed_session_survives_a_failed_comparison(
    borrowed_session, tmp_path, capsys
):
    """The error path unwinds the ExitStack too, and must not stop it either."""
    exit_code = main(
        [
            "compare",
            "--left",
            str(tmp_path / "absent.csv"),
            "--right",
            str(tmp_path / "absent.csv"),
            "--on",
            "id",
            "--backend",
            "spark",
        ]
    )

    assert exit_code == ERROR
    assert borrowed_session.createDataFrame([(1,)], ["x"]).count() == 1
