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

"""CLI integration tests for the Spark backend.

Skipped automatically when PySpark is not installed.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

pyspark = pytest.importorskip("pyspark")


def test_spark_match_exits_0(
    csv_match: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_match
    code, out, _ = cli(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--backend",
            "spark",
        ]
    )
    assert code == 0
    assert "DataComPy Comparison" in out


def test_spark_mismatch_exits_1(
    csv_mismatch: tuple[Path, Path],
    cli: Callable[[list[str]], tuple[int, str, str]],
) -> None:
    left, right = csv_mismatch
    code, _, _ = cli(
        [
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--on",
            "id",
            "--backend",
            "spark",
        ]
    )
    assert code == 1
