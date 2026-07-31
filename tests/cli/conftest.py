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

"""Shared fixtures for the CLI tests.

The datasets are deliberately tiny and their differences are known exactly, so
the threshold tests can assert on specific counts:

- ``id`` 1 matches on every column
- ``id`` 2 differs in ``amount`` by 0.005
- ``id`` 3 exists only on the left
- ``id`` 4 exists only on the right

That is 1 unequal row plus 2 unique rows, so 3 differing rows in total.
"""

import pandas as pd
import pytest

LEFT_ROWS = [
    {"id": 1, "name": "alice", "amount": 10.0},
    {"id": 2, "name": "bob", "amount": 20.0},
    {"id": 3, "name": "carol", "amount": 30.0},
]

RIGHT_ROWS = [
    {"id": 1, "name": "alice", "amount": 10.0},
    {"id": 2, "name": "bob", "amount": 20.005},
    {"id": 4, "name": "dave", "amount": 40.0},
]

#: Number of rows that differ between LEFT_ROWS and RIGHT_ROWS.
UNEQUAL_ROWS = 1
UNIQUE_ROWS = 2
TOTAL_DIFFERING_ROWS = UNEQUAL_ROWS + UNIQUE_ROWS


@pytest.fixture
def left_frame() -> pd.DataFrame:
    """Return the left hand dataset."""
    return pd.DataFrame(LEFT_ROWS)


@pytest.fixture
def right_frame() -> pd.DataFrame:
    """Return the right hand dataset, which differs from the left."""
    return pd.DataFrame(RIGHT_ROWS)


@pytest.fixture
def left_csv(tmp_path, left_frame):
    """Write the left dataset to a CSV file and return its path."""
    path = tmp_path / "left.csv"
    left_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def right_csv(tmp_path, right_frame):
    """Write the right dataset to a CSV file and return its path."""
    path = tmp_path / "right.csv"
    right_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def cli(left_csv, right_csv):
    """Return a callable that runs ``datacompy compare`` on the CSV fixtures.

    Extra arguments are appended, and ``--left`` / ``--right`` / ``--on`` can be
    overridden by passing them explicitly.
    """
    from datacompy.cli import main

    def _run(*extra: str) -> int:
        argv = ["compare"]
        if "--left" not in extra:
            argv += ["--left", str(left_csv)]
        if "--right" not in extra:
            argv += ["--right", str(right_csv)]
        if "--on" not in extra and "--on-index" not in extra:
            argv += ["--on", "id"]
        return main([*argv, *extra])

    return _run
