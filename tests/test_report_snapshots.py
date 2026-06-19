"""Snapshot regression tests for report() output across backends.

Each test case generates a report and compares it against a committed snapshot
in tests/snapshots/.  To regenerate all snapshots:

    DATACOMPY_REGEN_SNAPSHOTS=1 pytest tests/test_report_snapshots.py

Snapshots are named ``{backend}_{case}.txt``.  Any intentional change to
report output (e.g. adding a new section) must be accompanied by a snapshot
update committed alongside the code change.
"""

import os
from pathlib import Path

import pandas as pd
import polars as pl
from datacompy import PandasCompare, PolarsCompare

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
REGEN = os.environ.get("DATACOMPY_REGEN_SNAPSHOTS") == "1"


def _assert_or_regen(name: str, text: str) -> None:
    path = SNAPSHOT_DIR / f"{name}.txt"
    if REGEN or not path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    expected = path.read_text(encoding="utf-8")
    assert text == expected, (
        f"Report output for '{name}' changed.\n"
        f"Run with DATACOMPY_REGEN_SNAPSHOTS=1 if the change is intentional.\n"
        f"First diff at char {next((i for i, (a, b) in enumerate(zip(text, expected, strict=False)) if a != b), len(min(text, expected, key=len)))}"
    )


# ---------------------------------------------------------------------------
# Pandas
# ---------------------------------------------------------------------------


class TestPandasSnapshots:
    def test_no_mismatches(self):
        df1 = pd.DataFrame(
            {"id": [1, 2, 3], "val": [10, 20, 30], "name": ["a", "b", "c"]}
        )
        df2 = df1.copy()
        c = PandasCompare(df1, df2, "id", df1_name="left", df2_name="right")
        _assert_or_regen("pandas_no_mismatches", c.report())

    def test_with_mismatches(self):
        df1 = pd.DataFrame(
            {"id": [1, 2, 3], "val": [10, 20, 30], "score": [1.0, 2.0, 3.0]}
        )
        df2 = pd.DataFrame(
            {"id": [1, 2, 3], "val": [10, 99, 30], "score": [1.0, 2.5, 3.0]}
        )
        c = PandasCompare(df1, df2, "id")
        _assert_or_regen("pandas_with_mismatches", c.report())

    def test_unique_rows(self):
        df1 = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        df2 = pd.DataFrame({"id": [1, 2, 4], "val": [10, 20, 40]})
        c = PandasCompare(df1, df2, "id")
        _assert_or_regen("pandas_unique_rows", c.report())

    def test_unique_columns(self):
        df1 = pd.DataFrame({"id": [1, 2], "shared": [1, 2], "only_1": [9, 9]})
        df2 = pd.DataFrame({"id": [1, 2], "shared": [1, 2], "only_2": [8, 8]})
        c = PandasCompare(df1, df2, "id")
        _assert_or_regen("pandas_unique_columns", c.report())

    def test_on_index(self):
        df1 = pd.DataFrame({"val": [10, 20, 30]})
        df2 = pd.DataFrame({"val": [10, 99, 30]})
        c = PandasCompare(df1, df2, on_index=True)
        _assert_or_regen("pandas_on_index", c.report())

    def test_with_tolerances(self):
        df1 = pd.DataFrame({"id": [1, 2], "val": [1.0, 2.0]})
        df2 = pd.DataFrame({"id": [1, 2], "val": [1.0001, 2.0001]})
        c = PandasCompare(df1, df2, "id", abs_tol=0.001)
        _assert_or_regen("pandas_with_tolerances", c.report())

    def test_duplicates(self):
        df1 = pd.DataFrame({"id": [1, 1, 2], "val": [10, 20, 30]})
        df2 = pd.DataFrame({"id": [1, 1, 2], "val": [10, 20, 30]})
        c = PandasCompare(df1, df2, "id")
        _assert_or_regen("pandas_duplicates", c.report())

    def test_sample_count_zero(self):
        # sample_count=0 disables sample rows — output is fully deterministic
        df1 = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        df2 = pd.DataFrame({"id": [1, 2, 3], "val": [11, 22, 33]})
        c = PandasCompare(df1, df2, "id")
        _assert_or_regen("pandas_sample_count_zero", c.report(sample_count=0))


# ---------------------------------------------------------------------------
# Polars
# ---------------------------------------------------------------------------


class TestPolarsSnapshots:
    def test_no_mismatches(self):
        df1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        df2 = df1.clone()
        c = PolarsCompare(df1, df2, "id", df1_name="left", df2_name="right")
        _assert_or_regen("polars_no_mismatches", c.report())

    def test_with_mismatches(self):
        df1 = pl.DataFrame(
            {"id": [1, 2, 3], "val": [10, 20, 30], "score": [1.0, 2.0, 3.0]}
        )
        df2 = pl.DataFrame(
            {"id": [1, 2, 3], "val": [10, 99, 30], "score": [1.0, 2.5, 3.0]}
        )
        c = PolarsCompare(df1, df2, "id")
        _assert_or_regen("polars_with_mismatches", c.report())

    def test_unique_rows(self):
        df1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        df2 = pl.DataFrame({"id": [1, 2, 4], "val": [10, 20, 40]})
        c = PolarsCompare(df1, df2, "id")
        _assert_or_regen("polars_unique_rows", c.report())

    def test_unique_columns(self):
        df1 = pl.DataFrame({"id": [1, 2], "shared": [1, 2], "only_1": [9, 9]})
        df2 = pl.DataFrame({"id": [1, 2], "shared": [1, 2], "only_2": [8, 8]})
        c = PolarsCompare(df1, df2, "id")
        _assert_or_regen("polars_unique_columns", c.report())
