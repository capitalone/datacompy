from typing import Any

import pandas as pd
import polars as pl
import pytest
import fugue.api as fa
from datacompy import Compare, PolarsCompare
from datacompy.fsql import compare as fsql_compare

from ._utils import PerfTest


class FuguePerfTest(PerfTest):
    fugue_session: Any = None

    def load(self, path: str) -> Any:
        return fa.load(path)

    def run(self, base: Any, compare: Any) -> Any:
        return fsql_compare(base, compare, ["id"])

    def test_perf(self) -> None:
        with fa.engine_context(self.fugue_session):
            super().test_perf()


class TestFugueFromPathPerf(FuguePerfTest):
    @pytest.fixture(autouse=True)
    def _session(self, duckdb_session) -> None:
        self.fugue_session = duckdb_session

    def name(self) -> str:
        return "Fugue From Path"

    def load(self, path: str) -> Any:
        return path


class TestFugueDuckDBPerf(FuguePerfTest):
    @pytest.fixture(autouse=True)
    def _session(self, duckdb_session) -> None:
        self.fugue_session = duckdb_session

    def name(self) -> str:
        return "Fugue Duckdb"


class TestFuguePandasPerf(FuguePerfTest):
    def name(self) -> str:
        return "Fugue Pandas"

    def load(self, path: str) -> Any:
        return pd.read_parquet(path)


class TestFuguePolarsPerf(FuguePerfTest):
    def name(self) -> str:
        return "Fugue Polars"

    def load(self, path: str) -> Any:
        return pl.read_parquet(path)
