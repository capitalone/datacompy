from typing import Any

import polars as pl
from datacompy import PolarsCompare

from ._utils import PerfTest


class TestPolarsPerf(PerfTest):
    def name(self) -> str:
        return "Polars"

    def load(self, path: str) -> Any:
        return pl.read_parquet(path)

    def run(self, base: Any, compare: Any) -> Any:
        compare = PolarsCompare(base, compare, ["id"])
        return compare.report()
