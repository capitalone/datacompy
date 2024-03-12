from typing import Any

import pandas as pd

from datacompy import Compare

from ._utils import PerfTest


class TestPandasPerf(PerfTest):
    def name(self) -> str:
        return "Pandas"

    def load(self, path: str) -> Any:
        return pd.read_parquet(path)

    def run(self, base: Any, compare: Any) -> Any:
        compare = Compare(base, compare, ["id"])
        return compare.report()
