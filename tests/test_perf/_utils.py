from typing import Any

import pytest


class PerfTest:
    benchmark_params = dict(
        iterations=1,
        rounds=10,
        warmup_rounds=2,
    )

    @pytest.fixture(autouse=True)
    def _fixt(self, benchmark, paired_files) -> None:
        self.base_path, self.compare_path, self.n = paired_files
        self.benchmark = benchmark
        self.benchmark.group = self.name()
        self.benchmark.name = str(self.n)

    def name(self) -> str:
        raise NotImplementedError

    def load(self, path: str) -> Any:
        raise NotImplementedError

    def run(self, base: Any, compare: Any) -> Any:
        raise NotImplementedError

    def test_perf(self) -> None:
        base = self.load(self.base_path)
        compare = self.load(self.compare_path)
        self.benchmark.pedantic(self.run, args=(base, compare), **self.benchmark_params)
