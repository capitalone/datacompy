from ._utils import generate_dfs
from datacompy import Compare
from datacompy.fsql import compare as fsql_compare
import pytest


@pytest.fixture(params=[1000, 100000])
def paired_dfs(benchmark, request):
    benchmark.name = "test - %s" % request.param
    return generate_dfs(request.param)


def v1_df_pandas_perf(base, compare):
    compare = Compare(base, compare, ["id"])
    return compare.report()


def v2_df_fsql_perf(base, compare):
    res= fsql_compare(base, compare, "id")
    return res


def _test_v1_pandas_perf(benchmark, paired_dfs):
    base, compare = paired_dfs
    benchmark.pedantic(
        v1_df_pandas_perf,
        args=(base, compare),
        iterations=1,
        rounds=10,
        warmup_rounds=2,
    )


def test_v2_df_fsql_perf(benchmark, paired_dfs):
    base, compare = paired_dfs
    benchmark.pedantic(
        v2_df_fsql_perf,
        args=(base, compare),
        iterations=1,
        rounds=10,
        warmup_rounds=2,
    )
