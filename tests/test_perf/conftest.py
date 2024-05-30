import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest

from ._constants import TEST_SIZES


@pytest.fixture(params=TEST_SIZES, scope="session")
def paired_files(request, tmpdir_factory) -> Any:
    folder = tmpdir_factory.mktemp("perf")
    f1, f2 = generate_files(request.param, str(folder))
    return f1, f2, request.param


def generate_dfs(size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(0)

    base = pd.DataFrame(
        dict(
            id=np.arange(0, size),
            a=np.random.randint(0, 10, size),
            b=np.random.rand(size),
            c=np.random.choice(["aaa", "bbb", "ccc"], size),
            d=np.random.randint(0, 10, size),
            e=np.random.rand(size),
            f=np.random.choice(["aaa", "bbb", "ccc"], size),
            g=np.random.randint(0, 10, size),
            h=np.random.rand(size),
            i=np.random.choice(["aaa", "bbb", "ccc"], size),
        )
    )

    compare = pd.DataFrame(
        dict(
            id=np.arange(0, size),
            d=np.random.randint(0, 10, size),
            e=np.random.rand(size),
            f=np.random.choice(["aaa", "bbb", "ccc"], size),
            g=np.random.randint(0, 10, size),
            h=np.random.rand(size),
            i=np.random.choice(["aaa", "bbb", "ccc"], size),
            j=np.random.randint(0, 10, size),
            k=np.random.rand(size),
            l=np.random.choice(["aaa", "bbb", "ccc"], size),
        )
    )

    return base, compare


def generate_files(size: int, folder: str) -> Tuple[str, str]:
    base, compare = generate_dfs(size)
    base_file = os.path.join(folder, "base.parquet")
    compare_file = os.path.join(folder, "compare.parquet")
    base.to_parquet(base_file, index=False)
    compare.to_parquet(compare_file, index=False)
    return base_file, compare_file
