#
# Copyright 2020 Capital One Services, LLC
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

"""
Fugue SQL utils
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Tuple, Iterable

import duckdb
import fugue.api as fa
import pandas as pd
from fsspec.implementations.local import LocalFileSystem
from fugue import ExecutionEngine, NativeExecutionEngine
from triad.utils.io import url_to_fs

_CONF_GENERATORS: List[Tuple[Callable[[ExecutionEngine], bool], Any]] = []
_DUMMY_DF = pd.DataFrame([[0]], columns=["a"])


def _compare_conf_by_fugue_engine(
    check: Callable[[ExecutionEngine], bool],
) -> Any:
    def wrapper(func: Any) -> Any:
        _f = contextmanager(func)
        _CONF_GENERATORS.append((check, _f))
        return func

    return wrapper


@contextmanager
def infer_fugue_engine(df1: Any, df2: Any) -> Iterator[Dict[str, Any]]:
    infer_by = list(_get_infer_dfs(df1, df2))
    with fa.engine_context(infer_by=infer_by) as engine:
        for check, func in _CONF_GENERATORS:
            if check(engine):
                with func(engine) as conf:
                    yield conf
                return
        yield dict(
            engine=engine,
            persist_diff=False,
            use_map=False,
            num_buckets=1,
        )


@_compare_conf_by_fugue_engine(
    lambda e: not e.is_distributed and isinstance(e, NativeExecutionEngine)
)
def _on_native_engine(_: ExecutionEngine) -> Iterator[Dict[str, Any]]:
    with duckdb.connect() as con:
        with fa.engine_context(con) as e:
            yield dict(
                engine=e,
                persist_diff=False,
                use_map=False,
                num_buckets=1,
            )


try:
    from fugue_ray import RayExecutionEngine

    @_compare_conf_by_fugue_engine(lambda e: isinstance(e, RayExecutionEngine))
    def _on_ray_engine(engine: ExecutionEngine) -> Iterator[Dict[str, Any]]:
        yield dict(
            engine=engine,
            persist_diff=False,
            use_map=True,
            num_buckets=engine.get_current_parallelism() * 2,
        )

except (ImportError, ModuleNotFoundError):
    pass


def _get_infer_dfs(*dfs: Any) -> Iterable[Any]:
    for df in dfs:
        if isinstance(df, str):
            if _is_local_path(df):
                yield _DUMMY_DF
        else:
            yield df


def _is_local_path(path: Any) -> bool:
    try:
        fs, _ = url_to_fs(path)
        return isinstance(fs, LocalFileSystem)
    except Exception:
        return False
