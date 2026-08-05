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

"""Utility and helper functions for data comparison."""

from types import ModuleType
from typing import Any

# Optional dependencies initialization
ps = None
sp = None

try:
    import pyspark as ps
except ImportError:
    pass

try:
    import snowflake.snowpark as sp
except ImportError:
    pass


_CONNECT_MODULE_PREFIX = "pyspark.sql.connect."


def is_spark_connect_object(spark_object: Any) -> bool:
    """Check whether an object belongs to the Spark Connect API.

    Walks the MRO comparing module names instead of using ``isinstance`` against
    the Spark Connect classes: importing ``pyspark.sql.connect`` runs
    ``check_dependencies()``, which raises ``ImportError`` when the optional
    ``grpcio`` dependency is missing. A classic-only PySpark installation must
    never pay that cost. Walking the MRO rather than only looking at
    ``type(obj).__module__`` also covers subclasses layered on top by other
    runtimes.

    Parameters
    ----------
    spark_object : Any
        Any object, typically a ``DataFrame`` or a ``Column``.

    Returns
    -------
    bool
        True if the object comes from the Spark Connect API, False otherwise.
    """
    return any(
        klass.__module__.startswith(_CONNECT_MODULE_PREFIX)
        for klass in type(spark_object).__mro__
    )


def get_spark_functions(spark_object: Any) -> ModuleType:
    """Get the ``functions`` module matching a classic or Spark Connect object.

    ``pyspark.sql.functions`` only forwards to the Spark Connect implementations
    when the process-global ``SPARK_CONNECT_MODE_ENABLED`` environment variable
    is set, which is not the case for a Connect session handed over by a
    notebook runtime or another framework. Selecting the module from the object
    being operated on is correct however the session was created.

    Parameters
    ----------
    spark_object : Any
        The ``DataFrame`` or ``Column`` the expression is being built for.

    Returns
    -------
    ModuleType
        ``pyspark.sql.connect.functions`` for a Spark Connect object,
        ``pyspark.sql.functions`` otherwise.
    """
    if is_spark_connect_object(spark_object):
        from pyspark.sql.connect import functions as connect_functions

        return connect_functions

    from pyspark.sql import functions as classic_functions

    return classic_functions


def get_spark_window(spark_object: Any) -> Any:
    """Get the ``Window`` class matching a classic or Spark Connect object.

    ``pyspark.sql.Window`` dispatches through ``dispatch_window_method`` on the
    same process-global flag that :func:`get_spark_functions` works around.

    Parameters
    ----------
    spark_object : Any
        The ``DataFrame`` or ``Column`` the window is being built for.

    Returns
    -------
    Any
        ``pyspark.sql.connect.window.Window`` for a Spark Connect object,
        ``pyspark.sql.Window`` otherwise.
    """
    if is_spark_connect_object(spark_object):
        from pyspark.sql.connect.window import Window as ConnectWindow

        return ConnectWindow

    from pyspark.sql import Window as ClassicWindow

    return ClassicWindow


def get_spark_column_dtypes(
    dataframe: "ps.sql.DataFrame", col_1: str, col_2: str
) -> tuple[str, str]:
    """Get the dtypes of two columns.

    Parameters
    ----------
    dataframe: pyspark.sql.DataFrame
        DataFrame to do comparison on
    col_1 : str
        The first column to look at
    col_2 : str
        The second column

    Returns
    -------
    tuple(str, str)
        Tuple of base and compare datatype
    """
    base_dtype = next(d[1] for d in dataframe.dtypes if d[0].upper() == col_1.upper())
    compare_dtype = next(
        d[1] for d in dataframe.dtypes if d[0].upper() == col_2.upper()
    )
    return base_dtype, compare_dtype


def get_snowflake_column_dtypes(
    dataframe: "sp.DataFrame", col_1: str, col_2: str
) -> tuple[str, str]:
    """Get the dtypes of two columns.

    Parameters
    ----------
    dataframe: sp.DataFrame
        DataFrame to do comparison on
    col_1 : str
        The first column to look at
    col_2 : str
        The second column

    Returns
    -------
    Tuple(str, str)
        Tuple of base and compare datatype
    """
    base_dtype = next(
        d[1] for d in dataframe.dtypes if d[0].strip('"').upper() == col_1.upper()
    )
    compare_dtype = next(
        d[1] for d in dataframe.dtypes if d[0].strip('"').upper() == col_2.upper()
    )
    return base_dtype, compare_dtype
