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

"""Contains utilities for datacompy."""

from functools import wraps
from datacompy._typing import ArrowArrayLike
import pyarrow as pa


def check_module_available(module_available, extra_name):
    """Create a decorator to check if a module is available.

    This is for runtime checks in the library.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not module_available:
                raise ImportError(
                    f"The '{extra_name}' extra is not installed. Please install it to use this functionality, "
                    f"e.g. `pip install datacompy[{extra_name}]`"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator

def pyarrow_numeric (col: ArrowArrayLike) -> bool:
    """Check if a PyArrow Array is of numeric type.

    Parameters
    ----------
    col : ArrowArrayLike
        The PyArrow Array to check.

    Returns
    -------
    bool
        True if the Array is of numeric type, False otherwise.
    """
    if (
        pa.types.is_integer(col.type)
        or pa.types.is_floating(col.type)
        or pa.types.is_decimal128(col.type)
        or pa.types.is_decimal256(col.type)
    ):
        return True
    else:
        return False