#
# Copyright 2025 Capital One Services, LLC
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
