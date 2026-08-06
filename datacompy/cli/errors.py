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

"""Exception hierarchy for the DataComPy command line interface.

Every exception defined here carries an ``exit_code`` so that
:func:`datacompy.cli.main` can catch :class:`CLIError` at a single site and
translate it into a friendly message plus the right process exit code.
"""


class CLIError(Exception):
    """Base class for expected CLI failures. Always maps to exit code 2."""

    exit_code: int = 2


class BadArgsError(CLIError):
    """Raised when arguments are individually valid but invalid in combination."""


class LoadError(CLIError):
    """Raised when a dataset cannot be read."""


class MissingExtraError(CLIError):
    """Raised when a backend needs an optional dependency that is not installed."""


class OutputError(CLIError):
    """Raised when the report cannot be written to the requested destination."""
