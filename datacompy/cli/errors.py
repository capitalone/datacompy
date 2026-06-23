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

"""CLI-specific exception hierarchy.

All exceptions here carry an ``exit_code`` of ``2`` so that ``main()``
can catch them at a single site and return the right exit code.
"""


class CLIError(Exception):
    """Base class for all CLI errors.  Always maps to exit code 2."""

    exit_code: int = 2


class BadArgsError(CLIError):
    """Raised when the user provides logically invalid argument combinations."""


class LoadError(CLIError):
    """Raised when a source file cannot be read."""


class MissingExtraError(CLIError):
    """Raised when a required optional dependency is not installed."""
