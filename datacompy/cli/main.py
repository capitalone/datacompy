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

"""Entry point for the ``datacompy`` CLI.

``main()`` is the function registered as ``[project.scripts] datacompy``
in ``pyproject.toml`` and called by ``datacompy/cli/__main__.py`` for
``python -m datacompy`` invocations.
"""

from collections.abc import Sequence

from datacompy.cli.errors import CLIError
from datacompy.cli.output import print_error
from datacompy.cli.parser import build_parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse *argv* and dispatch the appropriate subcommand.

    Parameters
    ----------
    argv:
        Argument list.  When ``None`` :data:`sys.argv` is used by argparse.

    Returns
    -------
    int
        Exit code: ``0`` match, ``1`` mismatch, ``2`` error.
        Argparse itself calls ``sys.exit(2)`` on parse failures, which
        propagates before this function returns.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result: int = args.func(args)
        return result
    except CLIError as exc:
        if args.debug:
            raise
        print_error(str(exc))
        return exc.exit_code
    except FileNotFoundError as exc:
        if args.debug:
            raise
        print_error(f"file not found: {exc.filename}")
        return 2
