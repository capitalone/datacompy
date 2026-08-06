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

r"""DataComPy command line interface.

Invoked as ``datacompy`` once installed, or as ``python -m datacompy``.

Examples
--------
Compare two CSV files with the polars backend, which is the default:

.. code-block:: bash

    datacompy compare --left before.csv --right after.csv --on id

Emit a machine readable report for a CI pipeline and rely on the exit code:

.. code-block:: bash

    datacompy compare --left a.parquet --right b.parquet --on id,date \\
        --report-format json --max-unequal-rows 0

Write an HTML report to a file:

.. code-block:: bash

    datacompy compare --left a.csv --right b.csv --on id \\
        --report-format html --output report.html
"""

import argparse
from collections.abc import Callable, Sequence

from datacompy.cli.compare import run_compare
from datacompy.cli.errors import CLIError
from datacompy.cli.output import print_error
from datacompy.cli.parser import build_parser

#: Subcommand name to handler. Adding a command is additive.
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {"compare": run_compare}

__all__ = ["COMMANDS", "build_parser", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse *argv*, dispatch the subcommand, and return the exit code.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument list. When ``None``, argparse reads :data:`sys.argv`.

    Returns
    -------
    int
        ``0`` on a match, ``1`` on a mismatch, ``2`` on an expected error, and
        ``130`` on interrupt. Argparse exits with ``2`` itself on a parse
        failure, before this function returns.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    debug = getattr(args, "debug", False)
    try:
        return COMMANDS[args.command](args)
    except CLIError as exc:
        if debug:
            raise
        print_error(str(exc))
        return exc.exit_code
    except KeyboardInterrupt:
        print_error("interrupted")
        return 130
    # Anything else is an unexpected bug and propagates as a traceback.
