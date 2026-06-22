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

"""DataComPy command-line interface.

Entry point: ``datacompy`` (installed via ``[project.scripts]``) or
``python -m datacompy``.

Examples
--------
Compare two CSV files using the Polars backend (default):

.. code-block:: bash

    datacompy compare --left a.csv --right b.csv --on id

Emit a JSON report to stdout:

.. code-block:: bash

    datacompy compare --left a.csv --right b.csv --on id --json

Use Pandas and compare on the DataFrame index:

.. code-block:: bash

    datacompy compare --left a.csv --right b.csv --on-index --backend pandas
"""

from datacompy.cli.main import main
from datacompy.cli.parser import build_parser

__all__ = ["build_parser", "main"]
