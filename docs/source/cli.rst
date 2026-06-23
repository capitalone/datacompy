CLI Usage
=========

DataComPy ships a ``datacompy`` command-line tool for comparing two datasets
without writing a Python script.  It is installed automatically alongside the
library:

.. code-block:: bash

    pip install datacompy

Quickstart
----------

.. code-block:: bash

    # Compare two CSV files (Polars backend, default)
    datacompy compare --left before.csv --right after.csv --on id

    # Exit codes: 0 = match, 1 = mismatch, 2 = error
    echo $?

    # Python module form (useful when the script is not on PATH)
    python -m datacompy.cli compare --left before.csv --right after.csv --on id

Subcommands
-----------

``compare``
~~~~~~~~~~~

.. code-block:: text

    datacompy compare [OPTIONS]

Input flags
^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--left PATH``
     - Path or URI to the left (reference) dataset. **Required.**
   * - ``--right PATH``
     - Path or URI to the right (candidate) dataset. **Required.**
   * - ``--format {csv,parquet,json}``
     - Override format detection.  By default the format is inferred from
       the file extension (``*.csv`` / ``*.tsv``, ``*.parquet`` / ``*.pq``,
       ``*.json`` / ``*.jsonl``).
   * - ``--csv-delimiter CHAR``
     - Field delimiter for CSV files (default: comma).  Use ``';'`` for
       European CSVs or ``'\\t'`` for TSV files.  Both the escape sequence
       ``'\\t'`` and a literal tab character are accepted.

Join-key flags
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--on COL``
     - Join column name.  Repeat for composite keys:
       ``--on id --on date``.
   * - ``--on-index``
     - Join on the DataFrame index instead of columns.
       **Pandas backend only.**  Mutually exclusive with ``--on``.

Backend
^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--backend {pandas,polars,spark,snowflake}``
     - Comparison backend (default: **polars**).  Polars is faster and
       ships as a hard dependency.  Spark and Snowflake require their
       respective optional extras (``datacompy[spark]``,
       ``datacompy[snowflake]``).

Tolerances and comparison flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--abs-tol N``
     - Absolute tolerance for numeric comparisons (default 0.0).
   * - ``--rel-tol N``
     - Relative tolerance for numeric comparisons (default 0.0).
   * - ``--ignore-spaces``
     - Strip leading / trailing whitespace from string columns.
   * - ``--ignore-case``
     - Treat string comparisons as case-insensitive.
   * - ``--ignore-extra-columns``
     - Pass even if one dataset has columns the other lacks.
   * - ``--cast-column-names-lower`` / ``--no-cast-column-names-lower``
     - Normalise column names to lowercase before comparing
       (enabled by default).

Naming
^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--df1-name NAME``
     - Label for the left dataset in the report.  Defaults to the filename
       stem for file paths (``sales_data.csv`` → ``sales_data``) or the
       table name segment for Snowflake refs
       (``PROD.ANALYTICS.SALES_FACT`` → ``SALES_FACT``).
   * - ``--df2-name NAME``
     - Label for the right dataset in the report.  Same defaulting rules
       as ``--df1-name``.

Report shape
^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--sample-count N``
     - Maximum mismatch rows to sample per column (default 10).
   * - ``--column-count N``
     - Maximum columns to display in unique-row samples (default 10).
   * - ``--max-unequal-rows N``
     - Exit 0 if total differing rows ≤ *N*; exit 1 otherwise.  By default
       counts both value mismatches **and** rows that exist only in one
       dataset.  Must be a non-negative integer.
   * - ``--ignore-unique-rows``
     - With ``--max-unequal-rows``: exclude rows that exist only in one
       dataset from the count.  Only value mismatches in common rows are
       counted.

Output
^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--json``
     - Emit the full comparison report as a JSON object to stdout.
   * - ``--quiet``
     - Suppress all stdout output.  The exit code still reflects the
       comparison result.  Has no effect when ``--json`` is also given
       (JSON is always emitted when ``--json`` is set).

To write a report to a file, use shell redirection:

.. code-block:: bash

    datacompy compare --left a.csv --right b.csv --on id > report.txt
    datacompy compare --left a.csv --right b.csv --on id --json > report.json

Exit codes
----------

.. list-table::
   :widths: 10 90
   :header-rows: 1

   * - Code
     - Meaning
   * - ``0``
     - The datasets match (within any specified tolerances / thresholds).
   * - ``1``
     - The datasets differ, or ``--max-unequal-rows`` was violated.
   * - ``2``
     - An error occurred (bad arguments, missing file, import error, etc.).

Backend-specific notes
-----------------------

Spark
~~~~~

The Spark backend builds a local ``SparkSession`` via
``SparkSession.builder.getOrCreate()``.  Java 17 must be on ``PATH``.
PySpark's INFO/WARN output is suppressed at the ``ERROR`` level by default;
set ``DATACOMPY_SPARK_LOG_LEVEL=INFO`` to restore verbose logs.

Install the Spark extra before using ``--backend spark``:

.. code-block:: bash

    pip install "datacompy[spark]"

Snowflake
~~~~~~~~~

The Snowflake backend builds a Snowpark session from environment variables:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Notes
   * - ``SNOWFLAKE_ACCOUNT``
     - **Required.**
   * - ``SNOWFLAKE_USER``
     - **Required.**
   * - ``SNOWFLAKE_PASSWORD``
     - Required unless ``SNOWFLAKE_AUTHENTICATOR=externalbrowser``.
   * - ``SNOWFLAKE_AUTHENTICATOR``
     - Optional.  Set to ``externalbrowser`` for SSO.
   * - ``SNOWFLAKE_ROLE``
     - Optional.
   * - ``SNOWFLAKE_WAREHOUSE``
     - Optional.
   * - ``SNOWFLAKE_DATABASE``
     - Optional.
   * - ``SNOWFLAKE_SCHEMA``
     - Optional.

Override all variables with a JSON connection-parameter file:

.. code-block:: bash

    datacompy compare \
        --left DB.SCHEMA.TABLE_BEFORE \
        --right DB.SCHEMA.TABLE_AFTER \
        --on ID \
        --backend snowflake \
        --snowflake-config ~/.snowflake/conn.json

Pass ``DB.SCHEMA.TABLE`` (fully-qualified) or ``SCHEMA.TABLE`` (2-part)
identifiers directly as ``--left`` / ``--right`` values.  When a 2-part
ref is given, the CLI expands it to 3 parts using the session's current
database (set via ``SNOWFLAKE_DATABASE`` or the connection config file).
If no current database is set and a 2-part ref is used, the CLI exits
with an error asking you to use the fully-qualified form.
Local files are uploaded to temporary Snowflake tables automatically.

Install the Snowflake extra:

.. code-block:: bash

    pip install "datacompy[snowflake]"

Cloud paths
~~~~~~~~~~~

``s3://``, ``gs://``, and ``abfs://`` URIs are forwarded as-is to the
Pandas / Polars reader.  They require optional filesystem libraries:

.. code-block:: bash

    pip install s3fs        # Amazon S3
    pip install gcsfs       # Google Cloud Storage
    pip install adlfs       # Azure Data Lake Storage

CICD recipes
------------

GitHub Actions
~~~~~~~~~~~~~~

.. code-block:: yaml

    - name: Compare datasets
      run: |
        datacompy compare \
          --left data/expected.csv \
          --right data/actual.csv \
          --on id \
          --max-unequal-rows 0

Airflow BashOperator
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    compare_task = BashOperator(
        task_id="compare_datasets",
        bash_command=(
            "datacompy compare "
            "--left {{ params.left }} "
            "--right {{ params.right }} "
            "--on id --json > /tmp/report.json"
        ),
        params={"left": "s3://bucket/before.parquet", "right": "s3://bucket/after.parquet"},
    )

Limitations
-----------

- **Per-column tolerances** are not yet supported via the CLI.  Use the
  Python API to pass ``abs_tol={"col_a": 0.01}`` directly.
- **File output** (e.g. ``--output report.html``) is not yet supported.
  Use shell redirection (``> report.txt``, ``> report.json``) instead.
- **Mixed file formats** — ``--format`` applies to both sides.  If your
  left and right files have different formats (e.g. a CSV reference vs a
  Parquet snapshot), convert one before comparing.  Per-side
  ``--left-format`` / ``--right-format`` flags are planned for a future
  release.
- **Empty DataFrames** — comparing two datasets that both have zero rows
  currently exits 1 (mismatch) due to a known behaviour in the underlying
  comparison engine.  This will be addressed in a future library release.
