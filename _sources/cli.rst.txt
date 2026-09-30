Command Line Interface
======================

DataComPy ships a ``datacompy`` command so you can compare two datasets without
writing a script. It is aimed at ad hoc checks from a shell and at CI pipelines
that run as shell tasks, such as an Airflow ``BashOperator``, a GitHub Actions
step, or a GitLab CI job.

Quick start
-----------

.. code-block:: bash

    datacompy compare --left before.csv --right after.csv --on id

The command is also available as a module, which is handy when the console
script is not on ``PATH``:

.. code-block:: bash

    python -m datacompy compare --left before.csv --right after.csv --on id

Exit codes
----------

The exit code is the contract for automation.

======= ==============================================================
Code    Meaning
======= ==============================================================
``0``   The datasets match, or stay within ``--max-unequal-rows``
``1``   The datasets differ, or the threshold was exceeded
``2``   Bad arguments, unreadable input, or a missing optional backend
``130`` Interrupted
======= ==============================================================

Anything unexpected propagates as a traceback. Pass ``--debug`` to see the full
traceback for an error that would otherwise be reported as a short message.

Choosing a backend
------------------

``--backend`` selects the comparison engine. Polars is the default.

============== ================================================================
Backend        Use it for
============== ================================================================
``polars``     The default. Fast, in memory, no extra install
``pandas``     Index based joins (``--on-index``), or wider ecosystem parity
``spark``      Distributed data. Needs ``datacompy[spark]`` and Java 17
``snowflake``  Comparing two tables in place. Needs ``datacompy[snowflake]``
============== ================================================================

Inputs
------

``--left`` and ``--right`` accept local paths and cloud URIs. CSV, Parquet, and
JSON are supported, including tab separated CSV via a ``.tsv`` extension and
newline delimited JSON via a ``.jsonl`` or ``.ndjson`` extension.

The format is inferred per file from its extension, so mixed inputs work without
any extra flags:

.. code-block:: bash

    datacompy compare --left snapshot.csv --right snapshot.parquet --on id

The extensions recognised are ``.csv``, ``.tsv``, ``.parquet``, ``.pq``,
``.json``, ``.jsonl``, and ``.ndjson``. The delimiter is inferred the same way,
a tab for ``.tsv`` and a comma for everything else, so a tab separated file
compares against a comma separated one without any extra flags.

Use ``--input-format`` when the extension is missing or unusual. It selects the
reader only and says nothing about the delimiter, so pair it with
``--csv-delimiter``, which overrides inference for both inputs:

.. code-block:: bash

    datacompy compare --left extract.dat --right extract2.dat --on id \
        --input-format csv --csv-delimiter '\t'

``--csv-delimiter`` is also the way to correct a misleading extension. A comma
separated file named ``.tsv`` would otherwise be read with tabs, so force the
comma explicitly:

.. code-block:: bash

    datacompy compare --left export.tsv --right export2.tsv --on id \
        --csv-delimiter ','

A file read with the wrong delimiter collapses into a single column, which
surfaces as a missing join column. The CLI warns on stderr when it sees that,
naming the file and the delimiter it used.

Cloud URIs such as ``s3://``, ``gs://``, and ``abfs://`` are handed straight to
the underlying reader, so they work once the matching filesystem library
(``s3fs``, ``gcsfs``, ``adlfs``) is installed.

For ``--backend snowflake``, ``--left`` and ``--right`` are always table
references, either ``db.schema.table`` or ``schema.table``. A two part reference
is qualified with the session's current database. The CLI does not read local
files into Snowflake; use the pandas or polars backend for files, or load the
data into a table first.

.. code-block:: bash

    datacompy compare --backend snowflake \
        --left PROD.ANALYTICS.SALES \
        --right STAGE.ANALYTICS.SALES \
        --on sale_id

Join keys
---------

``--on`` accepts a comma separated list, a repeated flag, or a mix of the two:

.. code-block:: bash

    datacompy compare --left a.csv --right b.csv --on id,date
    datacompy compare --left a.csv --right b.csv --on id --on date

Use the repeated form for column names that contain a comma.

``--on-index`` joins on the DataFrame index instead, and is only available with
``--backend pandas``.

Tolerances
----------

``--abs-tol`` and ``--rel-tol`` take either a single number that applies to every
numeric column, or repeated ``COLUMN=VALUE`` pairs for per column tolerances:

.. code-block:: bash

    datacompy compare --left a.parquet --right b.parquet --on account_id \
        --abs-tol 0.01 --rel-tol 0.001

    datacompy compare --left a.parquet --right b.parquet --on account_id \
        --abs-tol price=0.01 --abs-tol quantity=0

The two forms cannot be mixed on the same flag, because the library takes either
a single tolerance or a per column mapping.

Normalisation
-------------

.. code-block:: bash

    datacompy compare --left a.csv --right b.csv --on id \
        --ignore-spaces --ignore-case

``--ignore-extra-columns`` treats the datasets as matching even when one side has
columns the other does not. Column names are lowercased before comparison by
default; pass ``--no-cast-column-names-lower`` to compare them as written. That
flag does not apply to Snowflake, which normalises identifiers to uppercase
itself.

Reports
-------

Rendering and destination are separate. ``--report-format`` picks between
``text`` (the default), ``json``, and ``html``. ``--output`` writes to a file
instead of, or as well as, stdout.

.. code-block:: bash

    # Human readable, to the terminal
    datacompy compare --left a.csv --right b.csv --on id

    # Machine readable, piped into another tool
    datacompy compare --left a.csv --right b.csv --on id \
        --report-format json | jq '.row_summary.unequal_rows'

    # An HTML report saved for a build artifact, nothing on stdout
    datacompy compare --left a.csv --right b.csv --on id \
        --report-format html --output reports/diff.html --quiet

``--quiet`` suppresses stdout only. A file named by ``--output`` is always
written, and parent directories are created as needed. The exit code is
unaffected by either flag.

``--sample-count`` and ``--column-count`` control how many sample rows and
columns the report shows.

Failing a build
---------------

Without a threshold, any difference exits ``1``. ``--max-unequal-rows`` lets a
known amount of drift pass:

.. code-block:: bash

    # Fail on any difference at all
    datacompy compare --left before.parquet --right after.parquet --on id \
        --max-unequal-rows 0 --quiet

    # Tolerate up to 5 differing rows
    datacompy compare --left before.parquet --right after.parquet --on id \
        --max-unequal-rows 5 --quiet

By default the count includes both value mismatches and rows present in only one
dataset. Add ``--ignore-unique-rows`` to count value mismatches in common rows
only:

.. code-block:: bash

    datacompy compare --left before.parquet --right after.parquet --on id \
        --max-unequal-rows 0 --ignore-unique-rows --quiet

A threshold run also fails when one side has extra columns, unless
``--ignore-extra-columns`` is given.

A GitHub Actions step looks like this:

.. code-block:: yaml

    - name: Check the nightly load against the previous snapshot
      run: |
        datacompy compare \
          --left s3://warehouse/snapshots/previous.parquet \
          --right s3://warehouse/snapshots/current.parquet \
          --on account_id,as_of_date \
          --abs-tol balance=0.01 \
          --max-unequal-rows 0 \
          --report-format json \
          --output reports/diff.json

Backend credentials
-------------------

Spark
~~~~~

The CLI creates its own session and stops it when the command finishes, on both
the success and the failure path. A session that already exists is borrowed and
left running, so calling the CLI from a process that owns a session is safe.
``--spark-app-name`` sets the application name, and has no effect when a session
already exists. PySpark's INFO and WARN logging is suppressed so it does not mix
with the report; set ``DATACOMPY_SPARK_LOG_LEVEL`` to override that.

Intermediate DataFrames are cached by default. Pass
``--no-cache-intermediates`` on Databricks Serverless and other environments
that do not support caching.

Snowflake
~~~~~~~~~

Connection parameters come either from a JSON file or from the environment.

.. code-block:: bash

    datacompy compare --backend snowflake \
        --left PROD.ANALYTICS.SALES --right STAGE.ANALYTICS.SALES --on sale_id \
        --snowflake-config ~/.snowflake/connection.json

The JSON file holds Snowpark connection parameters as top level keys, for
example ``account``, ``user``, ``password``, ``role``, ``warehouse``,
``database``, and ``schema``.

Without ``--snowflake-config``, the session is built from the environment:

==================================  ===============================================
Variable                            Notes
==================================  ===============================================
``SNOWFLAKE_ACCOUNT``               Required
``SNOWFLAKE_USER``                  Required, except under OAuth
``SNOWFLAKE_PASSWORD``              Required unless a token or authenticator is set
``SNOWFLAKE_TOKEN``                 OAuth access token; implies OAuth on its own
``SNOWFLAKE_AUTHENTICATOR``         ``oauth``, or SSO such as ``externalbrowser``
``SNOWFLAKE_ROLE``                  Optional
``SNOWFLAKE_WAREHOUSE``             Optional
``SNOWFLAKE_DATABASE``              Optional, qualifies two part references
``SNOWFLAKE_SCHEMA``                Optional
==================================  ===============================================

Full option list
----------------

.. code-block:: bash

    datacompy compare --help
