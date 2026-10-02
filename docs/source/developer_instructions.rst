Developer Instructions
======================

Guidance for developers.

Pre-Commit Hooks
----------------

We use the excellent `pre-commit <https://pre-commit.com/>`_ to run several hooks on all changes before commits.
``pre-commit`` is included in the ``dev`` extra installs. You'll have to run ``pre-commit install`` once per environment
before committing changes.

The reason behind running ruff, and others as a pre-commit hook is to let a machine make style decisions, based
on the collective wisdom of the Python community.

Generating Documentation
------------------------

You will need to ``pip install`` the ``dev`` requirements::

    pip install -e .[dev]

Then from the root of the repo you can type::

    make sphinx

This will automatically regenerate the api documentation using ``sphinx-apidoc``. The rendered documentation will be
stored in the ``/docs/build`` directory. The generated documentation is served from the ``gh-pages`` branch. Make sure
that the branch is clean and then to push to gh-pages you can type::

    make ghpages

Note about documentation: The `Numpy and Google style docstrings
<http://sphinx-doc.org/latest/ext/napoleon.html>`_ are activated by default.
Just make sure Sphinx 1.3 or above is installed.


Run unit tests
--------------

Run ``python -m pytest`` to run all tests defined in the ``tests`` subfolder.

CI runs the suite twice, once with the default ``pytest.ini`` and once with
``pytest-ansi.ini``, which differs only by enabling ``spark.sql.ansi.enabled``.
A change touching Spark casting or null handling should be run both ways::

    python -m pytest
    python -m pytest -c pytest-ansi.ini

The Spark tests need the ``spark`` extra and Java 17. Newer JDKs fail with
``py4j.protocol`` errors. If the JDK came from conda, ``JAVA_HOME`` has to point
at it, which a non-interactive shell will not inherit::

    export JAVA_HOME=$CONDA_PREFIX/lib/jvm


Local CI matrix with tox
------------------------

Running the matrix by hand means installing the right JDK per Spark version,
switching between ``pytest.ini``, ``pytest-ansi.ini`` and ``pytest-connect.ini``,
and re-pinning ``pyspark[connect]`` for each track. ``tox.ini`` does that for
you: it mirrors ``.github/workflows/test-package.yml`` job for job, and pulls
Java and PySpark from conda-forge via `tox-conda <https://github.com/tox-dev/tox-conda>`_
so no system JDK is required.

Using tox is optional. The plain ``pytest`` and ``make`` commands above still
work and are what the individual test targets are for.

Install the tooling, then run an environment::

    pip install "tox<4" tox-conda

    tox -e lint                      # ruff check + ruff format --check
    tox -e py312-nospark             # no pyspark; spark tests skip
    tox -e py312-spark4-pandas3     # Spark 4, ANSI mode, Spark Connect
    tox -e typecheck                  # mypy (not run by CI)
    tox                             # every env in envlist

Each environment maps to a CI job:

.. list-table::
   :header-rows: 1

   * - tox environment
     - CI job
     - What it covers
   * - ``lint``
     - ``lint-and-format``
     - ``ruff check`` and ``ruff format --check``
   * - ``py{310,311,312,313}-nospark``
     - ``test-basic-install``
     - pandas, polars, base, report and comparator tests; Spark and
       Snowflake tests skip via ``importorskip``
   * - ``py311-spark35-pandas2``
     - ``test-with-spark-3-install``
     - Spark 3.5 classic session, plus a second pass in ANSI mode
   * - ``py310-spark4-pandas2``
     - ``test-with-spark-4-install``
     - Spark 4 classic session
   * - ``py312-spark4-pandas3``
     - ``test-with-spark-4-install``
     - Spark 4 classic session, ANSI mode, and both Spark Connect suites

The Spark environments are the entire cost of a run, roughly twenty minutes
each, while the four no-Spark environments finish in under a minute. The
matrix therefore covers the Python, pandas and Spark axes independently
rather than as a cross product. Read the header of ``tox.ini`` before
widening it: it records which combinations are left out and what adding one
costs. Environments outside ``envlist`` still run on demand, so
``tox -e py313-spark4-pandas3`` works without being part of the default set.

Two constraints worth knowing before you edit ``tox.ini``:

- The Spark environments pin ``openjdk>=17.0.8,<18``. conda-forge ships
  GraalVM builds of openjdk 17 below that floor, and selecting one drags in
  ``graalpy-graalvm``, which replaces CPython with GraalPy and breaks pip.
  Do not relax the floor back to a bare ``openjdk=17``.
- A classic Spark session and a Spark Connect session cannot coexist in one
  pytest process: starting a local Connect server sets ``SPARK_LOCAL_REMOTE``,
  after which every later ``SparkSession.builder.getOrCreate()`` returns the
  Connect session. That is why the Connect suites are separate commands, and
  why ``SPARK_LOCAL_REMOTE`` is deliberately absent from ``passenv``.

Snowflake testing
-----------------

The Snowflake tests run either against a live Snowflake session or against
Snowpark's local testing mode::

    python -m pytest tests/test_snowflake.py
    python -m pytest tests/test_snowflake.py --snowflake-session local

Local testing mode is an emulator rather than Snowflake, and two of its
limitations matter here: ``eqNullSafe`` returns ``True`` for every row, and
high-precision decimals are truncated when a DataFrame is created. Tests that
depend on either request the ``requires_live_snowflake_session`` fixture, which
skips them in local mode. Changes to ``SnowflakeCompare`` still need a live
session to be fully validated, and that validation does not happen in CI.

A live session is built from the following environment variables, using
external browser authentication rather than a password:

- ``SF_ACCOUNT``: your Snowflake account
- ``SF_UID``: your Snowflake username
- ``SF_WAREHOUSE``: the warehouse to use
- ``SF_DATABASE``: a database you have access to
- ``SF_SCHEMA``: a schema belonging to that database


Management of Requirements
--------------------------

Requirements of the project should be added to ``pyproject.toml``.  Optional requirements used only for testing,
documentation, or code quality are added to ``pyproject.toml`` in the ``project.optional-dependencies`` section.



edgetest
--------

edgetest is a utility to help keep requirements up to date and ensure a subset of testing requirements still work.
More on edgetest `here <https://github.com/capitalone/edgetest>`_.

edgetest and tox do not interfere with each other. edgetest creates its
environments under ``.edgetest/`` in the repository root and tox creates
its own under ``.tox/``; both directories are gitignored, and neither tool
reads the other's configuration. The ``edgetest`` extra and the tox
environments can be installed side by side.

The ``pyproject.toml`` has configuration details on how to run edgetest. The process is automated by the
``edgetest`` GitHub Actions workflow, which opens a pull request with any dependency bumps it finds.

In order to execute edgetest locally you can run the following after install ``edgetest``:

.. code-block:: bash

    edgetest -c pyproject.toml --export

This should return output like the following and also updating ``pyproject.toml``:

.. code-block:: bash

    =============  ===============  ===================  =================
    Environment    Passing tests    Upgraded packages    Package version
    =============  ===============  ===================  =================
    core           True             boto3                1.21.7
    core           True             pandas               1.3.5
    core           True             PyYAML               6.0
    =============  ===============  ===================  =================




Release Guide
-------------

For ``datacompy`` we want to use a simple trunk-based workflow and follow
`Semantic Versioning <https://semver.org/>`_ for each release.

``main`` is the single active branch where all day-to-day development happens. All feature branches must be squash
merged into ``main``. The reason we squash merge is to keep the branch history clean and prevent it from being
polluted with interim commit messages. Squashing collapses all the commits into one single new commit, which also
makes it easier to back out changes if something breaks.

Releases are cut directly from ``main`` by tagging the desired commit with the appropriate version. Each tag should
correspond to a published artifact on PyPI that users can ``pip install``.

``gh-pages`` is where official documentation will go. After each release you should build the docs and push the HTML to
the pages branch. When first setting up the repo you want to make sure your gh-pages is a orphaned branch since it is
disconnected and independent from the code: ``git checkout --orphan gh-pages``.

The repo has a ``Makefile`` in the root folder which has helper commands such as ``make sphinx``, and
``make ghpages`` to help streamline building and pushing docs once they are setup right.


Generating distribution archives (PyPI)
---------------------------------------

After each release the package will need to be uploaded to PyPi. The instructions below are taken
from `packaging.python.org <https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives>`_

Update / Install ``build``, ``wheel``, and ``twine``::

    pip install --upgrade build wheel twine

Generate distributions::

    python -m build

Under the ``dist`` folder you should have something as follows::

    dist/
    datacompy-0.1.0-py3-none-any.whl
    datacompy-0.1.0.tar.gz


Finally upload to PyPi::

    # test pypi
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

    # real pypi
    twine upload dist/*
