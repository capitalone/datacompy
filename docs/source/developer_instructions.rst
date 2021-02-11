Developer Instructions
======================

Guidance for developers.

Pre-Commit Hooks
----------------

We use the excellent `pre-commit <https://pre-commit.com/>`_ to run several hooks on all changes before commits.
``pre-commit`` is included in the ``dev`` extra installs. You'll have to run ``pre-commit install`` once per environment
before committing changes.

The reason behind running black, isort, and others as a pre-commit hook is to let a machine make style decisions, based
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

Run ``python -m pytest`` to run all unittests defined in the subfolder
``tests`` with the help of `py.test <http://pytest.org/>`_ and
`pytest-runner <https://pypi.python.org/pypi/pytest-runner>`_.


Management of Requirements
--------------------------

Requirements of the project should be added to ``requirements.txt``.  Optional requirements used only for testing,
documentation, or code quality are added to ``setup.py`` and ``EXTRAS_REQUIRE``


Release Guide
-------------

For ``datacompy`` we want to use a simple workflow branching style and follow
`Semantic Versioning <https://semver.org/>`_ for each release.

``develop`` is the default branch where most people will work with day to day. All features must be squash merged into
this branch. The reason we squash merge is to prevent the develop branch from being polluted with endless commit messages
when people are developing. Squashing collapses all the commits into one single new commit. It will also make it much easier to
back out changes if something breaks.

``master`` is where official releases will go. Each release on ``master`` should be tagged properly to denote a "version"
that will have the corresponding artifact on pypi for users to ``pip install``.

``gh-pages`` is where official documentation will go. After each release you should build the docs and push the HTML to
the pages branch. When first setting up the repo you want to make sure your gh-pages is a orphaned branch since it is
disconnected and independent from the code: ``git checkout --orphan gh-pages``.

The repo has a ``Makefile`` in the root folder which has helper commands such as ``make sphinx``, and
``make ghpages`` to help streamline building and pushing docs once they are setup right.


Generating distribution archives (PyPI)
---------------------------------------

After each release the package will need to be uploaded to PyPi. The instructions below are taken
from `packaging.python.org <https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives>`_

Update / Install ``setuptools``, ``wheel``, and ``twine``::

    pip install --upgrade setuptools wheel twine

Generate distributions::

    python setup.py sdist bdist_wheel

Under the ``dist`` folder you should have something as follows::

    dist/
    datacompy-0.1.0-py3-none-any.whl
    datacompy-0.1.0.tar.gz


Finally upload to PyPi::

    # test pypi
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

    # real pypi
    twine upload dist/*