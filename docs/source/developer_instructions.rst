Developer Instructions
======================

Guidance for developers.

Pre-Commit Hooks
----------------

We use the excellent `pre-commit <https://pre-commit.com/>`_ to run the excellent
`black <https://github.com/ambv/black>`_ on all changes before commits.  ``pre-commit`` is included
in the test requirements below, and you'll have to run ``pre-commit install`` once per environment
before committing changes, or else manually install ``black`` and run it.  If you have ``pre-commit``
installed, trying to commit a change will first run black against any changed Python files, and force
you to add/commit any changes.

The reason behind running black as a pre-commit hook is to let a machine make style decisions, based
on the collective wisdom of the Python community.  The only change made from the default black setup
is to allow lines up to 100 characters long.

Generating Documentation
------------------------

You will need to ``pip install`` the test requirements::

    pip install -r test-requirements.txt

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

Requirements of the project should be added to ``requirements.txt``.  Optional
requirements used only for testing are added to ``test-requirements.txt``.


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
