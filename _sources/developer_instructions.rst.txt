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

Then enter the ``sphinx`` directory and type::

    make html

This will automatically regenerate the api documentation using ``sphinx-apidoc``.
The rendered documentation will be stored in the ``/docs`` directory.  The
GitHub pages endpoint
is served from the ``/docs`` folder on the master branch, so merge into master
to see the rendered docs changes.

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
