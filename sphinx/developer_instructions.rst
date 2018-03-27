Developer Instructions
======================

Guidance for developers.

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

