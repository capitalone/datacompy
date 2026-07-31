
Installation
============

DataComPy requires Python 3.10 or later. Every backend is tested against each
supported version.

PyPI (basic)
------------

::

    pip install datacompy

Installing extras
-----------------

Pandas and Polars work out of the box. Spark and Snowflake are optional::

    pip install datacompy[spark]
    pip install datacompy[snowflake]

.. note::

    On Python 3.12 and above the ``spark`` extra resolves to PySpark 4. The
    dependency markers pick the right version, so nothing extra is needed.

Installing the package also provides the ``datacompy`` command line tool. See
:doc:`cli`.


A Conda environment or virtual environment is highly recommended:

conda (installs dependencies from Conda Forge)
----------------------------------------------

::

    conda create --name datacompy python=3.10 pip conda
    source activate datacompy
    conda config --add channels conda-forge
    conda install datacompy


virtualenv (install dependencies from PyPI)
-------------------------------------------

::

    virtualenv env
    source env/bin/activate
    pip install --upgrade setuptools pip
    pip install datacompy
