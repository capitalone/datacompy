Installation
============


``datacompy`` supports the following configuration:

- python ``3.8.*`` - ``3.11.*``
- pyspark ``3.4.0``
- pandas ``1.3.4`` - ``1.5.3`` (Test matrix includes: ``1.5.3, 1.4.4, 1.3.5``)

.. note::

    Moving forward ``datacompy`` will not support Python 2. Please make sure you are using Python 3.8+

.. note::

    pandas ``2.0`` + is not supported due to a bug: https://issues.apache.org/jira/browse/SPARK-43194

.. note::

    Due to compatibility issues datacompy only supports pyspark ``3.4.0`` currently


PyPI (basic)
------------

::

    pip install datacompy


A Conda environment or virtual environment is highly recommended:

conda (installs dependencies from Conda Forge)
----------------------------------------------

::

    conda create --name test python=3.9
    source activate test
    conda config --add channels conda-forge
    conda install datacompy


virtualenv (install dependencies from PyPI)
-------------------------------------------

::

    virtualenv env
    source env/bin/activate
    pip install --upgrade setuptools pip
    pip install datacompy
