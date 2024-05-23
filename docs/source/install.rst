
Installation
============

.. important::

    If you are using Python 3.12 and above, please note that not all functioanlity will be supported.
    Pandas and Polars support should work fine and are tested.


PyPI (basic)
------------

::

    pip install datacompy


A Conda environment or virtual environment is highly recommended:

conda (installs dependencies from Conda Forge)
----------------------------------------------

::

    conda create --name datacompy python=3.9 pip conda
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
