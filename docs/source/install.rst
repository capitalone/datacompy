
Installation
============

.. note::

    Moving forward ``datacompy`` will not support Python 2. Please make sure you are using Python 3.8+


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
