
Installation
============

A Conda environment or virtual environment is highly recommended:

conda (installs dependencies from Conda)
----------------------------------------

::

    conda create --name test python=3
    source activate test
    git clone https://github.com/capitalone/datacompy.git
    cd datacompy
    conda install --file requirements.txt
    pip install git+https://github.com/capitalone/datacompy.git


virtualenv (install dependencies from PyPI)
-------------------------------------------

::

    virtualenv env
    source env/bin/activate
    pip install --upgrade setuptools pip
    pip install git+https://github.com/capitalone/datacompy.git

