from setuptools import setup, find_packages
import os
import sys

CURR_DIR = os.path.abspath(os.path.dirname(__file__))
INSTALL_REQUIRES = [
    'enum34>=1.1.6;python_version<"3.4"',
    'pandas>=0.19.0',
    'numpy>=1.11.3',
    'six>=1.10']
with open(os.path.join(CURR_DIR, 'README.rst')) as file_open:
    LONG_DESCRIPTION = file_open.read()

exec(open('datacompy/_version.py').read())

setup(
    name='datacompy',
    version=__version__,
    description='Dataframe comparison in Python',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/capitalone/datacompy',
    license='Apache-2.0',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    package_data={'': ['templates/*']},
    zip_safe=False)
