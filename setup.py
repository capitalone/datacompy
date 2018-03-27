from setuptools import setup, find_packages
import sys


def read(filename):
    with open(filename, 'r') as f:
        return f.read()

def clean_requirements(requirements):
    """I know this is a kludge - just want to get it working and can revisit"""
    output = []
    for line in requirements:
        if not(line.startswith('#' or line == '')):
            if ';' in line:
                add_on_req = line.split(';')[1]
                python_version = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)
                if eval(add_on_req):
                    output.append(line.split(';')[0])
            else:
                output.append(line)

requirements = read('requirements.txt').strip().split('\n')
requirements = clean_requirements(requirements)

exec(open('datacompy/_version.py').read())

setup(
    name='datacompy',
    version=__version__,
    description='Dataframe comparison in Python',
    url='https://github.com/capitalone/datacompy',
    license='Apache-2.0',
    packages=find_packages(),
    install_requires=requirements,
    package_data={'': ['templates/*']},
    zip_safe=False)
