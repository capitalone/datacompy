import os

from setuptools import find_packages, setup

CURR_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURR_DIR, "README.rst"), encoding="utf-8") as file_open:
    LONG_DESCRIPTION = file_open.read()

with open("requirements.txt", "r") as requirements_file:
    raw_requirements = requirements_file.read().strip().split("\n")

INSTALL_REQUIRES = [
    line for line in raw_requirements if not (line.startswith("#") or line == "")
]


exec(open("datacompy/_version.py").read())


# No versioning on extras for dev, always grab the latest
EXTRAS_REQUIRE = {
    "spark": ["pyspark>=2.2.0"],
    "docs": ["sphinx", "sphinx_rtd_theme"],
    "tests": [
        "pytest",
        "pytest-cov",
    ],
    "qa": [
        "pre-commit",
        "black",
        "isort",
    ],
    "build": ["twine", "wheel"],
}

EXTRAS_REQUIRE["dev"] = (
    EXTRAS_REQUIRE["tests"]
    + EXTRAS_REQUIRE["docs"]
    + EXTRAS_REQUIRE["qa"]
    + EXTRAS_REQUIRE["build"]
)


setup(
    name="datacompy",
    version=__version__,
    description="Dataframe comparison in Python",
    long_description=LONG_DESCRIPTION,
    url="https://github.com/capitalone/datacompy",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    package_data={"": ["templates/*"]},
    zip_safe=False,
)
