from setuptools import find_packages, setup

tests_requires = ["pylint==2.5.3", "pytest>=6.0.1"]
setup(
    name="dvclive",
    packages=find_packages(),
    extras_require={"tests": tests_requires},
)
