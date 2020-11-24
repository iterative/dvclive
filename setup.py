from setuptools import find_packages, setup

tests_requires = ["pylint==2.5.3", "pytest>=6.0.1"]
install_requires = ["keras", "dvc"]
setup(
    name="dvclive",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"tests": tests_requires},
)
