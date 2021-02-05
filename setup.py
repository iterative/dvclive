import importlib.util
import os

from setuptools import find_packages, setup

# Read package meta-data from version.py
# see https://packaging.python.org/guides/single-sourcing-package-version/
pkg_dir = os.path.dirname(os.path.abspath(__file__))
version_path = os.path.join(pkg_dir, "dvclive", "version.py")
spec = importlib.util.spec_from_file_location("dvclive.version", version_path)
dvc_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dvc_version)
version = dvc_version.__version__

tf = ["tensorflow"]
xgb = ["xgboost"]

all_libs = tf + xgb

tests_requires = [
    "pylint==2.5.3",
    "pytest>=6.0.1",
    "pre-commit",
    "pylint",
    "pylint-plugin-utils",
    "black",
    "flake8",
    "pytest-cov",
    "pytest-mock",
    "pandas",
    "sklearn",
] + all_libs
install_requires = ["dvc", "funcy", "ruamel.yaml"]

setup(
    name="dvclive",
    version=version,
    author="Paweł Redzyński",
    author_email="pawel@iterative.ai",
    packages=find_packages(exclude="tests"),
    description="Metric logger for ML projects.",
    long_description=open("README.md", "r").read(),
    install_requires=install_requires,
    extras_require={
        "tests": tests_requires,
        "all": all_libs,
        "tf": tf,
        "xgb": xgb,
    },
    keywords="data-science metrics machine-learning developer-tools ai",
    python_requires=">=3.6",
    url="http://dvc.org",
)
