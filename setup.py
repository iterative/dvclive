from setuptools import find_packages, setup

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
install_requires = ["dvc", "funcy"]
setup(
    name="dvclive",
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
