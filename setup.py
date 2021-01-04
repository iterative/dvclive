from setuptools import find_packages, setup

packages = []
try:
    # tf can be tensorflow-gpu
    import tensorflow  # noqa # pylint: disable=unused-import
except ImportError:
    packages.append("tensorflow-cpu")


tests_requires = [
    "pylint==2.5.3",
    "pytest>=6.0.1",
    "pre-commit",
    "pylint",
    "pylint-plugin-utils",
    "black",
    "flake8",
    "pytest-cov",
]
install_requires = ["dvc", "funcy", *packages]
setup(
    name="dvclive",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"tests": tests_requires},
)
