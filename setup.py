from setuptools import find_packages, setup

render = ["dvc_render[table]>=0.0.8"]
image = ["pillow"]
plots = ["scikit-learn"]
mmcv = ["mmcv"]
tf = ["tensorflow"]
xgb = ["xgboost"]
lgbm = ["lightgbm"]
hugginface = ["transformers", "datasets"]
catalyst = ["catalyst<=21.12"]
fastai = ["fastai"]
pl = ["pytorch_lightning>=1.6"]

all_libs = (
    render
    + image
    + mmcv
    + tf
    + xgb
    + lgbm
    + hugginface
    + catalyst
    + fastai
    + pl
    + plots
)

tests_requires = [
    "pylint==2.5.3",
    "pytest>=6.0.1",
    "pre-commit",
    "pylint-plugin-utils>=0.6",
    "pytest-cov>=2.12.1",
    "pytest-mock>=3.6.1",
    "pandas>=1.3.1",
    "funcy>=1.14",
    "dvc>=2.0.0",
] + all_libs

setup(
    name="dvclive",
    author="Paweł Redzyński",
    author_email="pawel@iterative.ai",
    packages=find_packages(exclude="tests"),
    description="Metric logger for ML projects.",
    long_description=open("README.rst", "r", encoding="UTF-8").read(),
    license="Apache License 2.0",
    license_files=("LICENSE",),
    install_requires=render,
    extras_require={
        "tests": tests_requires,
        "all": all_libs,
        "tf": tf,
        "xgb": xgb,
        "lgbm": lgbm,
        "mmcv": mmcv,
        "huggingface": hugginface,
        "catalyst": catalyst,
        "fastai": fastai,
        "pytorch_lightning": pl,
        "sklearn": plots,
        "image": image,
        "plots": plots,
    },
    keywords="data-science metrics machine-learning developer-tools ai",
    python_requires=">=3.8",
    url="https://dvc.org/doc/dvclive",
    download_url="https://github.com/iterative/dvclive",
)
