# Contributing to DVCLive

We welcome contributions to [DVCLive](https://github.com/iterative/dvclive) by the
community.

## How to report a problem

Please search [issue tracker](https://github.com/iterative/dvclive/issues) before
creating a new issue (problem or an improvement request). Feel free to add
issues related to the project.

If you feel that you can fix or implement it yourself, please read a few
paragraphs below to learn how to submit your changes.

## Submitting changes

- Open a new issue in the
  [issue tracker](https://github.com/iterative/dvclive/issues).
- Setup the [development environment](#development-environment) if you need to
  run tests or [run](#running-development-version) DVCLive with your changes.
- Fork [DVCLive](https://github.com/iterative/dvclive.git) and prepare necessary
  changes.
- [Add tests](#writing-tests) for your changes to `tests/`. You can skip this
  step if the effort to create tests for your change is unreasonable. Changes
  without tests are still going to be considered by us.
- [Run tests](#running-tests) and make sure all of them pass.
- Submit a pull request, referencing any issues it addresses.

We will review your pull request as soon as possible. Thank you for
contributing!

## Development environment

Get the latest development version. Fork and clone the repo:

```bash
git clone git@github.com:<your-username>/dvclive.git
```

Make sure that you have Python 3 installed. Version 3.6 or higher is required to
run style checkers on pre-commit. On macOS, we recommend using `brew` to install
Python. For Windows, we recommend an official
[python.org release](https://www.python.org/downloads/windows/).

Install DVCLive in editable mode with `pip install -e ".[tests]"`. But before we
do that, we **strongly** recommend creating a
[virtual environment](https://python.readthedocs.io/en/stable/library/venv.html):

```bash
$ cd dvclive
$ python3 -m venv .env
$ source .env/bin/activate
$ pip install -e ".[tests]"
```

Install coding style pre-commit hooks with:

```bash
$ pip install pre-commit
$ pre-commit install
```

That's it. You should be ready to make changes, run tests, and make commits! If
you experience any problems, please don't hesitate to ping us.


## Writing tests

We have unit tests in `tests/`. To get started, you can search for existing ones 
testing similar functionality to your changes.

## Running tests

The simplest way to run tests:

```bash
cd dvclive
pytest -vv tests
```

This uses `pytest` to run the full test suite and report the result. 
