Contributor Guide
=================

Thank you for your interest in improving this project.
This project is open-source under the `Apache 2.0 license`_ and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- `Source Code`_
- `Issue Tracker`_
- `Code of Conduct`_

.. _Apache 2.0 license: https://opensource.org/licenses/Apache-2.0
.. _Source Code: https://github.com/iterative/dvclive
.. _Issue Tracker: https://github.com/iterative/dvclive/issues

How to report a bug
-------------------

Report bugs on the `Issue Tracker`_.

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.


How to request a feature
------------------------

Request features on the `Issue Tracker`_.


How to set up your development environment
------------------------------------------

You need Python 3.9+.

- Clone the repository:

.. code:: console

   $ git clone https://github.com/iterative/dvclive
   $ cd dvclive

- Set up a virtual environment:

.. code:: console

   $ python -m venv .venv
   $ source .venv/bin/activate

Install in editable mode including development dependencies:

.. code:: console

   $ pip install -e .[tests]

If you need to test against a specific framework, you can install it separately:

.. code:: console

   $ pip install -e .[tests,tf]
   $ pip install -e .[tests,optuna]

How to test the project
-----------------------

Run the full test suite:

.. code:: console

   $ pytest -v tests

Tests are located in the ``tests`` directory,
and are written using the pytest_ testing framework.

.. _pytest: https://pytest.readthedocs.io/


How to submit changes
---------------------

Open a `pull request`_ to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The test suite must pass without errors and warnings.
- Include unit tests.
- If your changes add functionality, update the documentation accordingly.

Feel free to submit early, though—we can always iterate on this.

To run linting and code formatting checks, you can use `pre-commit`:

.. code:: console

   $ pre-commit run --all-files

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

.. _pull request: https://github.com/iterative/dvclive/pulls
.. github-only
.. _Code of Conduct: CODE_OF_CONDUCT.rst
