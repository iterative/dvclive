DVCLive
-------

|PyPI| |Status| |Python Version| |License|

|Tests| |Codecov| |pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/dvclive.svg
   :target: https://pypi.org/project/dvclive/
   :alt: PyPI
.. |Status| image:: https://img.shields.io/pypi/status/dvclive.svg
   :target: https://pypi.org/project/dvclive/
   :alt: Status
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/dvclive
   :target: https://pypi.org/project/dvclive
   :alt: Python Version
.. |License| image:: https://img.shields.io/pypi/l/dvclive
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License
.. |Tests| image:: https://github.com/iterative/dvclive/workflows/Tests/badge.svg
   :target: https://github.com/iterative/dvclive/actions?workflow=Tests
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/iterative/dvclive/branch/main/graph/badge.svg
   :target: https://app.codecov.io/gh/iterative/dvclive
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black

DVCLive is a Python library for logging machine learning metrics and other metadata in simple file formats, which is fully compatible with DVC.

`Documentation <https://dvc.org/doc/dvclive>`_
----------------------------------------------

- `Get Started <https://dvc.org/doc/dvclive/get-started>`_
- `How it Works <https://dvc.org/doc/dvclive/how-it-works>`_
- `API Reference <https://dvc.org/doc/dvclive/api-reference>`_

Installation
------------

You can install *dvclive* via pip_ from PyPI_:

.. code:: console

   $ pip install dvclive

Depending on the *ML framework* you plan to use to train your model, you might need to specify
one of the optional dependencies: ``mmcv``, ``tf``, ``xgb``. Or ``all`` to include them all.
For example, for TensorFlow the command should look like this:

.. code-block:: bash

    pip install dvclive[tf]

TensorFlow and its dependencies will be installed automatically.


To install the development version, run:

.. code-block:: bash

   pip install git+https://github.com/iterative/dvclive

Comparison to related technologies
----------------------------------

**DVCLive** is an *ML Logger*, similar to:

- `MLFlow <https://mlflow.org/>`_
- `Weights & Biases <https://wandb.ai/site>`_
- `Neptune <https://neptune.ai/>`_

The main difference with those *ML Loggers* is that **DVCLive** does not require any additional services or servers to run.

Logged metrics and metadata are stored as plain text files that can be versioned by version control tools (i.e Git) or tracked as pointers to files in DVC storage.

-----


Contributing
------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


License
-------

Distributed under the terms of the `Apache 2.0 license`_,
*dvclive* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


.. _Apache 2.0 license: https://opensource.org/licenses/Apache-2.0
.. _PyPI: https://pypi.org/
.. _file an issue: https://github.com/iterative/dvclive/issues
.. _pip: https://pip.pypa.io/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
