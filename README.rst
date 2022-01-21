DVCLive
=======


|CI| |Coverage| |Donate|

|PyPI|

DVCLive is a Python library for logging machine learning metrics and other metadata in simple file formats, which is fully compatible with DVC.

`Documentation <https://dvc.org/doc/dvclive>`_
==============================================

- `Get Started <https://dvc.org/doc/dvclive/get-started>`_
- `DVCLive with DVC <https://dvc.org/doc/dvclive/dvclive-with-dvc>`_
- `ML Frameworks <https://dvc.org/doc/dvclive/ml-frameworks>`_
- `API Reference <https://dvc.org/doc/dvclive/api-reference>`_

Installation
============

pip (PyPI)
----------

|PyPI|

.. code-block:: bash

   pip install dvclive

Depending on the *ML framework* you plan to use to train your model, you might need to specify
one of the optional dependencies: ``mmcv``, ``tf``, ``xgb``. Or ``all`` to include them all.
The command should look like this: ``pip install dvclive[tf]`` (in this case TensorFlow and it's dependencies
will be installed automatically).

To install the development version, run:

.. code-block:: bash

   pip install git+git://github.com/iterative/dvclive

Comparison to related technologies
==================================

**DVCLive** is an *ML Logger*, similar to:

- `MLFlow <https://mlflow.org/>`_
- `Weights & Biases <https://wandb.ai/site>`_
- `Neptune <https://neptune.ai/>`_ 

The main difference with those *ML Loggers* is that **DVCLive** does not require any additional services or servers to run. 

Logged metrics and metadata are stored as plain text files that can be versioned by version control tools (i.e Git) or tracked as pointers to files in DVC storage. 

Copyright
=========

This project is distributed under the Apache license version 2.0 (see the LICENSE file in the project root).

By submitting a pull request to this project, you agree to license your contribution under the Apache license version
2.0 to this project.

.. |CI| image:: https://github.com/iterative/dvclive/workflows/tests/badge.svg
   :target: https://github.com/iterative/dvclive/actions
   :alt: GHA Tests

.. |Coverage| image:: https://codecov.io/gh/iterative/dvclive/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/iterative/dvclive
   :alt: Codecov

.. |Donate| image:: https://img.shields.io/badge/patreon-donate-green.svg?logo=patreon
   :target: https://www.patreon.com/DVCorg/overview
   :alt: Donate

.. |PyPI| image:: https://img.shields.io/pypi/v/dvclive.svg?label=pip&logo=PyPI&logoColor=white
   :target: https://pypi.org/project/dvclive
   :alt: PyPI
