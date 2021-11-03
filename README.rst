DVCLive
=======

• `Docs <https://dvc.org/doc/dvclive>`_

|CI| |Coverage| |Donate|

|PyPI|

**DVCLive** is an **open-source** library for monitoring the progress of metrics during training of machine learning models. It's built with Git and MLOps principles in mind:

1. **Codification of data**. Tracked metrics are stored in readable text files that can be versioned by Git or other version control tools.
2. **Distributed**. No services or servers are required. Metrics are stored in a Git repository as text files, or pointers to files in `DVC <https://dvc.org>`_ storage.
3. **GitOps API**. Plots are generated through `DVC <https://dvc.org>`_ using Git commit SHAs or branch names, e.g.: :code:`dvc plots diff --target logs master`.

.. image:: https://raw.githubusercontent.com/iterative/dvc.org/master/static/uploads/images/2021-02-18/dvclive-diff-html.png

4. **Automation**. DVCLive metrics are easy to use by any automation, DevOps, or MLOps tool such as CI/CD (including `CML <https://cml.dev>`_), custom scripts, or ML platforms.

**DVCLive** integrates seamlessly with `DVC <https://dvc.org>`_; the logs/summaries it produces can be fed as :code:`dvc plots`/:code:`dvc metrics`. 

However, `DVC <https://dvc.org>`_ is *not required* to work with dvclive logs/summaries, and since they're saved as easily parsable :code:`.tsv`/:code:`.json` files, you can use your preferred visualization method.

.. contents:: **Contents**
  :backlinks: none

Quick Start
===========

Please read the `Get Started <https://dvc.org/doc/dvclive/get-started>`_ for a detailed version.

**DVCLive** is a Python library. The interface consists of three main steps:

1. Initialize DVCLive

.. code-block:: python

  from dvclive import Live

  live = Live()


2. Log metrics


.. code-block:: python

  live.log("metric", 1)

3. Increase the step number

.. code-block:: python

  live.next_step()
 

If you are ussing a ML training framework, check the existing `ML Frameworks <https://dvc.org/doc/dvclive/user-guide/ml-frameworks>`_ page.

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

Logged metrics are stored as plain text files that can be versioned by version control tools (i.e Git) or tracked as pointers to files in DVC storage. 

Call to collaboration
=====================

Today only Python is supported (while DVC is language agnostic), along with the following *ML frameworks*:

- `Catalyst <https://dvc.org/doc/dvclive/user-guide/ml-frameworks/catalyst>`_
- `Fast.ai <https://dvc.org/doc/dvclive/user-guide/ml-frameworks/fastai>`_
- `Hugging Face <https://dvc.org/doc/dvclive/user-guide/ml-frameworks/huggingface>`_
- `Keras <https://dvc.org/doc/dvclive/user-guide/ml-frameworks/keras>`_
- `LightGBM <https://dvc.org/doc/dvclive/user-guide/ml-frameworks/lightgbm>`_
- `MMCV <https://dvc.org/doc/dvclive/user-guide/ml-frameworks/mmcv>`_
- `PyTorch <https://dvc.org/doc/dvclive/user-guide/ml-frameworks/pytorch>`_
- `PyTorch Lightning <https://dvc.org/doc/dvclive/user-guide/ml-frameworks/pytorch-lightning>`_
- `Tensorflow <https://dvc.org/doc/dvclive/user-guide/ml-frameworks/tensorflow>`_
- `XGBoost <https://dvc.org/doc/dvclive/user-guide/ml-frameworks/xgboost>`_ 

The DVCLive team is happy to extend the functionality as needed. Please `create an issue <https://github.com/iterative/dvclive/issues>`_ or check the `existing ones <https://github.com/iterative/dvclive/issues?q=is%3Aissue+is%3Aopen+label%3Aintegrations>`_ to start a discussion!

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
