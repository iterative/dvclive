DVCLive
=======

• `Docs <https://dvc.org/doc/dvclive>`_

|CI| |Coverage| |Donate|

|PyPI|

**DVCLive** is an **open-source** library for monitoring the progress of metrics during training of machine learning models. It's built with Git and MLOps principles in mind:

1. **Codification of data**. Tracked metrics are stored in readable text files that can be versioned by Git or other version control tools.
2. **Distributed**. No services or servers are required. Metrics are stored in a Git repository as text files, or pointers to files in `DVC <https://dvc.org>`_ storage.
3. **GitOps API**. Plots are generated through `DVC <https://dvc.org>`_ using Git commit SHAs or branch names, e.g.: :code:`dvc plots diff --target logs master`.

.. image:: https://dvc.org/static/cdc4ec4dabed1d7de6b8606667ebfc83/9da93/dvclive-diff-html.png

4. **Automation**. DVCLive metrics are easy to use by any automation, DevOps, or MLOps tool such as CI/CD (including `CML <https://cml.dev>`_), custom scripts, or ML platforms.

**DVCLive** integrates seamlessly with `DVC <https://dvc.org>`_; the logs/summaries it produces can be fed as :code:`dvc plots`/:code:`dvc metrics`. 

However, `DVC <https://dvc.org>`_ is *not required* to work with dvclive logs/summaries, and since they're saved as easily parsable :code:`.tsv`/:code:`.json` files, you can use your preferred visualization method.

.. contents:: **Contents**
  :backlinks: none

Quick Start
===========

Please read the `Usage guide <https://dvc.org/doc/dvclive/usage>`_ for a detailed version.

**DVCLive** is a Python library. The interface consists of three main methods:

1. `dvclive.init() <https://dvc.org/doc/dvclive/api-reference/init>`_
2. `dvclive.log() <https://dvc.org/doc/dvclive/api-reference/log>`_ 
3. `dvclive.next_step() <https://dvc.org/doc/dvclive/api-reference/next_step>`_

If you are ussing a ML training framework, check the existing integrations_.

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

.. _integrations:

Call to collaboration
=====================

Today only Python is supported (while DVC is language agnostic), with a minimum number of integrations with *ML frameworks*:

- `MMCV <https://github.com/iterative/dvclive/blob/master/dvclive/mmcv.py>`_
- `Tensorflow/Keras <https://github.com/iterative/dvclive/blob/master/dvclive/keras.py>`_
- `XGBoost <https://github.com/iterative/dvclive/blob/master/dvclive/xgb.py>`_ 

The DVCLive team is happy to extend the functionality as needed. Please `create an issue <https://github.com/iterative/dvclive/issues>`_ to start a discussion!

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
