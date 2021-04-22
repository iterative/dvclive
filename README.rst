DVCLive
=======

DVCLive is an open-source library for monitoring machine learning model performance. It's an ML logger similar to MLFlow, Weights & Biases, Neptune, Tensorboard, etc., but built on top of `DVC <https://dvc.org>`_, and with Git and MLOps principles in mind:

1. **Codification of data**. Tracked metrics are stored in readable text files that can be versioned by Git or other version control tools.
2. **Distributed**. No services or servers are required. Metrics are stored in a Git repository as text files, or pointers to files in DVC storage.
3. **GitOps API**. Plots are generated through DVC using Git commit SHAs or branch names, e.g.: :code:`dvc plots diff --target logs master`.

.. image:: https://dvc.org/static/cdc4ec4dabed1d7de6b8606667ebfc83/9da93/dvclive-diff-html.png

4. **Automation**. DVCLive metrics are easy to use by any automation, DevOps, or MLOps tool such as CI/CD (including `CML <https://cml.dev>`_), custom scripts, or ML platforms.


Python API
==========

DVCLive is a Python library. The interface consists of three main methods:

1. :code:`dvclive.init(path)` - initializes a DVCLive logger. The metrics will be saved under :code:`path`.
2. :code:`dvclive.log(metric, value, step)` - logs the metric value. The :code:`value` and :code:`step` (optional) will be appended to :code:`path/{metric}.tsv` file.
3. :code:`dvclive.next_step()` - signals that the current step has ended (implied when the same :code:`metric` is logged again).


Call to collaboration
=====================

Today only Python is supported (while DVC is language agnostic), with a minimum number of connectors to ML libs (Keras, XGBoost).
The DVCLive team is happy to extend the functionality as needed. Please `create an issue <https://github.com/iterative/dvclive/issues>`_ to start a discussion!
