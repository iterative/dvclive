DVCLive
=======

DVCLive is an open-source library for monitoring machine learning model performance.
It is an ML loggers similar to MLFlow, wandb, Neptune, Tenserboard and other but it was build on top of `DVC <https://dvc.org>`_ and with Git and MLOps princioples in mind:

1. **Codification of data**. Tracked metrics are stored in readable text files that can be versioned by Git or other version control tools.
2. **Distributed**. No services or servers are required. Metrics are stored in Git repository as text files or pointers to files in a storage (through DVC).
3. **GitOps API**. Graphs are generated through Git using Git sha or branches as source: `dvc plots diff --target logs master`.

.. image:: https://dvc.org/static/cdc4ec4dabed1d7de6b8606667ebfc83/9da93/dvclive-diff-html.png

4. **Automation**. DVCLive metrics are easy to use by any automation, DevOps, MLOps tools such as CI/CD (including `CML <https://cml.dev>`_), custome scripts or ML platforms.


Python API
==========

DVCLive is a Python library. The interface consists of three main methods:

1. :code:`dvclive.init(path)` - initializes dvclive logger. The metrics will be saved under :code:`path`.

2. :code:`dvclive.log(metric, value, step)` - logs the metric value. The value and step will be appended to :code:`path/{metric}.tsv` file. The step value is optional.

3. :code:`dvclive.next_step()` - signals :code:`dvclive` that current step has ended. Executed automatically if same :code:`metric` is logged again.


Call to collaboration
=====================

Today only Python language is supported (while DVC is language agnostic) with a minimum number of connectors to libs (Keras, XGBoost).
DVC team is happy to help in extending the functionality - please `create an issue <https://github.com/iterative/dvclive/issues>`_ to start a discussion.
