dvclive
=======

dvclive is an open-source library for monitoring machine learning model performance.

dvclive aims to provide the user with simple python interface what will allow the

user to log the model metrics as the training progresses.

The interface consists of three main methods:

1. :code:`dvclive.init(path)` - initializes dvclive logger. The metrics will be saved under :code:`path`.

2. :code:`dvclive.log(metric, value, step)` - logs the metric value. The value and step will be appended to :code:`path/{metric}.tsv` file. The step value is optional.

3. :code:`dvclive.next_step()` - signals :code:`dvclive` that current step has ended. Executed automatically if same :code:`metric` is logged again.
