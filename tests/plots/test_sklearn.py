import json

import pytest
from sklearn import calibration, metrics

from dvclive import Live
from dvclive.plots.sklearn import SKLearnPlot

# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture
def y_true_y_pred_y_score():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_classification(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    return y_test, y_pred, y_score


def test_log_calibration_curve(tmp_dir, y_true_y_pred_y_score, mocker):
    live = Live()
    out = tmp_dir / live.plots_dir / SKLearnPlot.subfolder

    y_true, _, y_score = y_true_y_pred_y_score

    spy = mocker.spy(calibration, "calibration_curve")

    live.log_sklearn_plot("calibration", y_true, y_score)

    spy.assert_called_once_with(y_true, y_score)

    assert (out / "calibration.json").exists()


def test_log_det_curve(tmp_dir, y_true_y_pred_y_score, mocker):
    live = Live()
    out = tmp_dir / live.plots_dir / SKLearnPlot.subfolder

    y_true, _, y_score = y_true_y_pred_y_score

    spy = mocker.spy(metrics, "det_curve")

    live.log_sklearn_plot("det", y_true, y_score)

    spy.assert_called_once_with(y_true, y_score)
    assert (out / "det.json").exists()


def test_log_roc_curve(tmp_dir, y_true_y_pred_y_score, mocker):
    live = Live()
    out = tmp_dir / live.plots_dir / SKLearnPlot.subfolder

    y_true, _, y_score = y_true_y_pred_y_score

    spy = mocker.spy(metrics, "roc_curve")

    live.log_sklearn_plot("roc", y_true, y_score)

    spy.assert_called_once_with(y_true, y_score)
    assert (out / "roc.json").exists()


def test_log_prc_curve(tmp_dir, y_true_y_pred_y_score, mocker):
    live = Live()
    out = tmp_dir / live.plots_dir / SKLearnPlot.subfolder

    y_true, _, y_score = y_true_y_pred_y_score

    spy = mocker.spy(metrics, "precision_recall_curve")

    live.log_sklearn_plot("precision_recall", y_true, y_score)

    spy.assert_called_once_with(y_true, y_score)
    assert (out / "precision_recall.json").exists()


def test_log_confusion_matrix(tmp_dir, y_true_y_pred_y_score, mocker):
    live = Live()
    out = tmp_dir / live.plots_dir / SKLearnPlot.subfolder

    y_true, y_pred, _ = y_true_y_pred_y_score

    live.log_sklearn_plot("confusion_matrix", y_true, y_pred)

    cm = json.loads((out / "confusion_matrix.json").read_text())

    assert isinstance(cm, list)
    assert isinstance(cm[0], dict)
    assert cm[0]["actual"] == str(y_true[0])
    assert cm[0]["predicted"] == str(y_pred[0])


def test_dump_kwargs(tmp_dir, y_true_y_pred_y_score, mocker):
    live = Live()

    y_true, _, y_score = y_true_y_pred_y_score

    spy = mocker.spy(metrics, "roc_curve")

    live.log_sklearn_plot("roc", y_true, y_score, drop_intermediate=True)

    spy.assert_called_once_with(y_true, y_score, drop_intermediate=True)


def test_override_on_step(tmp_dir):
    live = Live()

    live.log_sklearn_plot("confusion_matrix", [0, 0], [0, 0])
    live.next_step()
    live.log_sklearn_plot("confusion_matrix", [0, 0], [1, 1])

    plot_path = tmp_dir / live.plots_dir / SKLearnPlot.subfolder
    plot_path = plot_path / "confusion_matrix.json"

    assert json.loads(plot_path.read_text()) == [
        {"actual": "0", "predicted": "1"},
        {"actual": "0", "predicted": "1"},
    ]


def test_cleanup(tmp_dir, y_true_y_pred_y_score):
    live = Live()
    out = tmp_dir / live.plots_dir / SKLearnPlot.subfolder

    y_true, y_pred, _ = y_true_y_pred_y_score

    live.log_sklearn_plot("confusion_matrix", y_true, y_pred)

    assert (out / "confusion_matrix.json").exists()

    Live()

    assert not (tmp_dir / live.plots_dir / SKLearnPlot.subfolder).exists()


def test_custom_name(tmp_dir, y_true_y_pred_y_score):
    live = Live()
    out = tmp_dir / live.plots_dir / SKLearnPlot.subfolder

    y_true, y_pred, _ = y_true_y_pred_y_score

    live.log_sklearn_plot("confusion_matrix", y_true, y_pred, name="train/cm")
    live.log_sklearn_plot("confusion_matrix", y_true, y_pred, name="val/cm")
    # ".json" should be stripped from the name
    live.log_sklearn_plot("confusion_matrix", y_true, y_pred, name="cm.json")

    assert (out / "train" / "cm.json").exists()
    assert (out / "val" / "cm.json").exists()
    assert (out / "cm.json").exists()
