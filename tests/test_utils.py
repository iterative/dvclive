import pytest

from dvclive.utils import standardize_metric_name


@pytest.mark.parametrize(
    ("framework", "logged", "standardized"),
    [
        ("dvclive.lightning", "epoch", "epoch"),
        ("dvclive.lightning", "train_loss", "train/loss"),
        ("dvclive.lightning", "train_loss_epoch", "train/epoch/loss"),
        ("dvclive.lightning", "train_model_error", "train/model_error"),
        ("dvclive.lightning", "grad_step", "grad_step"),
    ],
)
def test_standardize_metric_name(framework, logged, standardized):
    assert standardize_metric_name(logged, framework) == standardized
