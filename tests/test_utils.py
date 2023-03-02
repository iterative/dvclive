import pytest

from dvclive.utils import standardize_metric_name


@pytest.mark.parametrize(
    "framework,input,output",
    [
        ("dvclive.lightning", "epoch", "epoch"),
        ("dvclive.lightning", "train_loss", "train/loss"),
        ("dvclive.lightning", "train_loss_epoch", "train/epoch/loss"),
        ("dvclive.lightning", "train_model_error", "train/model_error"),
    ],
)
def test_standardize_metric_name(framework, input, output):
    assert standardize_metric_name(input, framework) == output
