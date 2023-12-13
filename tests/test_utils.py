import numpy as np
import pandas as pd
import pytest

from dvclive.utils import standardize_metric_name, convert_datapoints_to_list_of_dicts
from dvclive.error import InvalidDataTypeError


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


class TestConvertDatapointsToListOfDicts:
    def test_dataframe(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        expected_output = [{"A": 1, "B": 3}, {"A": 2, "B": 4}]
        assert convert_datapoints_to_list_of_dicts(df) == expected_output

    def test_ndarray(self):
        arr = np.array([[1, 3], [2, 4]])
        expected_output = [{0: 1, 1: 3}, {0: 2, 1: 4}]
        assert convert_datapoints_to_list_of_dicts(arr) == expected_output

    def test_structured_array(self):
        dtype = [("A", "i4"), ("B", "i4")]
        structured_array = np.array([(1, 3), (2, 4)], dtype=dtype)
        expected_output = [{"A": 1, "B": 3}, {"A": 2, "B": 4}]
        assert convert_datapoints_to_list_of_dicts(structured_array) == expected_output

    def test_list_of_dicts(self):
        list_of_dicts = [{"A": 1, "B": 3}, {"A": 2, "B": 4}]
        assert convert_datapoints_to_list_of_dicts(list_of_dicts) == list_of_dicts

    def test_unsupported_format(self):
        with pytest.raises(InvalidDataTypeError) as exc_info:
            convert_datapoints_to_list_of_dicts("unsupported data format")

        assert "not supported type" in str(exc_info.value)
