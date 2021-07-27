import os

import pytest

import dvclive
from dvclive.keras import DvcLiveCallback
from tests.test_main import read_logs

# pylint: disable=unused-argument, no-name-in-module, redefined-outer-name


@pytest.fixture
def xor_model():
    import numpy as np
    from tensorflow.python.keras import Sequential
    from tensorflow.python.keras.layers import Activation, Dense

    def make():
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        model = Sequential()
        model.add(Dense(8, input_dim=2))
        model.add(Activation("relu"))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(
            loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"]
        )

        return model, x, y

    yield make


def test_keras_callback(tmp_dir, xor_model, capture_wrap):
    model, x, y = xor_model()

    dvclive.init("logs")
    model.fit(
        x, y, epochs=1, batch_size=1, callbacks=[DvcLiveCallback()],
    )

    assert os.path.exists("logs")
    logs, _ = read_logs("logs")

    assert "accuracy" in logs
