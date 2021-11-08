import os

import pytest

from dvclive.data.scalar import Scalar
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

    model.fit(
        x,
        y,
        epochs=1,
        batch_size=1,
        callbacks=[DvcLiveCallback()],
    )

    assert os.path.exists("dvclive")
    logs, _ = read_logs(tmp_dir / "dvclive" / Scalar.subfolder)

    assert "accuracy" in logs


@pytest.mark.parametrize("save_weights_only", (True, False))
def test_keras_model_file(
    tmp_dir, xor_model, mocker, save_weights_only, capture_wrap
):
    model, x, y = xor_model()
    save = mocker.spy(model, "save")
    save_weights = mocker.spy(model, "save_weights")

    model.fit(
        x,
        y,
        epochs=1,
        batch_size=1,
        callbacks=[
            DvcLiveCallback(
                model_file="model.h5", save_weights_only=save_weights_only
            )
        ],
    )
    assert save.call_count != save_weights_only
    assert save_weights.call_count == save_weights_only


@pytest.mark.parametrize("save_weights_only", (True, False))
def test_keras_load_model_on_resume(
    tmp_dir, xor_model, mocker, save_weights_only, capture_wrap
):
    import dvclive.keras

    model, x, y = xor_model()

    if save_weights_only:
        model.save_weights("model.h5")
    else:
        model.save("model.h5")

    load_weights = mocker.spy(model, "load_weights")
    load_model = mocker.spy(dvclive.keras, "load_model")

    model.fit(
        x,
        y,
        epochs=1,
        batch_size=1,
        callbacks=[
            DvcLiveCallback(
                model_file="model.h5",
                save_weights_only=save_weights_only,
                resume=True,
            )
        ],
    )

    assert load_model.call_count != save_weights_only
    assert load_weights.call_count == save_weights_only


def test_keras_no_resume_skip_load(tmp_dir, xor_model, mocker, capture_wrap):
    model, x, y = xor_model()

    model.save_weights("model.h5")

    load_weights = mocker.spy(model, "load_weights")

    model.fit(
        x,
        y,
        epochs=1,
        batch_size=1,
        callbacks=[
            DvcLiveCallback(
                model_file="model.h5",
                save_weights_only=True,
                resume=False,
            )
        ],
    )

    assert load_weights.call_count == 0


def test_keras_no_existing_model_file_skip_load(
    tmp_dir, xor_model, mocker, capture_wrap
):
    model, x, y = xor_model()

    load_weights = mocker.spy(model, "load_weights")

    model.fit(
        x,
        y,
        epochs=1,
        batch_size=1,
        callbacks=[
            DvcLiveCallback(
                model_file="model.h5",
                save_weights_only=True,
                resume=True,
            )
        ],
    )

    assert load_weights.call_count == 0


def test_keras_None_model_file_skip_load(
    tmp_dir, xor_model, mocker, capture_wrap
):
    model, x, y = xor_model()

    model.save_weights("model.h5")

    load_weights = mocker.spy(model, "load_weights")

    model.fit(
        x,
        y,
        epochs=1,
        batch_size=1,
        callbacks=[
            DvcLiveCallback(
                save_weights_only=True,
                resume=True,
            )
        ],
    )

    assert load_weights.call_count == 0
