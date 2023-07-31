import shutil
from pathlib import Path

import pytest

from dvclive import Live
from dvclive.serialize import load_yaml

dvcyaml = """
stages:
  train:
    cmd: python train.py
    outs:
    - data
"""


@pytest.mark.parametrize("cache", [True, False])
def test_log_artifact(tmp_dir, dvc_repo, cache):
    data = tmp_dir / "data"
    data.touch()
    with Live() as live:
        live.log_artifact("data", cache=cache)
    assert data.with_suffix(".dvc").exists() is cache
    assert load_yaml(live.dvc_file) == {}


def test_log_artifact_on_existing_dvc_file(tmp_dir, dvc_repo):
    data = tmp_dir / "data"
    data.write_text("foo")
    with Live() as live:
        live.log_artifact("data")

    prev_content = data.with_suffix(".dvc").read_text()

    with Live() as live:
        data.write_text("bar")
        live.log_artifact("data")

    assert data.with_suffix(".dvc").read_text() != prev_content


def test_log_artifact_twice(tmp_dir, dvc_repo):
    data = tmp_dir / "data"
    with Live() as live:
        for i in range(2):
            data.write_text(str(i))
            live.log_artifact("data")
    assert data.with_suffix(".dvc").exists()


def test_log_artifact_with_save_dvc_exp(tmp_dir, mocker, mocked_dvc_repo):
    stage = mocker.MagicMock()
    stage.addressing = "data"
    mocked_dvc_repo.add.return_value = [stage]
    with Live(save_dvc_exp=True) as live:
        live.log_artifact("data")
    mocked_dvc_repo.experiments.save.assert_called_with(
        name=live._exp_name,
        include_untracked=[live.dir, "data", ".gitignore"],
        force=True,
        message=None,
    )


def test_log_artifact_type_model(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live() as live:
        live.log_artifact("model.pth", type="model")

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"model": {"path": "../model.pth", "type": "model"}}
    }


def test_log_artifact_dvc_symlink(tmp_dir, dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live() as live:
        live._dvc_repo.cache.local.cache_types = ["symlink"]
        live.log_artifact("model.pth", type="model")

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"model": {"path": "../model.pth", "type": "model"}}
    }


def test_log_artifact_copy(tmp_dir, dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live() as live:
        live.log_artifact("model.pth", type="model", copy=True)

    artifacts_dir = Path(live.artifacts_dir)
    assert (artifacts_dir / "model.pth").exists()
    assert (artifacts_dir / "model.pth.dvc").exists()

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"model": {"path": "artifacts/model.pth", "type": "model"}}
    }


def test_log_artifact_copy_overwrite(tmp_dir, dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live() as live:
        artifacts_dir = Path(live.artifacts_dir)
        # testing with symlink cache to make sure that DVC protected mode
        # does not prevent the overwrite
        live._dvc_repo.cache.local.cache_types = ["symlink"]
        live.log_artifact("model.pth", type="model", copy=True)
        assert (artifacts_dir / "model.pth").is_symlink()
        live.log_artifact("model.pth", type="model", copy=True)

    assert (artifacts_dir / "model.pth").exists()
    assert (artifacts_dir / "model.pth.dvc").exists()

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"model": {"path": "artifacts/model.pth", "type": "model"}}
    }


def test_log_artifact_copy_directory_overwrite(tmp_dir, dvc_repo):
    model_path = Path(tmp_dir / "weights")
    model_path.mkdir()
    (tmp_dir / "weights" / "model-epoch-1.pth").touch()

    with Live() as live:
        artifacts_dir = Path(live.artifacts_dir)
        # testing with symlink cache to make sure that DVC protected mode
        # does not prevent the overwrite
        live._dvc_repo.cache.local.cache_types = ["symlink"]
        live.log_artifact(model_path, type="model", copy=True)
        assert (artifacts_dir / "weights" / "model-epoch-1.pth").is_symlink()

        shutil.rmtree(model_path)
        model_path.mkdir()
        (tmp_dir / "weights" / "model-epoch-10.pth").write_text("Model weights")
        (tmp_dir / "weights" / "best.pth").write_text("Best model weights")
        live.log_artifact(model_path, type="model", copy=True)

    assert (artifacts_dir / "weights").exists()
    assert (artifacts_dir / "weights" / "best.pth").is_symlink()
    assert (artifacts_dir / "weights" / "best.pth").read_text() == "Best model weights"
    assert (artifacts_dir / "weights" / "model-epoch-10.pth").is_symlink()
    assert len(list((artifacts_dir / "weights").iterdir())) == 2

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"weights": {"path": "artifacts/weights", "type": "model"}}
    }


def test_log_artifact_type_model_provided_name(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live() as live:
        live.log_artifact("model.pth", type="model", name="custom")

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"custom": {"path": "../model.pth", "type": "model"}}
    }


def test_log_artifact_type_model_on_step_and_final(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live() as live:
        for _ in range(3):
            live.log_artifact("model.pth", type="model")
            live.next_step()
        live.log_artifact("model.pth", type="model", labels=["final"])
    assert load_yaml(live.dvc_file) == {
        "artifacts": {
            "model": {"path": "../model.pth", "type": "model", "labels": ["final"]},
        },
        "metrics": ["metrics.json"],
    }


def test_log_artifact_type_model_on_step(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live() as live:
        for _ in range(3):
            live.log_artifact("model.pth", type="model")
            live.next_step()

    assert load_yaml(live.dvc_file) == {
        "artifacts": {
            "model": {"path": "../model.pth", "type": "model"},
        },
        "metrics": ["metrics.json"],
    }


def test_log_artifact_attrs(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    attrs = {
        "type": "model",
        "name": "foo",
        "desc": "bar",
        "labels": ["foo"],
        "meta": {"foo": "bar"},
    }
    with Live() as live:
        live.log_artifact("model.pth", **attrs)
    attrs.pop("name")
    assert load_yaml(live.dvc_file) == {
        "artifacts": {
            "foo": {"path": "../model.pth", **attrs},
        }
    }


def test_log_artifact_type_model_when_dvc_add_fails(tmp_dir, mocker, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()
    mocked_dvc_repo.add.side_effect = Exception
    with Live(save_dvc_exp=True) as live:
        live.log_artifact("model.pth", type="model")

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"model": {"path": "../model.pth", "type": "model"}}
    }


def test_log_artifact_inside_exp(tmp_dir, mocked_dvc_repo):
    data = tmp_dir / "data"
    data.touch()
    with Live() as live:
        live._inside_dvc_exp = True
        live.log_artifact("data")
    mocked_dvc_repo.add.assert_not_called()


@pytest.mark.parametrize("tracked", ["data_source", "stage", None])
def test_log_artifact_inside_exp_logger(tmp_dir, mocker, dvc_repo, tracked):
    logger = mocker.patch("dvclive.live.logger")
    if tracked == "data_source":
        data = tmp_dir / "data"
        data.touch()
        dvc_repo.add(data)
    elif tracked == "stage":
        dvcyaml_path = tmp_dir / "dvc.yaml"
        with open(dvcyaml_path, "w") as f:
            f.write(dvcyaml)
    with Live() as live:
        live._inside_dvc_exp = True
        live.log_artifact("data")
    msg = "Skipping dvc add data because `dvc exp run` is running."
    if tracked == "data_source":
        msg += (
            "\nTo track it automatically during `dvc exp run`:"
            "\n1. Run `dvc exp remove data.dvc` "
            "to stop tracking it outside the pipeline."
            "\n2. Add it as an output of the pipeline stage."
        )
        logger.warning.assert_called_with(msg)
    elif tracked == "stage":
        msg += "\nIt is already being tracked automatically."
        logger.info.assert_called_with(msg)
    else:
        msg += (
            "\nTo track it automatically during `dvc exp run`, "
            "add it as an output of the pipeline stage."
        )
        logger.warning.assert_called_with(msg)
