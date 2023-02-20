# pylint: disable=unused-argument,protected-access
from dvclive import Live


def test_log_artifact(tmp_dir, dvc_repo):
    data = tmp_dir / "data"
    data.touch()
    with Live() as live:
        live.log_artifact("data")
    assert data.with_suffix(".dvc").exists()


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
    )
