import json
import os

import pytest

from dvclive import Live, env


@pytest.mark.vscode
@pytest.mark.parametrize("dvc_root", [True, False])
def test_vscode_dvclive_step_completed_signal_file(
    tmp_dir, dvc_root, mocker, monkeypatch
):
    signal_file = os.path.join(
        tmp_dir, ".dvc", "tmp", "exps", "run", "DVCLIVE_STEP_COMPLETED"
    )
    cwd = tmp_dir
    test_pid = 12345

    if dvc_root:
        cwd = tmp_dir / ".dvc" / "tmp" / "exps" / "asdasasf"
        monkeypatch.setenv(env.DVC_ROOT, tmp_dir.as_posix())
        (cwd / ".dvc").mkdir(parents=True)

    assert not os.path.exists(signal_file)

    dvc_repo = mocker.MagicMock()
    dvc_repo.index.stages = []
    dvc_repo.config = {}
    dvc_repo.scm.get_rev.return_value = "current_rev"
    dvc_repo.scm.get_ref.return_value = None
    dvc_repo.scm.no_commits = False
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
    mocker.patch("dvclive.live.os.getpid", return_value=test_pid)

    dvclive = Live(save_dvc_exp=True)
    assert not os.path.exists(signal_file)
    dvclive.next_step()
    assert dvclive.step == 1

    if dvc_root:
        assert os.path.exists(signal_file)
        with open(signal_file, encoding="utf-8") as f:
            assert json.load(f) == {"pid": test_pid, "step": 0}

    else:
        assert not os.path.exists(signal_file)

    dvclive.next_step()
    assert dvclive.step == 2

    if dvc_root:
        with open(signal_file, encoding="utf-8") as f:
            assert json.load(f) == {"pid": test_pid, "step": 1}

    dvclive.end()

    assert not os.path.exists(signal_file)


@pytest.mark.vscode
@pytest.mark.parametrize("dvc_root", [True, False])
def test_vscode_dvclive_only_signal_file(tmp_dir, dvc_root, mocker):
    signal_file = os.path.join(tmp_dir, ".dvc", "tmp", "exps", "run", "DVCLIVE_ONLY")
    test_pid = 12345

    if dvc_root:
        (tmp_dir / ".dvc").mkdir(parents=True)

    assert not os.path.exists(signal_file)

    dvc_repo = mocker.MagicMock()
    dvc_repo.index.stages = []
    dvc_repo.config = {}
    dvc_repo.scm.get_rev.return_value = "current_rev"
    dvc_repo.scm.get_ref.return_value = None
    dvc_repo.scm.no_commits = False
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
    mocker.patch("dvclive.live.os.getpid", return_value=test_pid)

    dvclive = Live(save_dvc_exp=True)

    if dvc_root:
        assert os.path.exists(signal_file)
        with open(signal_file, encoding="utf-8") as f:
            assert json.load(f) == {"pid": test_pid, "exp_name": dvclive._exp_name}

    else:
        assert not os.path.exists(signal_file)

    dvclive.end()

    assert not os.path.exists(signal_file)
