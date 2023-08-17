# ruff: noqa: SLF001
import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from dvclive.plots import Image, Metric
from dvclive.serialize import dump_yaml
from dvclive.utils import StrPath

if TYPE_CHECKING:
    from dvc.repo import Repo
    from dvc.stage import Stage


def _dvc_dir(dirname: StrPath) -> str:
    return os.path.join(dirname, ".dvc")


def _dvc_exps_run_dir(dirname: StrPath) -> str:
    return os.path.join(dirname, ".dvc", "tmp", "exps", "run")


def _dvclive_only_signal_file(root_dir: StrPath) -> str:
    dvc_exps_run_dir = _dvc_exps_run_dir(root_dir)
    return os.path.join(dvc_exps_run_dir, "DVCLIVE_ONLY")


def _find_dvc_root(root: Optional[StrPath] = None) -> Optional[str]:
    if not root:
        root = os.getcwd()

    root = os.path.realpath(root)

    if not os.path.isdir(root):
        raise NotADirectoryError(f"'{root}'")

    while True:
        if os.path.exists(_dvc_dir(root)):
            return root
        if os.path.ismount(root):
            break
        root = os.path.dirname(root)

    return None


def _write_file(file: str, contents=""):
    import builtins

    with builtins.open(file, "w", encoding="utf-8") as fobj:
        # NOTE: force flushing/writing empty file to disk, otherwise when
        # run in certain contexts (pytest) file may not actually be written
        fobj.write(str(contents))
        fobj.flush()
        os.fsync(fobj.fileno())


def get_dvc_repo() -> Optional["Repo"]:
    from dvc.exceptions import NotDvcRepoError
    from dvc.repo import Repo
    from dvc.scm import Git, SCMError
    from scmrepo.exceptions import SCMError as GitSCMError

    try:
        return Repo()
    except (NotDvcRepoError, SCMError):
        try:
            return Repo.init(Git().root_dir)
        except GitSCMError:
            return None


def make_dvcyaml(live) -> None:
    dvcyaml = {}
    if live._params:
        dvcyaml["params"] = [os.path.relpath(live.params_file, live.dir)]
    if live._metrics or live.summary:
        dvcyaml["metrics"] = [os.path.relpath(live.metrics_file, live.dir)]
    plots: List[Any] = []
    plots_path = Path(live.plots_dir)
    metrics_path = plots_path / Metric.subfolder
    if metrics_path.exists():
        metrics_relpath = metrics_path.relative_to(live.dir).as_posix()
        metrics_config = {metrics_relpath: {"x": "step"}}
        plots.append(metrics_config)
    if live._images:
        images_path = (plots_path / Image.subfolder).relative_to(live.dir)
        plots.append(images_path.as_posix())
    if live._plots:
        for plot in live._plots.values():
            plot_path = plot.output_path.relative_to(live.dir)
            plots.append({plot_path.as_posix(): plot.plot_config})
    if plots:
        dvcyaml["plots"] = plots

    if live._artifacts:
        dvcyaml["artifacts"] = copy.deepcopy(live._artifacts)
        for artifact in dvcyaml["artifacts"].values():  # type: ignore
            abs_path = os.path.abspath(artifact["path"])
            abs_dir = os.path.realpath(live.dir)
            relative_path = os.path.relpath(abs_path, abs_dir)
            artifact["path"] = Path(relative_path).as_posix()

    dump_yaml(dvcyaml, live.dvc_file)


def mark_dvclive_only_started() -> None:
    """
    Signal DVC VS Code extension that
    an experiment is running in the workspace.
    """
    root_dir = _find_dvc_root()
    if not root_dir:
        return

    exp_run_dir = _dvc_exps_run_dir(root_dir)
    os.makedirs(exp_run_dir, exist_ok=True)

    signal_file = _dvclive_only_signal_file(root_dir)

    _write_file(signal_file, os.getpid())


def mark_dvclive_only_ended() -> None:
    root_dir = _find_dvc_root()
    if not root_dir:
        return

    signal_file = _dvclive_only_signal_file(root_dir)

    if not os.path.exists(signal_file):
        return

    os.remove(signal_file)


def get_random_exp_name(scm, baseline_rev) -> str:
    from dvc.repo.experiments.utils import (
        get_random_exp_name as dvc_get_random_exp_name,
    )

    return dvc_get_random_exp_name(scm, baseline_rev)


def find_overlapping_stage(dvc_repo: "Repo", path: StrPath) -> Optional["Stage"]:
    abs_path = str(Path(path).absolute())
    for stage in dvc_repo.index.stages:
        for out in stage.outs:
            if str(out.fs_path) in abs_path:
                return stage
    return None


def ensure_dir_is_tracked(directory: str, dvc_repo: "Repo") -> None:
    from pathspec import PathSpec

    dir_spec = PathSpec.from_lines("gitwildmatch", [directory])
    outs_spec = PathSpec.from_lines(
        "gitwildmatch", [str(o) for o in dvc_repo.index.outs]
    )
    paths_to_track = [
        f
        for f in dvc_repo.scm.untracked_files()
        if (dir_spec.match_file(f) and not outs_spec.match_file(f))
    ]
    if paths_to_track:
        dvc_repo.scm.add(paths_to_track)


def get_dvc_stage_template(live):
    stage = {
        "cmd": "<python my_code_file.py my_args>",
        "deps": ["<my_code_file.py>"],
    }
    outs = []
    rel_path = Path(os.path.relpath(os.getcwd(), live._dvc_repo.root_dir))
    if live._images and live._cache_images:
        images_path = (rel_path / live.plots_dir / Image.subfolder).as_posix()
        outs.append(images_path)
    for o, cache in live._outs.items():
        artifact_path = Path(os.getcwd()) / o
        artifact_path = artifact_path.relative_to(live._dvc_repo.root_dir).as_posix()
        if cache:
            outs.append(artifact_path)
        else:
            outs.append({artifact_path: {"cache": False}})
    if outs:
        stage["outs"] = outs
    return {"stages": {"dvclive": stage}}
