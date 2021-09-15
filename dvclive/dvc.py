import os

from . import env

SIGNAL_FILE = "DVC_LIVE"
_CHECKPOINT_SLEEP = 0.1


def _dvc_dir(dirname):
    return os.path.join(dirname, ".dvc")


def _find_dvc_root(root=None):
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


def get_signal_file_path(root=None):
    root = _find_dvc_root(root)

    if not root:
        return None

    tmp = os.path.join(_dvc_dir(root), "tmp")

    if not os.path.exists(tmp):
        os.makedirs(tmp)

    return os.path.join(tmp, SIGNAL_FILE)


def make_html():
    signal_file_path = get_signal_file_path()
    if signal_file_path:
        if not os.path.exists(signal_file_path):
            with open(signal_file_path, "w"):
                pass


def make_checkpoint():
    import builtins
    from time import sleep

    if os.getenv(env.DVC_CHECKPOINT) is None:
        return

    root_dir = _find_dvc_root()
    if not root_dir:
        return

    signal_file = os.path.join(root_dir, ".dvc", "tmp", env.DVC_CHECKPOINT)

    with builtins.open(signal_file, "w") as fobj:
        # NOTE: force flushing/writing empty file to disk, otherwise when
        # run in certain contexts (pytest) file may not actually be written
        fobj.write("")
        fobj.flush()
        os.fsync(fobj.fileno())
    while os.path.exists(signal_file):
        sleep(_CHECKPOINT_SLEEP)
