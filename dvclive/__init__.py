import os
import shutil
import time
from collections import OrderedDict

from dvclive.error import DvcLiveError
from dvclive.io import update_tsv, write_json, write_yaml

SUFFIX_TSV = ".tsv"
SUFFIX_JSON = ".json"


class DvcLive:
    def __init__(self):
        self._dir = None
        self._step = None
        self._metrics = {}

    def init(self, directory: str, is_continue: bool = False, step: int = 0):
        self._dir = directory
        self._step = step

        if is_continue:
            if step == 0:
                self._step = self.read_step()
        else:
            shutil.rmtree(directory, ignore_errors=True)
            try:
                os.makedirs(dvclive.dir, exist_ok=True)
            except Exception as ex:
                raise DvcLiveError(
                    "dvc-live cannot create log dir - {}".format(ex)
                )

    @property
    def dir(self):
        return self._dir

    @property
    def history_path(self):
        path = os.path.join(self.dir, "history")
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    @property
    def summary_path(self):
        return os.path.join(self.dir, "latest.json")

    @property
    def metrics_path(self):
        path = os.path.join(self.dir, "metrics")
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def next_step(self):
        write_yaml(self._metrics, self.summary_path)
        self._metrics.clear()

        self._step += 1

    def log(self, name: str, val: float, step: int = None):
        if not self.dir:
            raise DvcLiveError(
                "Initialization error - call 'dvclive.init()' before "
                "'dvclive.log()'"
            )

        ts = int(time.time() * 1000)

        if not isinstance(val, (int, float)):
            raise DvcLiveError(
                "Metrics '{}' has not supported type {}".format(
                    name, type(val)
                )
            )
        if step:
            self._step = step

        all_path = os.path.join(self.history_path, name + SUFFIX_TSV)
        self._metrics[name] = val

        d = OrderedDict([("timestamp", ts), ("step", self._step), (name, val)])
        update_tsv(d, all_path)
        write_json(d, os.path.join(self.metrics_path, name + ".json"))

    def read_step(self):
        # ToDo
        return 66


def _find_dvc_root(root=None):
    if not root:
        root = os.getcwd()

    root = os.path.realpath(root)

    if not os.path.isdir(root):
        # TODO
        raise Exception("its not dir!")

    def dvc_dir(dirname):
        return os.path.join(dirname, ".dvc")

    while True:
        if os.path.exists(dvc_dir(root)):
            return root
        if os.path.ismount(root):
            break
        root = os.path.dirname(root)
    # TODO
    raise Exception("No DVC HERE!")


def make_dvc_checkpoint():
    import builtins
    from time import sleep

    if os.getenv("DVC_CHECKPOINT") is None:
        return

    root_dir = _find_dvc_root()
    signal_file = os.path.join(root_dir, ".dvc", "tmp", "DVC_CHECKPOINT")

    with builtins.open(signal_file, "w") as fobj:
        # NOTE: force flushing/writing empty file to disk, otherwise when
        # run in certain contexts (pytest) file may not actually be written
        fobj.write("")
        fobj.flush()
        os.fsync(fobj.fileno())
    while os.path.exists(signal_file):
        sleep(1)


dvclive = DvcLive()


def init(directory: str, is_continue: bool = False, step: int = 0):
    dvclive.init(directory, is_continue, step)
