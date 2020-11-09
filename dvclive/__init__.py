import os
import shutil
import time
from collections import OrderedDict

from dvclive.error import DvcLiveError
from dvclive.io import update_tsv, write_json

SUFFIX_TSV = ".tsv"
SUFFIX_JSON = ".json"


class DvcLive:
    def __init__(self):
        self._dir = None
        self._epoch = None

    def init(self, directory: str, is_continue: bool = False, epoch: int = 0):
        self._dir = directory
        self._epoch = epoch

        if is_continue:
            if epoch == 0:
                self._epoch = self.read_epoche()
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
    def summary_dir(self):
        path = os.path.join(self.dir, "all")
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def next_epoch(self):
        self._epoch += 1

    def log(self, name: str, val: float, epoche: int = None):
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
        if epoche:
            self._epoch = epoche

        all_path = os.path.join(self.summary_dir, name + SUFFIX_TSV)
        fpath = os.path.join(self.dir, name + SUFFIX_JSON)

        d = OrderedDict(
            [("timestamp", ts), ("epoch", self._epoch), (name, val)]
        )
        write_json(d, fpath)
        update_tsv(d, all_path)

    def read_epoche(self):
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


def init(directory: str, is_continue: bool = False, epoch: int = 0):
    dvclive.init(directory, is_continue, epoch)
