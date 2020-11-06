import os
import shutil
import time
from collections import OrderedDict

from dvclive.error import DvcLiveError
from dvclive.io import update_tsv

SUFFIX_TSV = ".tsv"


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

        fpath = os.path.join(self.dir, name + SUFFIX_TSV)

        d = OrderedDict(
            [("timestamp", ts), ("epoch", self._epoch), (name, val)]
        )
        update_tsv(d, fpath)

    def read_epoche(self):
        # ToDo
        return 66


dvclive = DvcLive()


def init(directory: str, is_continue: bool = False, epoch: int = 0):
    dvclive.init(directory, is_continue, epoch)
