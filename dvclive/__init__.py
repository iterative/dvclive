import os
import shutil
import time
from collections import OrderedDict

from dvclive.error import DvcLiveError, InitializationError
from dvclive.serialize import update_tsv, write_json


class DvcLive:
    def __init__(self):
        self._dir = None
        self._step = None
        self._metrics = {}
        self._summarize = False

    def init(
        self,
        directory: str,
        is_continue: bool = False,
        step: int = 0,
        summarize=False,
    ):
        self._dir = directory
        self._step = step
        self._summarize = summarize

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
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        return self.dir

    @property
    def summary_path(self):
        return self.dir + ".json"

    def next_step(self):
        write_json(self._metrics, self.summary_path)

        if self._summarize:
            from dvc.api.dvclive import summary

            summary(self.dir, debug=True)

        self._metrics.clear()

        self._step += 1

    def log(self, name: str, val: float, step: int = None):
        if not self.dir:
            raise InitializationError()

        ts = int(time.time() * 1000)

        if not isinstance(val, (int, float)):
            raise DvcLiveError(
                "Metrics '{}' has not supported type {}".format(
                    name, type(val)
                )
            )
        if step:
            self._step = step

        all_path = os.path.join(self.history_path, name + ".tsv")
        self._metrics[name] = val

        d = OrderedDict([("timestamp", ts), ("step", self._step), (name, val)])
        update_tsv(d, all_path)

    def read_step(self):
        # ToDo
        return 66


dvclive = DvcLive()


def init(
    directory: str, is_continue: bool = False, step: int = 0, summarize=False
):
    dvclive.init(directory, is_continue, step, summarize)
