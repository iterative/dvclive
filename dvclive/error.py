class DvcLiveError(Exception):
    pass


class InitializationError(DvcLiveError):
    def __init__(self):
        super().__init__(
            "Initialization error - call `dvclive.init()` before "
            "`dvclive.log()`"
        )
