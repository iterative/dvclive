class DvcLiveError(Exception):
    pass


class InitializationError(DvcLiveError):
    def __init__(self):
        super().__init__(
            "Initialization error - no call was made to `dvclive.init()` or `dvclive.log()` "
        )
