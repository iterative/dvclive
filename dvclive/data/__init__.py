from .image import Image
from .plot import Calibration, ConfusionMatrix, Det, PrecisionRecall, Roc
from .scalar import Scalar
from .utils import NumpyEncoder  # noqa: F401

PLOTS = {
    "calibration": Calibration,
    "confusion_matrix": ConfusionMatrix,
    "det": Det,
    "precision_recall": PrecisionRecall,
    "roc": Roc,
}
DATA_TYPES = list(PLOTS.values()) + [Scalar, Image]
