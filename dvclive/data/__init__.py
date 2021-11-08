from .image import Image
from .plot import Calibration, ConfusionMatrix, Det, PrecisionRecall, Roc
from .scalar import Scalar

PLOTS = {
    "calibration": Calibration,
    "confusion_matrix": ConfusionMatrix,
    "det": Det,
    "precision_recall": PrecisionRecall,
    "roc": Roc,
}
DATA_TYPES = list(PLOTS.values()) + [Scalar, Image]
