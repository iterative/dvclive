from .image import Image
from .metric import Metric
from .plot import Calibration, ConfusionMatrix, Det, PrecisionRecall, Roc
from .utils import NumpyEncoder  # noqa: F401

PLOTS = {
    "calibration": Calibration,
    "confusion_matrix": ConfusionMatrix,
    "det": Det,
    "precision_recall": PrecisionRecall,
    "roc": Roc,
}
DATA_TYPES = (*PLOTS.values(), Metric, Image)
