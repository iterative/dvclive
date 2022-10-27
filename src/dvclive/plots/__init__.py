from .image import Image
from .metric import Metric
from .sklearn import Calibration, ConfusionMatrix, Det, PrecisionRecall, Roc
from .utils import NumpyEncoder  # noqa: F401

SKLEARN_PLOTS = {
    "calibration": Calibration,
    "confusion_matrix": ConfusionMatrix,
    "det": Det,
    "precision_recall": PrecisionRecall,
    "roc": Roc,
}
PLOT_TYPES = (*SKLEARN_PLOTS.values(), Metric, Image)
