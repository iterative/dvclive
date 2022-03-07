import json
from pathlib import Path

from .base import Data


class Plot(Data):
    suffixes = [".json"]
    subfolder = "plots"

    @property
    def output_path(self) -> Path:
        _path = self.output_folder / self.name
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path.with_suffix(".json")

    @staticmethod
    def could_log(val: object) -> bool:
        if isinstance(val, tuple) and len(val) == 2:
            return True
        return False

    @property
    def no_step_output_path(self) -> Path:
        return super().no_step_output_path.with_suffix(".json")

    @property
    def summary(self):
        return {}

    @staticmethod
    def write_json(content, output_file):
        with open(output_file, "w") as f:
            json.dump(content, f, indent=4)

    def no_step_dump(self) -> None:
        raise NotImplementedError

    def first_step_dump(self) -> None:
        raise NotImplementedError(
            "DVCLive plots can only be used in no-step mode."
        )

    def step_dump(self) -> None:
        raise NotImplementedError(
            "DVCLive plots can only be used in no-step mode."
        )

    @staticmethod
    def get_properties():
        raise NotImplementedError


class Roc(Plot):
    @staticmethod
    def get_properties():
        return {
            "x": "fpr",
            "y": "tpr",
            "title": "Receiver operating characteristic (ROC)",
            "x_label": "False Positive Rate",
            "y_label": "True Positive Rate",
        }

    def no_step_dump(self) -> int:
        from sklearn import metrics

        fpr, tpr, roc_thresholds = metrics.roc_curve(
            y_true=self.val[0], y_score=self.val[1], **self._dump_kwargs
        )
        roc = {
            "roc": [
                {"fpr": fp, "tpr": tp, "threshold": t}
                for fp, tp, t in zip(fpr, tpr, roc_thresholds)
            ]
        }
        self.write_json(roc, self.output_path)


class PrecisionRecall(Plot):
    @staticmethod
    def get_properties():
        return {
            "x": "recall",
            "y": "precision",
            "title": "Precision-Recall Curve",
            "x_label": "Recall",
            "y_label": "Precision",
        }

    def no_step_dump(self) -> int:
        from sklearn import metrics

        precision, recall, prc_thresholds = metrics.precision_recall_curve(
            y_true=self.val[0], probas_pred=self.val[1], **self._dump_kwargs
        )

        prc = {
            "precision_recall": [
                {"precision": p, "recall": r, "threshold": t}
                for p, r, t in zip(precision, recall, prc_thresholds)
            ]
        }
        self.write_json(prc, self.output_path)


class Det(Plot):
    @staticmethod
    def get_properties():
        return {
            "x": "fpr",
            "y": "fnr",
            "title": "Detection error tradeoff (DET)",
            "x_label": "False Positive Rate",
            "y_label": "False Negative Rate",
        }

    def no_step_dump(self) -> int:
        from sklearn import metrics

        fpr, fnr, roc_thresholds = metrics.det_curve(
            y_true=self.val[0], y_score=self.val[1], **self._dump_kwargs
        )

        det = {
            "det": [
                {"fpr": fp, "fnr": fn, "threshold": t}
                for fp, fn, t in zip(fpr, fnr, roc_thresholds)
            ]
        }
        self.write_json(det, self.output_path)


class ConfusionMatrix(Plot):
    @staticmethod
    def get_properties():
        return {
            "template": "confusion",
            "x": "actual",
            "y": "predicted",
            "title": "Confusion Matrix",
            "x_label": "True Label",
            "y_label": "Predicted Label",
        }

    def no_step_dump(self) -> int:
        cm = [
            {"actual": str(actual), "predicted": str(predicted)}
            for actual, predicted in zip(self.val[0], self.val[1])
        ]
        self.write_json(cm, self.output_path)


class Calibration(Plot):
    @staticmethod
    def get_properties():
        return {
            "x": "prob_pred",
            "y": "prob_true",
            "title": "Calibration Curve",
            "x_label": "Mean Predicted Probability",
            "y_label": "Fraction of Positives",
        }

    def no_step_dump(self) -> int:
        from sklearn import calibration

        prob_true, prob_pred = calibration.calibration_curve(
            y_true=self.val[0], y_prob=self.val[1], **self._dump_kwargs
        )

        calibration = {
            "calibration": [
                {"prob_true": pt, "prob_pred": pp}
                for pt, pp in zip(prob_true, prob_pred)
            ]
        }
        self.write_json(calibration, self.output_path)
