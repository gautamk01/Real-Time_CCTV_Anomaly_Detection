"""Confusion Matrix & Metrics Calculator for Violence Detection Evaluation."""

from typing import Dict, List, Optional, Tuple


def time_str_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS timestamp string to seconds.

    Args:
        ts: Timestamp in HH:MM:SS format

    Returns:
        Total seconds as float
    """
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return float(parts[0])


def match_investigation_to_annotation(
    trigger_ts: str, annotations: List[dict]
) -> Optional[dict]:
    """Map an investigation's trigger timestamp to the ground truth annotation
    whose [start_time, end_time] range contains it.

    Args:
        trigger_ts: Investigation trigger timestamp (HH:MM:SS)
        annotations: List of annotation dicts with start_time, end_time, label

    Returns:
        Matching annotation dict, or None if no range contains the timestamp
    """
    trigger_sec = time_str_to_seconds(trigger_ts)
    for ann in annotations:
        start = time_str_to_seconds(ann["start_time"])
        end = time_str_to_seconds(ann["end_time"])
        if start <= trigger_sec <= end:
            return ann
    return None


class ConfusionMatrix:
    """Binary confusion matrix for violence detection evaluation.

    Maps 3-class system output (ALERT, CLEAR, INVESTIGATE) to binary
    (VIOLENCE / NO_VIOLENCE) and computes standard classification metrics.
    """

    def __init__(self, investigate_as: str = "VIOLENCE"):
        """Initialize confusion matrix.

        Args:
            investigate_as: How to map INVESTIGATE status.
                "VIOLENCE" - treat as positive (default, safety-first)
                "NO_VIOLENCE" - treat as negative
                "EXCLUDE" - exclude from metrics
        """
        if investigate_as not in ("VIOLENCE", "NO_VIOLENCE", "EXCLUDE"):
            raise ValueError(
                f"investigate_as must be VIOLENCE, NO_VIOLENCE, or EXCLUDE, "
                f"got {investigate_as}"
            )
        self.investigate_as = investigate_as
        self.tp = 0  # True Positive: predicted VIOLENCE, actual VIOLENCE
        self.fp = 0  # False Positive: predicted VIOLENCE, actual NO_VIOLENCE
        self.tn = 0  # True Negative: predicted NO_VIOLENCE, actual NO_VIOLENCE
        self.fn = 0  # False Negative: predicted NO_VIOLENCE, actual VIOLENCE
        self.excluded = 0
        self.unmatched = 0
        self.entries: List[dict] = []

    def _map_prediction(self, status: str) -> Optional[str]:
        """Map system status to binary prediction.

        Returns:
            "VIOLENCE", "NO_VIOLENCE", or None (if excluded)
        """
        if status == "ALERT":
            return "VIOLENCE"
        elif status == "CLEAR":
            return "NO_VIOLENCE"
        elif status == "INVESTIGATE":
            if self.investigate_as == "EXCLUDE":
                return None
            return self.investigate_as
        return "NO_VIOLENCE"

    def add(self, prediction_status: str, ground_truth_label: str,
            confidence: int = 0, timestamp: str = "", matched: bool = True):
        """Add a prediction/ground-truth pair.

        Args:
            prediction_status: System output (ALERT, CLEAR, INVESTIGATE)
            ground_truth_label: Ground truth (VIOLENCE, NO_VIOLENCE)
            confidence: System confidence score (0-100)
            timestamp: Investigation trigger timestamp
            matched: Whether this was matched to an annotation
        """
        mapped = self._map_prediction(prediction_status)

        entry = {
            "prediction_status": prediction_status,
            "mapped_prediction": mapped,
            "ground_truth": ground_truth_label,
            "confidence": confidence,
            "timestamp": timestamp,
            "matched": matched,
        }
        self.entries.append(entry)

        if not matched:
            self.unmatched += 1
            return

        if mapped is None:
            self.excluded += 1
            return

        if mapped == "VIOLENCE" and ground_truth_label == "VIOLENCE":
            self.tp += 1
        elif mapped == "VIOLENCE" and ground_truth_label == "NO_VIOLENCE":
            self.fp += 1
        elif mapped == "NO_VIOLENCE" and ground_truth_label == "NO_VIOLENCE":
            self.tn += 1
        elif mapped == "NO_VIOLENCE" and ground_truth_label == "VIOLENCE":
            self.fn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate = FP / (FP + TN)"""
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    @property
    def fnr(self) -> float:
        """False Negative Rate = FN / (FN + TP)"""
        denom = self.fn + self.tp
        return self.fn / denom if denom > 0 else 0.0

    def summary(self) -> dict:
        """Return all metrics as a dictionary."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "total": self.total,
            "excluded": self.excluded,
            "unmatched": self.unmatched,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "f1_score": round(self.f1_score, 4),
            "fpr": round(self.fpr, 4),
            "fnr": round(self.fnr, 4),
            "investigate_as": self.investigate_as,
        }

    def print_matrix(self):
        """Print formatted confusion matrix to console."""
        print(f"\n{'='*60}")
        print(f"  CONFUSION MATRIX (Binary: Violence vs No-Violence)")
        print(f"  INVESTIGATE mapped as: {self.investigate_as}")
        print(f"{'='*60}")
        print()
        print(f"                    Predicted")
        print(f"                    VIOLENCE    NO_VIOLENCE")
        print(f"  Actual VIOLENCE   {self.tp:>5}       {self.fn:>5}")
        print(f"  Actual NO_VIOL    {self.fp:>5}       {self.tn:>5}")
        print()
        print(f"  Accuracy:         {self.accuracy:.4f}")
        print(f"  Precision:        {self.precision:.4f}")
        print(f"  Recall (TPR):     {self.recall:.4f}")
        print(f"  Specificity:      {self.specificity:.4f}")
        print(f"  F1 Score:         {self.f1_score:.4f}")
        print(f"  FPR:              {self.fpr:.4f}")
        print(f"  FNR:              {self.fnr:.4f}")
        if self.excluded > 0:
            print(f"  Excluded:         {self.excluded}")
        if self.unmatched > 0:
            print(f"  Unmatched:        {self.unmatched}")
        print(f"{'='*60}\n")


def generate_roc_data(entries: List[dict]) -> List[Tuple[float, float]]:
    """Sweep confidence thresholds to generate ROC curve data points.

    For each threshold, predictions with confidence >= threshold are mapped
    to VIOLENCE, below threshold to NO_VIOLENCE.

    Args:
        entries: List of dicts with 'confidence' and 'ground_truth' keys.
            Only entries with matched=True and ground_truth set are used.

    Returns:
        List of (FPR, TPR) tuples for thresholds 0 to 100 step 5
    """
    # Filter to matched entries with valid ground truth
    valid = [
        e for e in entries
        if e.get("matched", True)
        and e.get("ground_truth") in ("VIOLENCE", "NO_VIOLENCE")
    ]

    if not valid:
        return []

    roc_points = []
    for threshold in range(0, 105, 5):
        tp = fp = tn = fn = 0
        for e in valid:
            pred = "VIOLENCE" if e["confidence"] >= threshold else "NO_VIOLENCE"
            gt = e["ground_truth"]
            if pred == "VIOLENCE" and gt == "VIOLENCE":
                tp += 1
            elif pred == "VIOLENCE" and gt == "NO_VIOLENCE":
                fp += 1
            elif pred == "NO_VIOLENCE" and gt == "NO_VIOLENCE":
                tn += 1
            else:
                fn += 1

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        roc_points.append((fpr_val, tpr))

    return roc_points
