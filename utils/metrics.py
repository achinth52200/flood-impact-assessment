"""
Segmentation evaluation metrics for binary flood mapping.
All functions operate on numpy arrays or PyTorch tensors.
"""
import numpy as np
import torch


def _to_binary(pred, threshold: float = 0.5):
    """Convert logits/probabilities tensor or array to binary (0/1)."""
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred) if pred.min() < 0 else pred
        return (pred >= threshold).long()
    # numpy
    pred = 1.0 / (1.0 + np.exp(-pred)) if pred.min() < 0 else pred
    return (pred >= threshold).astype(np.uint8)


def iou_score(pred, target, threshold: float = 0.5, eps: float = 1e-6):
    """Intersection over Union (Jaccard index)."""
    p = _to_binary(pred, threshold).reshape(-1)
    t = (target > 0.5).reshape(-1)
    if isinstance(p, torch.Tensor):
        inter = (p & t.bool()).sum().float()
        union = (p | t.bool()).sum().float()
    else:
        inter = (p & t).sum()
        union = (p | t).sum()
    return float((inter + eps) / (union + eps))


def f1_score(pred, target, threshold: float = 0.5, eps: float = 1e-6):
    """F1 / Dice coefficient."""
    p = _to_binary(pred, threshold).reshape(-1)
    t = (target > 0.5).reshape(-1)
    if isinstance(p, torch.Tensor):
        tp = (p & t.bool()).sum().float()
        fp = (p & ~t.bool()).sum().float()
        fn = (~p.bool() & t.bool()).sum().float()
    else:
        tp = (p & t).sum()
        fp = (p & ~t).sum()
        fn = (~p.astype(bool) & t.astype(bool)).sum()
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def precision_score(pred, target, threshold: float = 0.5, eps: float = 1e-6):
    p = _to_binary(pred, threshold).reshape(-1)
    t = (target > 0.5).reshape(-1)
    if isinstance(p, torch.Tensor):
        tp = (p & t.bool()).sum().float()
        fp = (p & ~t.bool()).sum().float()
    else:
        tp = (p & t).sum()
        fp = (p & ~t.astype(bool)).sum()
    return float((tp + eps) / (tp + fp + eps))


def recall_score(pred, target, threshold: float = 0.5, eps: float = 1e-6):
    p = _to_binary(pred, threshold).reshape(-1)
    t = (target > 0.5).reshape(-1)
    if isinstance(p, torch.Tensor):
        tp = (p & t.bool()).sum().float()
        fn = (~p.bool() & t.bool()).sum().float()
    else:
        tp = (p & t).sum()
        fn = (~p.astype(bool) & t.astype(bool)).sum()
    return float((tp + eps) / (tp + fn + eps))


def pixel_accuracy(pred, target, threshold: float = 0.5):
    p = _to_binary(pred, threshold).reshape(-1)
    t = (target > 0.5).reshape(-1)
    if isinstance(p, torch.Tensor):
        return float((p == t.long()).float().mean())
    return float((p == t.astype(np.uint8)).mean())


def compute_all(pred, target, threshold: float = 0.5) -> dict:
    """Return a dict with all metrics."""
    return {
        "iou":       iou_score(pred, target, threshold),
        "f1":        f1_score(pred, target, threshold),
        "precision": precision_score(pred, target, threshold),
        "recall":    recall_score(pred, target, threshold),
        "accuracy":  pixel_accuracy(pred, target, threshold),
    }


class MetricTracker:
    """Running average tracker for a training epoch."""
    def __init__(self):
        self._totals: dict = {}
        self._counts: dict = {}

    def update(self, metrics: dict, n: int = 1):
        for k, v in metrics.items():
            self._totals[k] = self._totals.get(k, 0.0) + float(v) * n
            self._counts[k] = self._counts.get(k, 0)   + n

    def means(self) -> dict:
        return {k: self._totals[k] / self._counts[k]
                for k in self._totals}

    def reset(self):
        self._totals.clear()
        self._counts.clear()
