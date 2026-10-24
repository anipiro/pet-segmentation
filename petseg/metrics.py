"""Segmentation metrics.

Evaluation target: trimap border counts as foreground, so every model is
scored against the same labels regardless of how it was trained.

  fg_iou     IoU of the pet class (headline metric for background removal)
  miou       mean of pet IoU and background IoU
  dice       F1 of the pet class
  pixel_acc  fraction of correct pixels (misleading alone: ~70% is background)

Dataset-level IoU accumulates intersections/unions over all pixels; we also
keep per-image IoU for error analysis (a model can score well overall while
failing completely on a few images).
"""
import numpy as np
import torch


class ConfusionMeter:
    """Accumulates a 2x2 confusion matrix (rows = target, cols = prediction)."""

    def __init__(self):
        self.cm = torch.zeros(2, 2, dtype=torch.long)

    def update(self, pred, target):
        valid = target != 255
        idx = target[valid] * 2 + pred[valid]
        self.cm += torch.bincount(idx.flatten(), minlength=4).reshape(2, 2).cpu()

    def compute(self) -> dict:
        cm = self.cm.double()
        tp = cm.diag()
        iou = tp / (cm.sum(0) + cm.sum(1) - tp).clamp(min=1)
        dice_fg = 2 * tp[1] / (cm[1].sum() + cm[:, 1].sum()).clamp(min=1)
        return {
            "fg_iou": iou[1].item(),
            "bg_iou": iou[0].item(),
            "miou": iou.mean().item(),
            "dice": dice_fg.item(),
            "pixel_acc": (tp.sum() / cm.sum().clamp(min=1)).item(),
        }


def image_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Foreground IoU for a single image (1.0 if both are empty)."""
    p, t = pred == 1, target == 1
    union = np.logical_or(p, t).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, t).sum() / union)
