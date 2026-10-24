import torch.nn as nn
import torch.nn.functional as F

from .data import IGNORE


def soft_dice_loss(logits, target, eps=1.0):
    """1 - Dice on the foreground probability, ignoring IGNORE pixels.

    Computed over the whole batch (not per image) so images with tiny pets
    don't produce unstable gradients.
    """
    valid = target != IGNORE
    prob = logits.softmax(dim=1)[:, 1][valid]
    tgt = (target[valid] == 1).float()
    inter = (prob * tgt).sum()
    return 1 - (2 * inter + eps) / (prob.sum() + tgt.sum() + eps)


class SegLoss(nn.Module):
    """"ce" = cross-entropy; "ce_dice" = CE + soft Dice.

    CE optimises each pixel independently; Dice directly targets region
    overlap (close to the IoU we report) and is insensitive to the
    foreground/background ratio. The combination is a common default.
    """

    def __init__(self, kind="ce_dice"):
        super().__init__()
        if kind not in ("ce", "ce_dice"):
            raise ValueError(kind)
        self.kind = kind

    def forward(self, logits, target):
        loss = F.cross_entropy(logits, target, ignore_index=IGNORE)
        if self.kind == "ce_dice":
            loss = loss + soft_dice_loss(logits, target)
        return loss
