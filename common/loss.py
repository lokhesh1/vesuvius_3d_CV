import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return (1 - self.dice_weight) * bce + self.dice_weight * dice


def compute_dice(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0
    return (2.0 * intersection / union).item()
