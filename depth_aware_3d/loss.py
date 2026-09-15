import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyAwareLoss(nn.Module):
    """BCE + Dice + connectivity penalty for topology-clean predictions.

    The connectivity penalty discourages fragmented predictions by
    comparing the original prediction with a morphologically opened version
    (erosion followed by dilation). Fragments that disappear after opening
    are penalized.
    """

    def __init__(self, dice_weight=0.4, topo_weight=0.1):
        super().__init__()
        self.dice_weight = dice_weight
        self.topo_weight = topo_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        p = pred.reshape(-1)
        t = target.reshape(-1)
        return 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)

    def connectivity_penalty(self, pred):
        pred_prob = torch.sigmoid(pred)
        kernel = torch.ones(1, 1, 3, 3, 3, device=pred.device) / 27
        eroded = F.conv3d(pred_prob, kernel, padding=1)
        dilated = F.conv3d(eroded, kernel, padding=1)
        fragments = F.relu(pred_prob - dilated)
        return fragments.mean()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        topo = self.connectivity_penalty(pred)
        return (1 - self.dice_weight - self.topo_weight) * bce + self.dice_weight * dice + self.topo_weight * topo
