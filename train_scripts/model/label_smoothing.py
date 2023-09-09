import torch
from torch import nn


class LabelSmoothingBCELoss(nn.Module):
    """Implement label smoothing for the BCELoss."""

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.smoothing = smoothing

    def forward(self, x, target):
        smooth_target = target * (1 - self.smoothing * 2) + self.smoothing  # 0, 1 -> 0.1, 0.9
        return self.criterion(x, smooth_target)