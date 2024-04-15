import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-8

    def forward(self, input, target):
        num = 2 * torch.sum(torch.sigmoid(input) * target, axis=(1, 2, 3))
        den = torch.sum(torch.sigmoid(input) + target, axis=(1, 2, 3))
        res = 1 - (num + self.smooth) / (den + self.smooth)
        return torch.mean(res)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return self.alpha * self.dice_loss(pred, target) + (1 - self.alpha) * self.bce(pred, target)
