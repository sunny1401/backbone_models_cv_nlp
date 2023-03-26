import torch
from torch import nn


def RMSELoss(y_pred, y_true):
    """
    RMSELoss -> TODO
    """
    return torch.sqrt(
        torch.mean(
            (y_pred - y_true)**2)
        )

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive):
        distance = nn.PairwiseDistance()(anchor, positive)
        loss = torch.mean(torch.clamp(self.margin - distance, min=0.0) ** 2)
        return loss