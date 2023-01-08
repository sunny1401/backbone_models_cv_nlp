import torch


def RMSELoss(y_pred, y_true):
    """
    RMSELoss -> TODO
    """
    return torch.sqrt(
        torch.mean(
            (y_pred - y_true)**2)
        )