import torch
import torch.nn as nn


class Sparse8(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def forward(self, X):
        return X
        if self.parent.epoch < 100:
            return X
        top = 0.1 * (self.parent.epoch - 100) / (self.parent.total_epoch - 100)
        X = torch.where(torch.abs(X) < top, 0, X)
        return X

