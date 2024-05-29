import torch
import torch.nn as nn


class Sparse8(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def forward(self, X):
        if self.parent.epoch < 100:
            return X
        X = torch.where(X < 0.001, 0, X)
        return X

