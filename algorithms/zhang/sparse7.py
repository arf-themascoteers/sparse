import torch
import torch.nn as nn


class Sparse7(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X = torch.where(X < 0.001, 0, X)
        return X

