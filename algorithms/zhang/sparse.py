import torch
import torch.nn as nn


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 100/128

    def forward(self, X):
        X = torch.where(X < self.k, 0, X)
        return X

