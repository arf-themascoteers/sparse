import torch
import torch.nn as nn


class SparseOnes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.ones_like(X).to(X.device)

