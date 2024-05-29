import torch.nn as nn
import torch.nn.functional as F
import torch


class SparseReluBN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X = torch.relu(X - 0.05)
        return X
