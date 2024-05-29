import torch.nn as nn
import torch.nn.functional as F
import torch


class SparseLReluBN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X = F.leaky_relu(X - 0.08)
        return X
