import torch
import torch.nn as nn


class SM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.softmax(X, dim=1)

