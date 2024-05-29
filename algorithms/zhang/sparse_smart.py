import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        F.relu(X-0.03)
