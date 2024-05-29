import torch.nn as nn
import torch.nn.functional as F


class SparseRelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return F.relu(X-0.03)
