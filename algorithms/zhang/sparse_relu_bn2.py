import torch.nn as nn
import torch.nn.functional as F
import torch


class SparseReluBN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(200)

    def forward(self, X):
        #X = torch.relu(X - 0.01)
        X = self.bn(X)
        return X
