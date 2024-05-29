import torch
import torch.nn as nn


class SparseNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, X):
        k = torch.mean(X)/3
        X = torch.where(X < k, 0, X)
        return X

