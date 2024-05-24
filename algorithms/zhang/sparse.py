import torch
import torch.nn as nn


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, X):
        batch_size = X.shape[0]
        threshold = 100
        k = torch.tensor(threshold/batch_size).to(X.device)
        X = torch.where(X < k, 0, X)
        return X

