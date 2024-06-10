import torch.nn as nn
import torch
from algorithms.zhang_min.sparse_min import SparseMin


class ZhangNetMin(nn.Module):
    def __init__(self, bands, number_of_classes):
        super().__init__()
        torch.manual_seed(3)
        self.bands = bands
        self.number_of_classes = number_of_classes
        self.weighter = nn.Sequential(
            nn.Linear(self.bands, 100),
            nn.LeakyReLU(),
            nn.Linear(100, self.bands),
            nn.Softmax(dim=1)
        )
        self.classnet = nn.Sequential(
            nn.Linear(self.bands, 100),
            nn.LeakyReLU(),
            nn.Linear(100,self.number_of_classes)
        )
        self.sparse = SparseMin()
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X):
        channel_weights = self.weighter(X)
        #sparse_weights = self.sparse(channel_weights)
        sparse_weights = channel_weights
        reweight_out = X * sparse_weights
        output = self.classnet(reweight_out)
        return channel_weights, sparse_weights, output






