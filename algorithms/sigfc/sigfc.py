import torch.nn as nn
import torch
from algorithms.sig.sparse import Sparse


class SigFC(nn.Module):
    def __init__(self, bands, number_of_classes, last_layer_input):
        super().__init__()
        torch.manual_seed(3)
        self.bands = bands
        self.number_of_classes = number_of_classes
        self.last_layer_input = last_layer_input
        self.weighter = nn.Sequential(
            nn.Linear(self.bands, 512),
            nn.ReLU(),
            nn.Linear(512, self.bands),
            nn.Sigmoid()
        )
        self.classnet = nn.Sequential(
            nn.Linear(self.bands,64),
            nn.LeakyReLU(),
            nn.Linear(64, self.number_of_classes)
        )
        self.sparse = Sparse()
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X):
        channel_weights = self.weighter(X)
        sparse_weights = self.sparse(channel_weights)
        reweight_out = X * sparse_weights
        reweight_out = reweight_out.reshape(reweight_out.shape[0],1,reweight_out.shape[1])
        output = self.classnet(reweight_out)
        return channel_weights, sparse_weights, output






