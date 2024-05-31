import torch.nn as nn
import torch
from algorithms.zhang.sparse import Sparse


class ZhangNetCNNPar(nn.Module):
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
        self.classnet1 = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(last_layer_input,self.number_of_classes)
        )
        self.classnet2 = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(last_layer_input,self.number_of_classes)
        )
        self.sparse = Sparse()
        self.sparse.k = 0.3
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X):
        channel_weights = self.weighter(X)
        sparse_weights = self.sparse(channel_weights)
        cw_reweight_out = X * channel_weights
        cw_reweight_out = cw_reweight_out.reshape(cw_reweight_out.shape[0],1,cw_reweight_out.shape[1])
        sparse_reweight_out = X * sparse_weights
        sparse_reweight_out = sparse_reweight_out.reshape(sparse_reweight_out.shape[0],1,sparse_reweight_out.shape[1])
        output1 = self.classnet1(cw_reweight_out)
        output2 = self.classnet2(sparse_reweight_out)
        return channel_weights, sparse_weights, output1, output2







