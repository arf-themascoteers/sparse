import torch.nn as nn
import torch


class BSNetFC(nn.Module):
    def __init__(self, bands):
        super().__init__()
        torch.manual_seed(3)
        self.bands = bands
        self.weighter = nn.Sequential(
            nn.BatchNorm1d(self.bands),
            nn.Linear(self.bands, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.channel_weight_layer = nn.Sequential(
            nn.Linear(128, self.bands),
            nn.Sigmoid()
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.bands, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.bands),
            nn.BatchNorm1d(self.bands),
            nn.Sigmoid()
        )

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X):
        channel_weights = self.weighter(X)
        channel_weights = self.channel_weight_layer(channel_weights)
        reweight_out = X * channel_weights
        output = self.encoder(reweight_out)
        return channel_weights, output






