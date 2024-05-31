import torch
from data_splits import DataSplits
from algorithms.algorithm_zhang import Algorithm_zhang
import torch.nn as nn


class Algorithm_zhang_fc(Algorithm_zhang):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.zhangnet.classnet = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.zhangnet.bands,128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128, self.zhangnet.number_of_classes)
        ).to(self.device)

