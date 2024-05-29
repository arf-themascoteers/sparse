import torch
from data_splits import DataSplits
from algorithms.algorithm_zhang import Algorithm_zhang
import torch.nn as nn


class Algorithm_zhang_relu(Algorithm_zhang):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.zhangnet.weighter[3] = nn.ReLU()

    def get_lambda(self, epoch):
        return 0
