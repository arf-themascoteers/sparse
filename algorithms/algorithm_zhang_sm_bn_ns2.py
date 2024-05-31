from data_splits import DataSplits
from algorithms.algorithm_zhang import Algorithm_zhang
import torch.nn as nn
import torch
import math


class Algorithm_zhang_sm_bn_ns2(Algorithm_zhang):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.zhangnet.sparse = nn.Sequential()
        self.zhangnet.weighter[3] = nn.Softmax(dim=1).to(self.device)
        self.zhangnet.weighter[3] = nn.Sequential(
            self.zhangnet.weighter[3],
            nn.BatchNorm1d(self.zhangnet.bands)
        ).to(self.device)

    def get_lambda(self, epoch):
        return 0.0001 * math.exp(-epoch/self.total_epoch)
