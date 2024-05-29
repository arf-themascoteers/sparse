from data_splits import DataSplits
from algorithms.algorithm_zhang_fc_new_l1 import Algorithm_zhang_fc_new_l1
import torch.nn as nn
from algorithms.zhang.sparse_relu_bn2 import SparseReluBN2
import torch


class Algorithm_zhang_sm6(Algorithm_zhang_fc_new_l1):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.zhangnet.sparse = nn.Sequential()
        self.zhangnet.weighter[3] = nn.Softmax(dim=1).to(self.device)
        self.zhangnet.weighter[3] = nn.Sequential(
            self.zhangnet.weighter[3],
            nn.BatchNorm1d(self.zhangnet.bands),
            nn.ReLU()
        ).to(self.device)

    def get_lambda(self, epoch):
        if epoch < 100:
            return 0.0
        else:
            return 0.7 * (epoch - 100) / (self.total_epoch - 100)

    def l1_loss(self, channel_weights):
        return torch.mean(channel_weights)
