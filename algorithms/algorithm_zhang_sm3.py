from data_splits import DataSplits
from algorithms.algorithm_zhang_fc_new_l1 import Algorithm_zhang_fc_new_l1
import torch.nn as nn
from algorithms.zhang.sparse_lrelu_bn import SparseLReluBN


class Algorithm_zhang_sm2(Algorithm_zhang_fc_new_l1):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.zhangnet.sparse = SparseLReluBN().to(self.device)
        self.zhangnet.weighter[3] = nn.Softmax(dim=1).to(self.device)
        self.zhangnet.weighter[3] = nn.Sequential(
            self.zhangnet.weighter[3],
            nn.BatchNorm1d(self.zhangnet.bands)
        ).to(self.device)

    def get_lambda(self, epoch):
        return 0
