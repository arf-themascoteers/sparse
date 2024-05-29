import torch
from data_splits import DataSplits
from algorithms.algorithm_zhang_fc_new_l1 import Algorithm_zhang_fc_new_l1
import torch.nn as nn


class Algorithm_zhang_fc_new2_l1_ns(Algorithm_zhang_fc_new_l1):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.zhangnet.sparse = nn.Sequential().to(self.device)
