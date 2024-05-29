import torch
from data_splits import DataSplits
from algorithms.algorithm_zhang import Algorithm_zhang


class Algorithm_zhang_new_l1(Algorithm_zhang):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)

    def l1_loss(self, channel_weights):
        channel_weights = torch.mean(channel_weights)
        return channel_weights
