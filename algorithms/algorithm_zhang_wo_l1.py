import torch
from data_splits import DataSplits
from algorithms.algorithm_zhang import Algorithm_zhang


class Algorithm_zhang_wl1(Algorithm_zhang):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)

    def get_lambda(self, epoch):
        return 0
