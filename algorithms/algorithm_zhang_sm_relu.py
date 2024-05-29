from data_splits import DataSplits
from algorithms.algorithm_zhang_sm import Algorithm_zhang_sm
from algorithms.zhang.sparse_relu import SparseRelu


class Algorithm_zhang_sm_relu(Algorithm_zhang_sm):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.zhangnet.sparse = SparseRelu()


    def get_lambda(self, epoch):
        return 0
