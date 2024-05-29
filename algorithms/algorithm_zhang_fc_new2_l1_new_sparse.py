from data_splits import DataSplits
from algorithms.algorithm_zhang_fc_new2_l1_ns import Algorithm_zhang_fc_new2_l1_ns
from algorithms.zhang.sparse_new import SparseNew


class Algorithm_zhang_fc_new2_l1_new_sparse(Algorithm_zhang_fc_new2_l1_ns):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.zhangnet.sparse = SparseNew()
