from algorithm import Algorithm
from algorithms.bsdr4000.bsdr import BSDR
import numpy as np
import torch


class AlgorithmBSDR4000(Algorithm):
    def __init__(self, target_size, splits, repeat, fold, verbose=True):
        super().__init__(target_size, splits, repeat, fold)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        self.verbose = verbose
        class_size = len(np.unique(self.splits.train_y))
        self.bsdr = BSDR(self.target_size, class_size, self.splits, self.get_name(), self.repeat, self.fold, self.verbose)

    def get_selected_indices(self):
        self.bsdr.fit(self.splits.train_x, self.splits.train_y, self.splits.validation_x, self.splits.validation_y)
        return self.bsdr, self.bsdr.get_indices()

    def get_name(self):
        return "bsdr4000"