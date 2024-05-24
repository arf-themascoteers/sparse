from algorithm import Algorithm
from sklearn.linear_model import Lasso
import numpy as np


class AlgorithmLasso(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)
        self.indices = None

    def get_selected_indices(self):
        lasso = Lasso(alpha=0.01)
        lasso.fit(self.splits.train_x, self.splits.train_y)
        all_indices = np.argsort(np.abs(lasso.coef_))[::-1]
        super()._set_all_indices(all_indices)
        self.indices = all_indices[:self.target_size]
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]

    def get_name(self):
        return "lasso"