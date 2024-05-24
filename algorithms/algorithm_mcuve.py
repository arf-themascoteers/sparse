from algorithm import Algorithm
from auswahl import MCUVE


class AlgorithmMCUVE(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)

    def get_selected_indices(self):
        selector = MCUVE(n_features_to_select=self.target_size)
        selector.fit(self.splits.train_x, self.splits.train_y)
        return selector, selector.get_support(indices=True)

    def get_name(self):
        return "mcuve"