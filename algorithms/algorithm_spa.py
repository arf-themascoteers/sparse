from algorithm import Algorithm
from auswahl import SPA, VIP


class AlgorithmSPA(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)

    def get_selected_indices(self):
        vip = VIP()
        selector = SPA(n_features_to_select=self.target_size)
        vip.fit(self.splits.train_x, self.splits.train_y)
        mask = vip.vips_ > 0.3
        selector.fit(self.splits.train_x, self.splits.train_y, mask=mask)
        return selector, selector.get_support(indices=True)

    def get_name(self):
        return "spa"
