from algorithm import Algorithm
from sklearn.decomposition import PCA
import numpy as np


class AlgorithmPCALoading(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)
        self.indices = None

    def get_selected_indices(self):
        pca = PCA(n_components=self.target_size)
        pca.fit(self.splits.train_x)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        feature_importance = np.sum(np.abs(loadings), axis=1)
        feature_ranking = np.argsort(feature_importance)[::-1]
        self.indices = feature_ranking[:self.target_size]
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]

    def get_name(self):
        return "pcal"