from abc import ABC, abstractmethod
from data_splits import DataSplits
from metrics import Metrics
from datetime import datetime
from train_test_evaluator import evaluate_train_test_pair
import torch


class Algorithm(ABC):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        self.target_size = target_size
        self.splits = splits
        self.tag = tag
        self.reporter = reporter
        self.verbose = verbose
        self.selected_indices = []
        self.model = None
        self.all_indices = None
        self.reporter.create_epoch_report(tag, self.get_name(), self.splits.get_name(), self.target_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self):
        self.model, self.selected_indices = self.get_selected_indices()
        return self.selected_indices

    def transform(self, X):
        if len(self.selected_indices) != 0:
            return self.transform_with_selected_indices(X)
        return self.model.transform(X)

    def transform_with_selected_indices(self, X):
        return X[:,self.selected_indices]

    def compute_performance(self):
        start_time = datetime.now()
        selected_features = self.fit()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        evaluation_train_x = self.transform(self.splits.evaluation_train_x)
        evaluation_test_x = self.transform(self.splits.evaluation_test_x)
        oa, aa, k = evaluate_train_test_pair(evaluation_train_x, self.splits.evaluation_train_y, evaluation_test_x, self.splits.evaluation_test_y)
        return Metrics(elapsed_time, oa, aa, k, selected_features)

    @abstractmethod
    def get_selected_indices(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_all_indices(self):
        return self.all_indices

    def _set_all_indices(self, all_indices):
        self.all_indices = all_indices

    def set_selected_indices(self, selected_indices):
        self.selected_indices = selected_indices

    def is_cacheable(self):
        return True
