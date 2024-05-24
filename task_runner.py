from ds_manager import DSManager
from algorithm_creator import AlgorithmCreator
from reporter import Reporter
import pandas as pd
from metrics import Metrics
from algorithm import Algorithm
from train_test_evaluator import evaluate_train_test_pair


class TaskRunner:
    def __init__(self, task, repeat=1, folds=1, filename="results.csv", skip_all_bands=False, verbose=False):
        self.task = task
        self.repeat = repeat
        self.folds = folds
        self.skip_all_bands = skip_all_bands
        self.verbose = verbose
        self.reporter = Reporter(filename, self.skip_all_bands)
        self.cache = pd.DataFrame(columns=["dataset","fold","algorithm","repeat",
                                           "oa","aa","k","time","selected_features"])

    def evaluate(self):
        for dataset_name in self.task["datasets"]:
            dataset = DSManager(name=dataset_name, folds=self.folds)
            if not self.skip_all_bands:
                self.evaluate_for_all_features(dataset)
            for target_size in self.task["target_sizes"]:
                for fold, splits in enumerate(dataset.get_k_folds()):
                    print(splits.splits_description())
                    for algorithm in self.task["algorithms"]:
                        for repeat_no in range(self.repeat):
                            algorithm_object = AlgorithmCreator.create(algorithm, target_size, splits, repeat_no, fold, self.verbose)
                            self.process_a_case(algorithm_object, fold, repeat_no)

    def process_a_case(self, algorithm:Algorithm, fold, repeat):
        metric = self.reporter.get_saved_metrics(algorithm, fold, repeat)
        if metric is not None:
            print(f"{algorithm.splits.get_name()} for size {algorithm.target_size} for fold {fold} for {algorithm.get_name()} was done")
            return
        metric = self.get_results_for_a_case(algorithm, fold, repeat)
        self.reporter.write_details(algorithm, fold, repeat, metric)

    def get_results_for_a_case(self, algorithm:Algorithm, fold, repeat):
        metric = self.get_from_cache(algorithm, fold, repeat)
        if metric is not None:
            print(f"Selected features got from cache for {algorithm.splits.get_name()} for size {algorithm.target_size} for fold {fold} for {algorithm.get_name()}")
            oa, aa, k = algorithm.compute_performance_with_selected_indices(metric.selected_features)
            return Metrics(metric.time, oa, aa, k, metric.selected_features)
        print(f"Computing {algorithm.get_name()} {algorithm.splits.get_name()} Fold {fold} Repeat {repeat}")
        metric = algorithm.compute_performance()
        self.save_to_cache(algorithm, fold, repeat, metric)
        return metric

    def save_to_cache(self, algorithm, fold, repeat, metric:Metrics):
        if not algorithm.is_independent_of_target_size():
            return
        self.cache.loc[len(self.cache)] = {
            "dataset":algorithm.splits.get_name(), "algorithm": algorithm.get_name(),
            "fold": fold, "repeat":repeat,
            "time":metric.time,"oa":metric.oa,"aa":metric.aa,"k":metric.k, "selected_features":algorithm.get_all_indices()
        }

    def get_from_cache(self, algorithm:Algorithm, fold, repeat_no):
        if not algorithm.is_independent_of_target_size():
            return None
        if len(self.cache) == 0:
            return None
        rows = self.cache.loc[
            (self.cache["dataset"] == algorithm.splits.get_name()) &
            (self.cache["fold"] == fold) &
            (self.cache["algorithm"] == algorithm.get_name()) &
            (self.cache["repeat"] == repeat_no)
        ]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        selected_features = row["selected_features"][0:algorithm.target_size]
        return Metrics(row["time"], row["oa"],row["aa"], row["k"], selected_features)

    def evaluate_for_all_features(self, dataset):
        for fold, splits in enumerate(dataset.get_k_folds()):
            self.evaluate_for_all_features_fold(fold, splits)

    def evaluate_for_all_features_fold(self, fold, splits):
        oa, aa, k = self.reporter.get_saved_metrics_for_all_feature(fold, splits.get_name())
        if oa is not None and k is not None:
            print(f"Fold {fold} for {splits.get_name()} was done")
            return
        oa, aa, k = evaluate_train_test_pair(splits.evaluation_train_x, splits.evaluation_train_y,
                                                    splits.evaluation_test_x, splits.evaluation_test_y)
        self.reporter.write_details_all_features(fold, splits.get_name(), oa, aa, k)






