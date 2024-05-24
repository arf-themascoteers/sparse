import os
import pandas as pd
from metrics import Metrics


class Reporter:
    def __init__(self, filename="results.csv", skip_all_bands=False):
        self.skip_all_bands = skip_all_bands
        self.summary_filename = filename
        self.details_filename = f"details_{self.summary_filename}"
        self.summary_file = os.path.join("results", self.summary_filename)
        self.details_file = os.path.join("results", self.details_filename)

        if not os.path.exists(self.summary_file):
            with open(self.summary_file, 'w') as file:
                file.write("dataset,target_size,algorithm,time,oa,aa,k,selected_features\n")

        if not os.path.exists(self.details_file):
            with open(self.details_file, 'w') as file:
                file.write("dataset,target_size,fold,algorithm,repeat,time,oa,aa,k,selected_features\n")

        if not self.skip_all_bands:
            self.all_features_details_filename = f"all_features_details_{self.summary_filename}"
            self.all_features_summary_filename = f"all_features_summary_{self.summary_filename}"
            self.all_features_summary_file = os.path.join("results", self.all_features_summary_filename)
            self.all_features_details_file = os.path.join("results", self.all_features_details_filename)

            if not os.path.exists(self.all_features_summary_file):
                with open(self.all_features_summary_file, 'w') as file:
                    file.write("dataset,oa,aa,k\n")

            if not os.path.exists(self.all_features_details_file):
                with open(self.all_features_details_file, 'w') as file:
                    file.write("fold,dataset,oa,aa,k\n")

    def write_details(self, algorithm, fold, repeat, metric:Metrics):
        time = Reporter.sanitize_metric(metric.time)
        oa = Reporter.sanitize_metric(metric.oa)
        aa = Reporter.sanitize_metric(metric.aa)
        k = Reporter.sanitize_metric(metric.k)
        metric.selected_features = sorted(metric.selected_features)
        with open(self.details_file, 'a') as file:
            file.write(f"{algorithm.splits.get_name()},{algorithm.target_size},{fold},{algorithm.get_name()},"
                       f"{repeat},"
                       f"{time},{oa},{aa},{k},{'-'.join([str(i) for i in metric.selected_features])}\n")
        self.update_summary(algorithm)

    def update_summary(self, algorithm):
        df = pd.read_csv(self.details_file)
        df = df[(df["dataset"] == algorithm.splits.get_name()) & (df["algorithm"] == algorithm.get_name()) & (df["target_size"] == algorithm.target_size)]
        if len(df) == 0:
            return
        time = round(df["time"].mean(), 2)
        oa = Reporter.sanitize_metric(df["oa"].mean())
        aa = Reporter.sanitize_metric(df["aa"].mean())
        k = Reporter.sanitize_metric(df["k"].mean())
        selected_features = '||'.join(df['selected_features'].astype(str))

        df2 = pd.read_csv(self.summary_file)
        mask = ((df2["dataset"] == algorithm.splits.get_name()) & (df2["target_size"] == algorithm.target_size) & (df2["algorithm"] == algorithm.get_name()))
        if len(df2[mask]) == 0:
            df2.loc[len(df2)] = {
                "dataset":algorithm.splits.get_name(), "target_size":algorithm.target_size, "algorithm": algorithm.get_name(),
                "time":time,"oa":oa,"aa":aa,"k":k, "selected_features":selected_features
            }
        else:
            df2.loc[mask, 'time'] = time
            df2.loc[mask, 'oa'] = oa
            df2.loc[mask, 'aa'] = aa
            df2.loc[mask, 'k'] = k
            df2.loc[mask, 'selected_features'] = selected_features
        df2.to_csv(self.summary_file, index=False)

    def write_details_all_features(self, fold, dataset, oa, aa, k):
        oa = Reporter.sanitize_metric(oa)
        aa = Reporter.sanitize_metric(aa)
        k = Reporter.sanitize_metric(k)
        with open(self.all_features_details_file, 'a') as file:
            file.write(f"{fold},{dataset},{oa},{aa},{k}\n")
        self.update_summary_for_all_features(dataset)

    def update_summary_for_all_features(self, dataset):
        df = pd.read_csv(self.all_features_details_file)
        df = df[df["dataset"] == dataset]
        if len(df) == 0:
            return

        oa = max(df["oa"].mean(),0)
        aa = max(df["aa"].mean(),0)
        k = max(df["k"].mean(),0)

        df2 = pd.read_csv(self.all_features_summary_file)
        mask = (df2['dataset'] == dataset)
        if len(df2[mask]) == 0:
            df2.loc[len(df2)] = {"dataset":dataset, "oa":oa, "aa":aa, "k": k}
        else:
            df2.loc[mask, 'oa'] = oa
            df2.loc[mask, 'aa'] = aa
            df2.loc[mask, 'k'] = k
        df2.to_csv(self.all_features_summary_file, index=False)

    def get_saved_metrics(self, algorithm, fold, repeat):
        df = pd.read_csv(self.details_file)
        if len(df) == 0:
            return None
        rows = df.loc[(df["dataset"] == algorithm.splits.get_name()) & (df["target_size"] == algorithm.target_size) &
                      (df["fold"] == fold) & (df["algorithm"] == algorithm.get_name()) &
                      (df["repeat"] == repeat)
                      ]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        return Metrics(row["time"], row["oa"], row["aa"], row["k"], row["selected_features"])

    def get_saved_metrics_for_all_feature(self, fold, dataset):
        df = pd.read_csv(self.all_features_details_file)
        if len(df) == 0:
            return None, None, None
        rows = df.loc[(df['fold'] == fold) & (df['dataset'] == dataset)]
        if len(rows) == 0:
            return None, None, None
        row = rows.iloc[0]
        return row["oa"], row["aa"], row["k"]

    @staticmethod
    def sanitize_metric(metric):
        return max(metric, 0)