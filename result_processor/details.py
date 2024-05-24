import pandas as pd
import os
import plotters.utils as utils

main_df = pd.DataFrame()
all_df = pd.DataFrame()
root = "../saved_results"
locations = [os.path.join(root, subfolder) for subfolder in os.listdir(root)]
locations = [loc for loc in locations if os.path.exists(loc)]
algorithms = ["mcuve","bsnet","bsdr"]
datasets = ["lucas"]
targets = [5,10,15,20,25,30]
df2 = pd.DataFrame(columns=["dataset","target_size","algorithm","time","oa","k"])


def add_df(base_df, path):
    df = pd.read_csv(path)
    if len(df) == 0:
        print(f"Empty {path}")
        return base_df
    df['source'] = path
    if base_df is None:
        return df
    else:
        base_df = pd.concat([base_df, df], axis=0)
        return base_df


def create_dfs():
    global all_df, main_df
    for loc in locations:
        files = os.listdir(loc)
        for f in files:
            path = os.path.join(loc, f)
            if "all_features_details" in f:
                all_df = add_df(all_df, path)
            elif "details" in f:
                main_df = add_df(main_df, path)


def merge_all_main():
    global all_df, df2, main_df
    for index, row in main_df.iterrows():
        df2.loc[len(df2)] = {
            "dataset": row["dataset"],
            "target_size": row["target_size"],
            "algorithm": row["algorithm"],
            "time": row["time"],
            "oa": row["oa"],
            "k": row["k"]
        }
    for index, row in all_df.iterrows():
        df2.loc[len(df2)] = {
            "dataset": row["dataset"],
            "target_size": 0,
            "algorithm": "All Bands",
            "time": 0,
            "oa": row["oa"],
            "k": row["k"]
        }
    df2 = df2.sort_values(by=['algorithm', 'dataset', 'target_size'])


def rename_algorithms():
    global df2
    for key, value in utils.algorithm_map.items():
        df2.loc[df2["algorithm"] == key, "algorithm"] = value


def rename_datasets():
    global df2
    for key, value in utils.dataset_map.items():
        df2.loc[df2["dataset"] == key, "dataset"] = value


create_dfs()
#make_complete_main_df()
merge_all_main()
rename_algorithms()
rename_datasets()

df2.to_csv("../final_results/details.csv", index=False)

