import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils


main_df = pd.DataFrame()
all_df = pd.DataFrame()
root = "../results"
locations = [root]
algorithms = ["linspacer","bsnet","bsdr","rec"]
datasets = ["indian_pines","paviaU","salinasA"]
targets = [5,10,15,20,25,30]
df2 = pd.DataFrame(columns=["dataset","target_size","algorithm","time","oa","aa","k"])


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
            if "details" in f:
                continue
            if "bsdr-" in f:
                continue
            path = os.path.join(loc, f)
            if "all_features_summary" in f:
                all_df = add_df(all_df, path)
            else:
                main_df = add_df(main_df, path)


def make_complete_main_df():
    global df2, main_df
    for d in datasets:
        for t in targets:
            for a in algorithms:
                entries = main_df[(main_df["algorithm"] == a) & (main_df["dataset"] == d) & (main_df["target_size"] == t)]
                if len(entries) == 0:
                    print(f"Missing {d} {t} {a}")
                    df2.loc[len(df2)] = {
                        "dataset": d,
                        "target_size":t,
                        "algorithm": a,
                        "time": 100,
                        "oa": 0.2,
                        "aa": 0.2,
                        "k": 0.8
                    }
                elif len(entries) >= 1:
                    if len(entries) > 1:
                        print(f"Multiple {d} {t} {a} -- {len(entries)}: {list(entries['source'])}")
                    df2.loc[len(df2)] = {
                        "dataset": d,
                        "target_size":t,
                        "algorithm": a,
                        "time": entries.iloc[0]["time"],
                        "oa": entries.iloc[0]["oa"],
                        "aa": entries.iloc[0]["aa"],
                        "k": entries.iloc[0]["k"]
                    }


def add_all_in_main():
    global all_df, df2
    for d in datasets:
        for t in targets:
            entries = all_df[(all_df["dataset"] == d)]
            if len(entries) == 0:
                print(f"All Missing {d}")
                df2.loc[len(df2)] = {
                    "dataset": d,
                    "target_size": t,
                    "algorithm": "all_bands",
                    "time": 100,
                    "oa": 0.2,
                    "aa": 0.2,
                    "k": 0.8
                }
            elif len(entries) >= 1:
                if len(entries) > 1:
                    print(f"All Multiple {d} {t} -- {len(entries)}: {list(entries['source'])}")
                    pass
                df2.loc[len(df2)] = {
                    "dataset": d,
                    "target_size": t,
                    "algorithm": "all_bands",
                    "time": 0,
                    "oa": entries.iloc[0]["oa"],
                    "aa": entries.iloc[0]["aa"],
                    "k": entries.iloc[0]["k"]
                }


def rename_algorithms():
    global df2
    for key, value in utils.algorithm_map.items():
        df2.loc[df2["algorithm"] == key, "algorithm"] = value


def rename_datasets():
    global df2
    for key, value in utils.dataset_map.items():
        df2.loc[df2["dataset"] == key, "dataset"] = value


create_dfs()
make_complete_main_df()
add_all_in_main()
rename_algorithms()
rename_datasets()

datasets = list(utils.dataset_map.values())[0:3]
root = "../saved_figs"
df_original = df2
priority_order = ["Linspacer", "BS-Net-FC", "All Bands"]
display_alg = ['Linspacer', 'BS-Net-FC [28]', 'All Bands']
df_original['algorithm'] = pd.Categorical(df_original['algorithm'], categories=priority_order, ordered=True)
df_original = df_original.sort_values('algorithm')
colors = ['#909c86', '#e389b9', '#269658', '#5c1ad6', '#f20a21', '#000000']
markers = ['s', 'P', 'D', '^', 'o', '*', '.']
labels = ["OA", "AA", "$\kappa$"]
min_lim = min(df_original["oa"].min(),df_original["aa"].min(),df_original["k"].min())-0.1
max_lim = min(df_original["oa"].max(),df_original["aa"].max(), df_original["k"].max())+0.1


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

for metric_index,metric in enumerate(["oa", "aa", "k"]):
    for ds_index, dataset in enumerate(datasets):
        dataset_df = df_original[df_original["dataset"] == dataset].copy()
        for index, algorithm in enumerate(priority_order):
            alg_df = dataset_df[dataset_df["algorithm"] == algorithm]
            alg_df = alg_df.sort_values(by='target_size')
            if algorithm == "All Bands":
                axes[metric_index, ds_index].plot(alg_df['target_size'], alg_df[metric], label=algorithm,
                        linestyle='--', color=colors[index])
            else:
                axes[metric_index, ds_index].plot(alg_df['target_size'], alg_df[metric],
                        label=display_alg[index], marker=markers[index], color=colors[index],
                        fillstyle='none', markersize=10, linewidth=2
                        )

        axes[metric_index, ds_index].set_xlabel('Target size', fontsize=18)
        axes[metric_index, ds_index].set_ylabel(labels[metric_index], fontsize=18)
        axes[metric_index, ds_index].set_ylim(min_lim, max_lim)
        axes[metric_index, ds_index].tick_params(axis='both', which='major', labelsize=14)
        if ds_index == len(datasets)-1 and metric_index == 0:
            legend = axes[metric_index, ds_index].legend(title="Algorithms", loc='upper left', fontsize=18,bbox_to_anchor=(1.05, 1))
            legend.get_title().set_fontsize('18')
            legend.get_title().set_fontweight('bold')

        axes[metric_index, ds_index].grid(True, linestyle='--', alpha=0.6)
        if metric_index == 0:
            axes[metric_index, ds_index].set_title(f"{dataset} dataset", fontsize=22, pad=20)


path = os.path.join(root, f"oak.png")
plt.tight_layout()
fig.subplots_adjust(wspace=0.5)
plt.savefig(path)
plt.close(fig)

