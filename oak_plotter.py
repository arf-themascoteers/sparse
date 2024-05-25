import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_oak(source="final_results/final.csv", dest = "saved_figs/oak.png"):
    df_original = pd.read_csv(source)
    priority_order = ["ns","zhang","scnn","sfc","nsfc","zhangfc"]
    priority_order = ["zhang","zhangfc"]
    display_alg = priority_order
    df_original['algorithm'] = pd.Categorical(df_original['algorithm'], categories=priority_order, ordered=True)
    df_original = df_original.sort_values('algorithm')
    colors = ['#909c86', '#e389b9', '#269658', '#5c1ad6', '#f20a21', '#000000']
    markers = ['s', 'P', 'D', '^', 'o', '*', '.']
    labels = ["Overall Accuracy (OA)", "Average Accuracy (AA)", "Cohen's kappa ($\kappa$)"]
    min_lim = min(df_original["oa"].min(),df_original["aa"].min(),df_original["k"].min())-0.1
    max_lim = max(df_original["oa"].max(),df_original["aa"].max(),df_original["k"].max())+0.1
    datasets = ["indian_pines"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
    axes = np.reshape(axes, (3, -1))

    for metric_index,metric in enumerate(["oa", "aa", "k"]):
        for ds_index, dataset in enumerate(datasets):
            dataset_df = df_original[df_original["dataset"] == dataset].copy()
            for index, algorithm in enumerate(priority_order):
                alg_df = dataset_df[dataset_df["algorithm"] == algorithm]
                alg_df = alg_df.sort_values(by='target_size')
                if len(alg_df) == 0:
                    continue
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
                axes[metric_index, ds_index].set_title(f"{dataset}", fontsize=22, pad=20)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    plt.savefig(dest)
    plt.close(fig)

