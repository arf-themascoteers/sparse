import pandas as pd
import os
import plotters.utils as utils
import plotly.graph_objects as go

main_df = pd.DataFrame()
root = "../results"
algorithms = ["bsdr","bsdr1","bsdr2","bsdr3","bsdr4","bsdr5"]
datasets = ["ghisaconus","indian_pines","lucas"]
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
    global main_df
    main_df = add_df(main_df, "../results/bsdr.csv")


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


def plot_me():
    df_original = pd.read_csv("../final_results/bsdrs.csv")
    priority_order = ["bsdr","bsdr1","bsdr2","bsdr3","bsdr4","bsdr5"]
    df_original['algorithm'] = pd.Categorical(df_original['algorithm'], categories=priority_order, ordered=True)
    df_original = df_original.sort_values('algorithm')
    colors = ['#909c86', '#d2ff41', '#269658', '#5c1ad6', '#f20a21', '#000000']
    markers = ['star-open', 'pentagon-open', 'circle-open', 'hash-open', 'triangle-up-open', 'diamond-open',
               'square-open', None]
    datasets = ["GHISACONUS", "Indian Pines"]
    for metric in ["time", "oa", "k"]:
        for ds_index, dataset in enumerate(datasets):
            fig = go.Figure()
            dataset_df = df_original[df_original["dataset"] == dataset].copy()
            #dataset_df["time"] = dataset_df["time"].apply(lambda x: np.log10(x) if x != 0 else 0)
            algorithms = dataset_df["algorithm"].unique()
            for index, algorithm in enumerate(algorithms):
                alg_df = dataset_df[dataset_df["algorithm"] == algorithm]
                alg_df = alg_df.sort_values(by='target_size')
                line = dict()
                mode = 'lines+markers'
                if algorithm == "All Bands":
                    if metric == "time":
                        continue
                    else:
                        line["dash"] = "dash"
                        mode = 'lines'
                # elif algorithm == "BSDR":
                #     line["width"] = 3
                fig.add_trace(
                    go.Scatter(
                        x=alg_df['target_size'],
                        y=alg_df[metric], mode=mode,
                        name=algorithm, marker=dict(color=colors[index], symbol=markers[index], size=10),
                        line=line
                    )
                )

            fig.update_layout({
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'title_x': 0.5
            })

            fig.update_layout(
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    gridwidth=1
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    gridwidth=1
                )
            )

            fig.update_layout(
                # title='Performance Metrics by Algorithm and Dataset',
                xaxis_title="Target size",
                yaxis_title=utils.metric_map[metric][dataset],
                legend_title="Algorithm"
            )

            #fig.update_layout(showlegend=False)

            fig.update_layout(
                font=dict(size=17),
            )
            fig.write_image(f"../saved_figs/bsdrs_{dataset}_{metric}.png", scale=5)

create_dfs()
make_complete_main_df()
rename_algorithms()
rename_datasets()
plot_me()

df2.to_csv("../final_results/bsdrs.csv", index=False)

