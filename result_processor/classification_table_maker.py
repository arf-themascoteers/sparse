import pandas as pd


def get_metrics(df2, target_size=None):
    if target_size is not None:
        df3 = df2[df2["target_size"] == target_size]
    else:
        df3 = df2
    if len(df3) == 0:
        return None, None, None

    time_mean = df3["time"].mean()
    oa_mean = df3["oa"].mean()
    kappa_mean = df3["k"].mean()

    time_std = df3["time"].std()
    oa_std = df3["oa"].std()
    kappa_std = df3["k"].std()

    time_str = f"{time_mean:.2f}"
    oa_str = f"{oa_mean:.2f}"
    kappa_str = f"{kappa_mean:.2f}"

    if target_size is not None and len(df3)>1:
        time_str = f"{time_str}±{time_std:.2f}"
        oa_str = f"{oa_str}±{oa_std:.2f}"
        kappa_str = f"{kappa_str}±{kappa_std:.2f}"

    return time_str, oa_str, kappa_str


def get_metrics_for_6_targets(df, dataset, algorithm):
    df = df[(df["algorithm"] == algorithm) & (df["dataset"] == dataset)]
    if len(df) == 0:
        print(f"Algorithm {algorithm} not found for {dataset}")
        return None, None, None
    time_strs = []
    oa_strs = []
    kappa_strs = []

    if algorithm == "All Bands":
        time_str, oa_str, kappa_str = get_metrics(df)
        time_strs = time_strs + ([time_str] * 6)
        oa_strs = oa_strs + ([oa_str] * 6)
        kappa_strs = kappa_strs + ([kappa_str] * 6)
        return time_strs, oa_strs, kappa_strs

    for target_size in [5, 10, 15, 20, 25, 30]:
        time_str, oa_str, kappa_str = get_metrics(df, target_size)
        time_strs.append(time_str)
        oa_strs.append(oa_str)
        kappa_strs.append(kappa_str)

    return time_strs, oa_strs, kappa_strs


def create_table(dataset):
    df = pd.read_csv("../final_results/details.csv")
    algorithms = df["algorithm"].unique()

    time_df = pd.DataFrame(columns=["algorithm", "time_5", "time_10", "time_15", "time_15", "time_20", "time_25", "time_30"])
    oa_df = pd.DataFrame(columns=["algorithm", "oa_5", "oa_10", "oa_15", "oa_15", "oa_20", "oa_25", "oa_30"])
    kappa_df = pd.DataFrame(columns=["algorithm", "kappa_5", "kappa_10", "kappa_15", "kappa_15", "kappa_20", "kappa_25", "kappa_30"])

    for algorithm in algorithms:
        time_strs, oa_strs, kappa_strs = get_metrics_for_6_targets(df, dataset, algorithm)
        if time_strs is None:
            continue
        print(f"Algorithm: {algorithm}; {time_strs}")
        time_df.loc[len(time_df)] = {"algorithm": algorithm, "time_5": time_strs[0], "time_10": time_strs[1], "time_15": time_strs[2], "time_20": time_strs[3], "time_25": time_strs[4], "time_30": time_strs[5]}
        oa_df.loc[len(oa_df)] = {"algorithm": algorithm, "oa_5": oa_strs[0], "oa_10": oa_strs[1], "oa_15": oa_strs[2], "oa_20": oa_strs[3], "oa_25": oa_strs[4], "oa_30": oa_strs[5]}
        kappa_df.loc[len(kappa_df)] = {"algorithm": algorithm, "kappa_5": kappa_strs[0], "kappa_10": kappa_strs[1], "kappa_15": kappa_strs[2], "kappa_20": kappa_strs[3], "kappa_25": kappa_strs[4], "kappa_30": kappa_strs[5]}

    time_df.to_csv(f"../final_results/{dataset}_time.csv", index=False)
    oa_df.to_csv(f"../final_results/{dataset}_oa.csv", index=False)
    kappa_df.to_csv(f"../final_results/{dataset}_kappa.csv", index=False)


def create_tables():
    for dataset in ["GHISACONUS", "Indian Pines"]:
        create_table(dataset)


create_tables()
