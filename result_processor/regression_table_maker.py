import pandas as pd


def get_metrics(df2, target_size=None):
    if target_size is not None:
        df3 = df2[df2["target_size"] == target_size]
    else:
        df3 = df2
    if len(df3) == 0:
        return None, None, None

    time_mean = df3["time"].mean()
    r2_mean = df3["oa"].mean()
    rmse_mean = df3["k"].mean()

    time_std = df3["time"].std()
    r2_std = df3["oa"].std()
    rmse_std = df3["k"].std()

    time_str = f"{time_mean:.2f}"
    r2_str = f"{r2_mean:.2f}"
    rmse_str = f"{rmse_mean:.2f}"

    if target_size is not None and len(df3)>1:
        time_str = f"{time_str}±{time_std:.2f}"
        r2_str = f"{r2_str}±{r2_std:.2f}"
        rmse_str = f"{rmse_str}±{rmse_std:.2f}"

    return time_str, r2_str, rmse_str


def get_metrics_for_6_targets(df, algorithm):
    df = df[(df["algorithm"] == algorithm) & (df["dataset"] == "LUCAS")]
    if len(df) == 0:
        print(f"Algorithm {algorithm} not found for LUCAS")
        return None, None, None
    time_strs = []
    r2_strs = []
    rmse_strs = []

    if algorithm == "All Bands":
        time_str, r2_str, rmse_str = get_metrics(df)
        print(time_str, r2_str, rmse_str)
        time_strs = time_strs + ([time_str] * 6)
        r2_strs = r2_strs + ([r2_str] * 6)
        rmse_strs = rmse_strs + ([rmse_str] * 6)
        return time_strs, r2_strs, rmse_strs

    for target_size in [5, 10, 15, 20, 25, 30]:
        time_str, r2_str, rmse_str = get_metrics(df, target_size)
        time_strs.append(time_str)
        r2_strs.append(r2_str)
        rmse_strs.append(rmse_str)

    return time_strs, r2_strs, rmse_strs


def create_table():
    df = pd.read_csv("../final_results/details.csv")
    algorithms = df["algorithm"].unique()

    time_df = pd.DataFrame(columns=["algorithm", "time_5", "time_10", "time_15", "time_15", "time_20", "time_25", "time_30"])
    r2_df = pd.DataFrame(columns=["algorithm", "r2_5", "r2_10", "r2_15", "r2_15", "r2_20", "r2_25", "r2_30"])
    rmse_df = pd.DataFrame(columns=["algorithm", "rmse_5", "rmse_10", "rmse_15", "rmse_15", "rmse_20", "rmse_25", "rmse_30"])

    for algorithm in algorithms:
        time_strs, r2_strs, rmse_strs = get_metrics_for_6_targets(df, algorithm)
        if time_strs is None:
            continue
        print(f"Algorithm: {algorithm}; {time_strs}")
        time_df.loc[len(time_df)] = {"algorithm": algorithm, "time_5": time_strs[0], "time_10": time_strs[1], "time_15": time_strs[2], "time_20": time_strs[3], "time_25": time_strs[4], "time_30": time_strs[5]}
        r2_df.loc[len(r2_df)] = {"algorithm": algorithm, "r2_5": r2_strs[0], "r2_10": r2_strs[1], "r2_15": r2_strs[2], "r2_20": r2_strs[3], "r2_25": r2_strs[4], "r2_30": r2_strs[5]}
        rmse_df.loc[len(rmse_df)] = {"algorithm": algorithm, "rmse_5": rmse_strs[0], "rmse_10": rmse_strs[1], "rmse_15": rmse_strs[2], "rmse_20": rmse_strs[3], "rmse_25": rmse_strs[4], "rmse_30": rmse_strs[5]}

    time_df.to_csv("../final_results/LUCAS_time.csv", index=False)
    r2_df.to_csv("../final_results/LUCAS_r2.csv", index=False)
    rmse_df.to_csv("../final_results/LUCAS_rmse.csv", index=False)

create_table()
