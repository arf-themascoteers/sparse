import pandas as pd
import os
import plotters.utils as utils


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


def create_dfs(all_df, main_df, locations):
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
    return all_df, main_df


def make_complete_main_df(main_df, datasets, targets, algorithms, final_df):
    for d in datasets:
        for t in targets:
            for a in algorithms:
                entries = main_df[(main_df["algorithm"] == a) & (main_df["dataset"] == d) & (main_df["target_size"] == t)]
                if len(entries) == 0:
                    print(f"Missing {d} {t} {a}")
                    final_df.loc[len(final_df)] = {
                        "dataset": d,
                        "target_size":t,
                        "algorithm": a,
                        "time": 100,
                        "oa": 0,
                        "aa": 0,
                        "k": 0
                    }
                elif len(entries) >= 1:
                    if len(entries) > 1:
                        print(f"Multiple {d} {t} {a} -- {len(entries)}: {list(entries['source'])}")
                    final_df.loc[len(final_df)] = {
                        "dataset": d,
                        "target_size":t,
                        "algorithm": a,
                        "time": entries.iloc[0]["time"],
                        "oa": entries.iloc[0]["oa"],
                        "aa": entries.iloc[0]["aa"],
                        "k": entries.iloc[0]["k"]
                    }
    return final_df


def add_all_in_main(all_df, datasets, targets, final_df):
    if len(all_df) == 0:
        return final_df
    for d in datasets:
        for t in targets:
            entries = all_df[(all_df["dataset"] == d)]
            if len(entries) == 0:
                print(f"All Missing {d}")
                final_df.loc[len(final_df)] = {
                    "dataset": d,
                    "target_size": t,
                    "algorithm": "all_bands",
                    "time": 100,
                    "oa": 0,
                    "aa": 0,
                    "k": 0
                }
            elif len(entries) >= 1:
                if len(entries) > 1:
                    print(f"All Multiple {d} {t} -- {len(entries)}: {list(entries['source'])}")
                    pass
                final_df.loc[len(final_df)] = {
                    "dataset": d,
                    "target_size": t,
                    "algorithm": "all_bands",
                    "time": 0,
                    "oa": entries.iloc[0]["oa"],
                    "aa": entries.iloc[0]["aa"],
                    "k": entries.iloc[0]["k"]
                }
    return final_df


def create_csv(name="final",filter=None):
    if filter is None:
        filter = os.listdir("saved_results")
    main_df = pd.DataFrame()
    all_df = pd.DataFrame()
    locations = [os.path.join("saved_results", subfolder) for subfolder in filter]
    locations = [loc for loc in locations if os.path.exists(loc)]
    algorithms = ["linspacer","bsdr","bsdr500","bsdr3000"]
    datasets = ["indian_pines"]
    targets = [5,10,15,20,25,30]
    all_df, main_df = create_dfs(all_df, main_df, locations)
    final_df = pd.DataFrame(columns=["dataset","target_size","algorithm","time","oa","aa","k"])
    final_df = make_complete_main_df(main_df, datasets, targets, algorithms, final_df)
    final_df = add_all_in_main(all_df, datasets, targets, final_df)
    dest = f"final_results/{name}.csv"
    final_df.to_csv(dest, index=False)
    return dest

