import pandas as pd
import pywt


def create_lucas_full():
    lucas = pd.read_csv("../data/lucas/reflectance.csv")
    bands = list(lucas.columns)
    bands = bands[13:]
    all_columns = ["oc"]+bands
    lucas_full = lucas[all_columns]
    lucas_full.to_csv("../data/lucas_full.csv", index=False)


def create_lucas_min():
    lucas_full = pd.read_csv("../data/lucas_full.csv")
    lucas_min = lucas_full.sample(frac=0.04, random_state=40)
    lucas_min.to_csv("../data/lucas_min.csv", index=False)


def create_lucas_downsampled():
    lucas_full = pd.read_csv("../data/lucas_full.csv")
    reduced_columns = ["oc"] + [str(i) for i in range(66)]
    lucas_down = pd.DataFrame(columns=reduced_columns)
    for index, row in lucas_full.iterrows():
        signal = row.iloc[1:]
        short_signal, _, _, _, _, _, _ = pywt.wavedec(signal, 'db1', level=6)
        lucas_down.loc[len(lucas_down)] = [row.iloc[0]] + list(short_signal)
    lucas_down.to_csv(f"../data/lucas_downsampled.csv", index=False)


def create_lucas_downsampled_min():
    lucas_down = pd.read_csv("../data/lucas_downsampled.csv")
    lucas_down = lucas_down.sample(frac=0.04, random_state=40)
    lucas_down.to_csv("../data/lucas_downsampled_min.csv", index=False)


def create_lucas_skipped():
    lucas_full = pd.read_csv("../data/lucas_full.csv")
    reduced_columns = ["oc"] + [str(i) for i in range(66)]
    selected_indices = list(range(1,4201,64))
    lucas_skipped = pd.DataFrame(columns=reduced_columns)
    for index, row in lucas_full.iterrows():
        signal = row.iloc[selected_indices]
        lucas_skipped.loc[len(lucas_skipped)] = [row.iloc[0]] + list(signal)
    lucas_skipped.to_csv(f"../data/lucas_skipped.csv", index=False)

create_lucas_full()
create_lucas_downsampled()
create_lucas_downsampled_min()

print("Done all")