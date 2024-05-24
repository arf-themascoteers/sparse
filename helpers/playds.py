import pandas as pd
import os

for file in os.listdir("../data"):
    if file.endswith(".csv"):
        df = pd.read_csv(f"../data/{file}")
        uns = df["class"].unique().astype(int)
        uns = sorted(uns)
        print(file, len(uns), uns)
        counts = df["class"].value_counts()
        print(counts)
