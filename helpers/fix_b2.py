import pandas as pd

df = pd.read_csv("b2.csv")
df = df[df["band1"] < df["band2"]]
df.to_csv("b22.csv", index=False)