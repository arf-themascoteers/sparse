import pandas as pd

df = pd.read_csv("../data/indian_pines.csv")

df = df.sample(frac=0.01)

df.to_csv("../data/indian_pines_min.csv")