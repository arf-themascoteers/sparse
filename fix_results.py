import pandas as pd

# df = pd.read_csv("saved_results/1/1.csv")
# print(len(df))
# df = df[df["algorithm"] != "bsdr"]
# print(len(df))
# df.to_csv("saved_results/1/1.csv", index=False)

# df = pd.read_csv("saved_results/1/details_1.csv")
# print(len(df))
# df = df[df["algorithm"] != "bsdr"]
# print(len(df))
# df.to_csv("saved_results/1/details_1.csv", index=False)

# df = pd.read_csv("saved_results/2/2.csv")
# print(len(df))
# df = df[df["algorithm"] != "bsdr"]
# print(len(df))
# df.to_csv("saved_results/2/2.csv", index=False)

df = pd.read_csv("saved_results/2/details_2.csv")
print(len(df))
df = df[df["algorithm"] != "bsdr"]
print(len(df))
df.to_csv("saved_results/2/details_2.csv", index=False)