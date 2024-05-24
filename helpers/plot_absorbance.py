import pandas as pd
import matplotlib.pyplot as plt

lucas = pd.read_csv(r"E:\data\lucas\absorbance.csv")
data = lucas.iloc[120,13:].to_numpy()
x = list(range(data.shape[0]))
plt.plot(x,data)
plt.show()