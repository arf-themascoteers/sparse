import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

fig, axs = plt.subplots(1, 3)

lucas = pd.read_csv("data/ghisaconus.csv")
data = lucas.iloc[120,1:].to_numpy()
x = list(range(data.shape[0]))
axs[0].plot(x, data)
axs[0].set_title('GHISACONUS')

lucas = pd.read_csv("data/indian_pines.csv")
data = lucas.iloc[120,1:].to_numpy()
x = list(range(data.shape[0]))
axs[1].plot(x, data)
axs[1].set_title('Indian Pines')

lucas = pd.read_csv("data/LUCAS.csv")
data = lucas.iloc[120,1:].to_numpy()
x = list(range(data.shape[0]))
axs[2].plot(x, data)
axs[2].set_title('LUCAS')

plt.tight_layout()
plt.show()



