import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("combo_2b_filtered.csv")
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['band1'], df['band2'], c=df['score'], cmap='viridis', s=15)  # Decreased marker size
cb = plt.colorbar(scatter, label='Score', orientation='vertical')
cb.set_label('')
cb.ax.set_title('$R^2$', loc='left', pad=10, fontsize=16)
plt.xlabel('Target Index 1', fontsize=16)
plt.ylabel('Target Index 2', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([1,66])
plt.ylim([1,66])
row = df[(df['band1']==58) & (df['band2']==59)]
value = round(row.iloc[0]['score'],2)
plt.scatter(58, 59, color='red', s=50, label=f'Global Maximum: {value:0.2f}')

row = df[(df['band1']==7) & (df['band2']==14)]
value = round(row.iloc[0]['score'],2)
plt.scatter(7, 14, color='magenta', s=50, label=f'Local Maximum 1: {value:0.2f}')

row = df[(df['band1']==47) & (df['band2']==54)]
value = round(row.iloc[0]['score'],2)
plt.scatter(47, 54, color='coral', s=50, label=f'Local Maximum 2: {value:0.2f}')

plt.legend(loc='lower right', markerscale=1, fontsize=16, scatterpoints=1)
plt.savefig("fig1.png")