import matplotlib.pyplot as plt

x = range(1, 7)
y1 = [3, 7, 2, 5, 8, 4]
y2 = [5, 4, 6, 2, 7, 3]

fig, axs = plt.subplots(3, 2, figsize=(10, 12))

# Plot data and set titles
for i in range(3):
    for j in range(2):
        axs[i, j].plot(x, y1, marker='o', label='Data 1')
        axs[i, j].plot(x, y2, marker='o', label='Data 2')
        axs[i, j].legend()
        axs[i, j].set_title(f'Data {i * 2 + j + 1}')

# Remove borders of all corners for all plots
for ax in axs.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.show()
