import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate x values
x_values = np.linspace(-7, 7, 200)
y_values = sigmoid(x_values)
fig, ax = plt.subplots()
ax.plot(x_values, y_values, marker='o',markersize=5)
ax.axhline(0, color='black', linewidth=4, linestyle='-', label='y=0')
ax.axvline(0, color='black', linewidth=4, linestyle='-', label='x=0')  # Add a vertical line at x=0
ax.axis('off')

# circle = plt.Circle((0, 0), 0.4, transform=ax.transAxes, color='black', fill=False, alpha=1)
# ax.add_patch(circle)
# plt.xlim(-4, 4)
# plt.ylim(0, 1)
# ax.set_aspect('equal', adjustable='box')
plt.show()