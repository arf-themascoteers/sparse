import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = 1 - (0.1 * x + np.random.normal(0, 0.1, 100)) ** 2  # Simulated R^2 values
y2 = np.exp(-0.1 * x) + np.random.normal(0, 0.01, 100)    # Simulated Normalized RMSE values

fig, ax1 = plt.subplots()

# Plotting R^2 on the left y-axis
ax1.set_xlabel('X (units)')
ax1.set_ylabel('R^2', color='tab:blue')
ax1.plot(x, y1, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Creating a second y-axis for the normalized RMSE
ax2 = ax1.twinx()
ax2.set_ylabel('Normalized RMSE', color='tab:red')
ax2.plot(x, y2, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Title and show
plt.title('Dual Y-Axis Plot: R^2 and Normalized RMSE')
plt.show()
