import torch
import matplotlib.pyplot as plt

X = torch.linspace(1,128,1000)
k = torch.tensor(100)
Y = torch.where(X < k, X/20, X)

plt.plot(X,Y)
plt.show()
