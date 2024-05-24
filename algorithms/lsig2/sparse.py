import torch
import torch.nn as nn


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.a = nn.Parameter(torch.tensor(10.0))
        self.k = nn.Parameter(torch.tensor(100/128))

    def forward(self, X):
        return torch.where(X > self.k, X, 1 / (1 + torch.exp(-self.a * (X - self.k))))


if __name__ == "__main__":
    s = Sparse()
    X = torch.linspace(0,1,1000).reshape(-1,1)
    y = s(X)
    import matplotlib.pyplot as plt
    plt.plot(X,y)
    plt.show()
