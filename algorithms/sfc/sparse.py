import torch
import torch.nn as nn


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, X):
        threshold = 100/128
        k = torch.tensor(threshold).to(X.device)
        X = torch.where(X < k, torch.tanh(X)*.3, X)
        return X


if __name__ == "__main__":
    sparse = Sparse()
    X = torch.linspace(0,1,1000).reshape(-1,1)
    y = sparse(X)
    import matplotlib.pyplot as plt
    plt.plot(X,y)
    plt.show()

