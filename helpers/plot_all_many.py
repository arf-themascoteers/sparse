import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_demmin():
    df = pd.read_csv(r"data/demmin.csv")
    df = df.sample(frac=0.1, random_state=40)
    data = df.iloc[:,1:].to_numpy()
    x = list(range(data.shape[1]))
    for i in range(data.shape[0]):
        plt.plot(x,data[i])
    data = np.mean(data, axis=0)
    plt.plot(x, data, linestyle='--')
    plt.title("Demmin")
    plt.show()


def plot_lucas():
    df = pd.read_csv(r"data/lucas_min.csv")
    df = df.sample(frac=0.01, random_state=40)
    data = df.iloc[:,1:].to_numpy()
    x = list(range(data.shape[1]))
    for i in range(data.shape[0]):
        plt.plot(x,data[i])
    data = np.mean(data, axis=0)
    plt.plot(x, data, linestyle='--')
    plt.title("LUCAS")
    plt.show()


def plot_brazilian():
    df = pd.read_csv(r"data/brazilian.csv")
    df = df.sample(frac=0.05, random_state=40)
    data = df.iloc[:,1:].to_numpy()
    x = list(range(data.shape[1]))
    for i in range(data.shape[0]):
        plt.plot(x,data[i])
    data = np.mean(data, axis=0)
    plt.plot(x, data, linestyle='--')
    plt.title("Brazilian")
    plt.show()


if __name__ == "__main__":
    plot_brazilian()
