import pandas as pd
import matplotlib.pyplot as plt


def plot_brazilian():
    df = pd.read_csv(r"data/brazilian.csv")
    data = df.iloc[0,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()


def plot_lucas_absorbance():
    df = pd.read_csv(r"../data/lucas/absorbance.csv")
    data = df.iloc[0,13:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,13:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()


def plot_lucas_reflectance():
    df = pd.read_csv(r"../data/lucas/reflectance.csv")
    data = df.iloc[0,13:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,13:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()


def plot_lucas_full():
    df = pd.read_csv(r"../data/lucas_full.csv")
    data = df.iloc[0,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()


def plot_lucas_min():
    df = pd.read_csv(r"data/lucas_min.csv")
    data = df.iloc[0,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()


def plot_lucas_down_min():
    df = pd.read_csv(r"data/lucas_down_min.csv")
    data = df.iloc[0,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()


def plot_lucas_down():
    df = pd.read_csv(r"data/lucas_down.csv")
    data = df.iloc[0,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()


def plot_lucas_skipped():
    df = pd.read_csv(r"data/lucas_skipped.csv")
    data = df.iloc[0,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()


def plot_lucas_skipped_min():
    df = pd.read_csv(r"data/lucas_skipped_min.csv")
    data = df.iloc[0,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()


def plot_demmin():
    df = pd.read_csv(r"data/demmin.csv")
    data = df.iloc[0,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[20,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

if __name__ == "__main__":
    # plot_brazilian()
    # plot_lucas_absorbance()
    # plot_lucas_reflectance()
    # plot_lucas_full()
    plot_lucas_min()
    plot_lucas_down_min()
    # plot_lucas_down()
    # plot_lucas_skipped()
    plot_lucas_skipped_min()
    #plot_demmin()
