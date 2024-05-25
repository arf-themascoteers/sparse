import oak_csv_creator
import oak_plotter


def plot_oak(loc):
    csv = oak_csv_creator.create_csv(loc)
    oak_plotter.plot_oak(source=csv)
