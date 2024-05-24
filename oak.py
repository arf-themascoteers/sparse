import oak_csv_creator
import oak_plotter

csv = oak_csv_creator.create_csv(filter=["4"])
oak_plotter.plot_oak()