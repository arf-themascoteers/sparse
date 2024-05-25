import oak_csv_creator
import oak_plotter

csv = oak_csv_creator.create_csv(filter=["6"])
oak_plotter.plot_oak(source=csv)