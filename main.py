from task_runner import TaskRunner
import oak_plotter

if __name__ == '__main__':
    tag = "10"
    tasks = {
        "algorithms" : ["zhang_sm9"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [25, 30, 5, 10, 15, 20]
    }
    ev = TaskRunner(tasks,10,tag,skip_all_bands=True, verbose=True)
    summary, details = ev.evaluate()
    oak_plotter.plot_oak(source=summary)