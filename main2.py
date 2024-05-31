from task_runner import TaskRunner
import oak_plotter

if __name__ == '__main__':
    tag = "par2"
    tasks = {
        "algorithms" : ["zhang_fc_par"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = TaskRunner(tasks,1,tag,skip_all_bands=True, verbose=True)
    summary, details = ev.evaluate()
    #oak_plotter.plot_oak(source=summary)