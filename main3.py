from task_runner import TaskRunner
import oak_plotter

if __name__ == '__main__':
    tag = "f1"
    tasks = {
        "algorithms" : ["zhang","lw","lw_relu"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = TaskRunner(tasks,1,tag,skip_all_bands=False)
    summary, details = ev.evaluate()
