from task_runner import TaskRunner
import oak_plotter

if __name__ == '__main__':
    tag = "X"
    tasks = {
        "algorithms" : ["linspacer", "zhang_min"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [5, 10, 15]
    }
    ev = TaskRunner(tasks,1,tag,skip_all_bands=True, verbose=True)
    summary, details = ev.evaluate()
