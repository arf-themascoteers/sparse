from task_runner import TaskRunner
import oak_plotter

if __name__ == '__main__':
    tag = "10"
    tasks = {
        "algorithms" : ["linspacer","zhang"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [25, 30, 5, 10, 15, 20]
    }
    ev = TaskRunner(tasks,1,10,tag,skip_all_bands=True)
    file = ev.evaluate()
    oak_plotter.plot_oak(source=file)