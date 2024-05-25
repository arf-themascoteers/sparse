from task_runner import TaskRunner
import oak_plotter

if __name__ == '__main__':
    result_file = "10.csv"
    tasks = {
        "algorithms" : ["linspacer","zhang"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [25, 30, 5, 10, 15, 20]
    }
    ev = TaskRunner(tasks,1,10,result_file,skip_all_bands=True)
    ev.evaluate()
    oak_plotter.plot_oak(source=result_file)