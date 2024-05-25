from task_runner import TaskRunner
import oak

if __name__ == '__main__':
    result_file = "10.csv"
    tasks = {
        "algorithms" : ["zhangfc","zhang"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [25, 30, 5, 10, 15, 20]
    }
    ev = TaskRunner(tasks,1,10,"10.csv",skip_all_bands=True)
    ev.evaluate()