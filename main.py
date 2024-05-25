from task_runner import TaskRunner

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["zhangfc","zhang"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [25, 30, 5, 10, 15, 20]
    }
    ev = TaskRunner(tasks,1,10,"9.csv",skip_all_bands=True)
    ev.evaluate()