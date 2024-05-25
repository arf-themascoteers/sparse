from algorithms.algorithm_zhang import AlgorithmZhang
from algorithms.algorithm_linspacer import AlgorithmLinspacer


class AlgorithmCreator:
    @staticmethod
    def create(name, target_size, splits, repeat, fold, verbose=False):

        algorithms = {
            "zhang" : AlgorithmZhang,
            "linspacer": AlgorithmLinspacer,
        }

        if name not in algorithms:
            raise KeyError(f"No algorithm named {name} exists")

        return algorithms[name](target_size, splits,  repeat, fold, verbose=False)