from algorithms.algorithm_bsnet import AlgorithmBSNet
from algorithms.algorithm_zhang import AlgorithmZhang
from algorithms.algorithm_bsdr import AlgorithmBSDR
from algorithms.algorithm_bsdr500 import AlgorithmBSDR500
from algorithms.algorithm_bsdr3000 import AlgorithmBSDR3000
from algorithms.algorithm_bsdr4000 import AlgorithmBSDR4000
from algorithms.algorithm_bsdr6000 import AlgorithmBSDR6000
from algorithms.algorithm_linspacer import AlgorithmLinspacer


class AlgorithmCreator:
    @staticmethod
    def create(name, target_size, splits, repeat, fold, verbose=True):

        algorithms = {
            "bsnet" : AlgorithmBSNet,
            "zhang" : AlgorithmZhang,
            "bsdr" : AlgorithmBSDR,
            "bsdr500" : AlgorithmBSDR500,
            "bsdr3000" : AlgorithmBSDR3000,
            "bsdr4000" : AlgorithmBSDR4000,
            "bsdr6000" : AlgorithmBSDR6000,
            "linspacer": AlgorithmLinspacer,
        }

        if name not in algorithms:
            raise KeyError(f"No algorithm named {name} exists")

        if name in ["bsdr", "bsdr500", "bsdr3000","bsdr4000", "bsdr6000"]:
            return algorithms[name](target_size, splits, repeat, fold, verbose=verbose)

        return algorithms[name](target_size, splits)