import numpy as np


class Utils:

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))