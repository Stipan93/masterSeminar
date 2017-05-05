import numpy as np
import math


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


class AveragedPerceptron:

    def __init__(self, D, activation_function=_sigmoid, shuffle=False):
        self.weights = np.zeros(D)

    def train(self, max_iter=100):
        pass

    def predict(self, features):
        pass
