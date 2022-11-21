from typing import List

from numpy import linalg, ndarray

class Neuron:

    def __init__(self, weights: ndarray) -> None:
        self.weights = weights

    def distance(self, input: ndarray):
        if len(input) != len(self.weights):
            raise Exception("Input dimension doesn't match weights dimension")
        return linalg.norm(input - self.weights)