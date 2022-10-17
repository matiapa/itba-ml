import numpy as np
import pandas as pd

class SVM:
    def __init__(self, C, input_size, b):
        self.C = C
        self.k_w = 0.01
        self.k_b = 0.01
        self.weight = np.zeros(input_size)
        self.b = b

    def train(self, training_inputs, labels):
        
        for inputs, label in zip(training_inputs, labels):
            t = label * (np.dot(self.weight, inputs) + self.b)
            if t < 1:
                self.weight -= self.k_w * (self.weight + self.C * (-1 * np.sum(training_inputs * labels[:, np.newaxis], axis=0)))
                self.b -= self.k_b * self.C * np.sum(-1*labels)
            else:
                self.weight -= self.k_w * self.weight

    def evaluate(self, inputs):
        return 1 if np.dot(self.weight, inputs) + self.b >= 0 else -1



