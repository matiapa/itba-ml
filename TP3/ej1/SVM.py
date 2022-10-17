import numpy as np
import pandas as pd

class SVM:
    def __init__(self, C, input_size, b):
        self.C = C
        self.k_w = 0.01
        self.k_b = 0.01
        self.weights = np.zeros(input_size)
        self.b = b

    def train(self, training_inputs, labels, epochs=10):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                t = label * (np.dot(self.weights, inputs) + self.b)
                if t < 1:
                    self.weights = self.weights - self.k_w * (self.weights + self.C * (np.sum(-1 * training_inputs * labels[:, np.newaxis], axis=0)))
                    self.b = self.b - self.k_b * self.C * np.sum(-1 * labels)
                else:
                    self.weights = self.weights - self.k_w * self.weights

    def evaluate(self, inputs):
        return 1 if np.dot(self.weights, inputs) + self.b >= 0 else -1
