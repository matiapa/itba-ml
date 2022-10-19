import numpy as np
import pandas as pd

class SVM:
    def __init__(self, C, input_size, b, Aw, Ab, k):
        self.C = C
        self.b = b
        self.Aw = Aw
        self.Ab = Ab
        self.k = k
        self.k_w = 0.01
        self.k_b = 0.01


    def train(self, training_inputs, labels, iter=1000, min_error=0.01, write_errors=False):
        self.weights = np.random.rand(training_inputs.shape[1])
        # create file
        if write_errors:
            f = open("error.csv", "w")
            f.write("error,iteration\n")

        for i in range(iter):
            idx = np.random.randint(0, len(training_inputs))
            x = training_inputs[idx]
            y = labels[idx]

            # update kw and kb values based on a exponential decay
            self.k_w = self._exp(i, self.Aw, self.k)
            self.k_b = self._exp(i, self.Ab, self.k)

            t = y * (np.dot(self.weights, x) + self.b)
            if t < 1:
                self.weights -= self.k_w * (self.weights + self.C * np.dot(y, x)*(-1))
                self.b -= self.k_b * self.C * y * (-1)
            else:
                self.weights = self.weights + (-self.k_w * self.weights)

            error = self.error(training_inputs, labels)
            if write_errors:
                f.write("{},{}\n".format(error, i))
            if error < min_error:
                break

            

    def evaluate(self, inputs):
        return np.sign(np.dot(inputs, self.weights) + self.b)

    def _exp(self, iter, A, k):
        return A*np.exp(-k*iter)

    def error(self, inputs, labels):
        error = 0
        for i in range(len(inputs)):
            error += self.evaluate(inputs[i]) != labels[i]
        return error / len(inputs)
