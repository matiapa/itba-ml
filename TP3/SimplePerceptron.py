import numpy as np

class SimplePerceptron():
    
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.min_weights = np.zeros(input_size + 1)

    def predict(self, x, bias):
        x = np.insert(x, 0, bias) # add bias
        u = self._exitation(x, weights=self.min_weights)
        return 1 if u >= 0 else -1

    def train(self, training_inputs, labels, bias):
        error = 1
        min_error = 2 * len(training_inputs)
        for inputs, label in zip(training_inputs, labels):
            inputs = np.insert(inputs, 0, bias) # add bias
            exitation = self._exitation(inputs)
            activation = self._activation(exitation)
            delta_w = self._new_weight(label, activation, inputs)
            self.weights += delta_w

            error = self.error(training_inputs, labels, bias)
            if (error < min_error):
                min_error = error
                self.min_weights = self.weights


    def _new_weight(self, label, prediction, inputs):
        return self.learning_rate * (label - prediction) * inputs

    def _exitation(self, inputs, weights=None):
        if weights is None:
            weights = self.weights
        return np.dot(inputs, weights)

    def _activation(self, value):
        return 1 if value >= 0 else -1

    def error(self, inputs, labels, bias):
        error = 0
        for inputs, label in zip(inputs, labels):
            inputs = np.insert(inputs, 0, bias)
            exitation = self._exitation(inputs)
            activation = self._activation(exitation)
            error += (label - activation) ** 2
        return error
