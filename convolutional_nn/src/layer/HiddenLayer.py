import numpy as np
from promise import Promise

from convolutional_nn.src.layer.Layer import Layer


class HiddenLayer(Layer):

    def __init__(self, filters):
        Layer.__init__(self, filters)

    def f(self, input):
        self.input = HiddenLayer.flatten(input)
        dot = np.dot(self.input, self.weights)
        self.output = HiddenLayer.tanh(dot)
        return Promise.resolve(self.output)

    @staticmethod
    def tanh(x):
        t = 2.0 * HiddenLayer.sigmoid(2.0 * x) - 1.0
        return t

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -10.0, 10.0)
        s = 1.0 / (1.0 + np.exp(-x))
        return s

    @staticmethod
    def flatten(input):
        m = np.reshape(input, (input.shape[0], input.shape[1] * input.shape[2]))
        return np.insert(m, m.shape[1], 1, axis=1)
