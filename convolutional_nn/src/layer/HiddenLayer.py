import numpy as np
from promise import Promise

from convolutional_nn.src.layer.Layer import Layer

step_size = 1e-0
class HiddenLayer(Layer):

    def __init__(self, filters):
        Layer.__init__(self, filters)

    def f(self, input):
        self.input = np.insert(input, input.shape[1], 1, axis=1)
        dot = np.dot(self.input, self.weights)
        self.output = HiddenLayer.tanh(dot)
        return Promise.resolve(self.output)

    def b(self, gradient):
        gradient *=  2.0 * (self.output * ( 1.0 - self.output))
        dW = np.dot(self.input.T, gradient)
        self.d_input = np.dot(gradient, self.weights[0:self.weights.shape[0] - 1].T)
        self.update(dW, step_size)
        return Promise.resolve(self.d_input)

    @staticmethod
    def tanh(x):
        t = 2.0 * HiddenLayer.sigmoid(2.0 * x) - 1.0
        return t

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -10.0, 10.0)
        s = 1.0 / (1.0 + np.exp(-x))
        return s
