import numpy as np
from promise import Promise

from convolutional_nn.src.layer.Layer import Layer


class OutputLayer(Layer):

    def __init__(self, filters, convo_filter_num):
        Layer.__init__(self, filters)
        self.convo_filter_num = convo_filter_num

    def set_target(self, y):
        self.y = OutputLayer.one_hot_encoding(y)
        w = self.y
        i = self.convo_filter_num
        while i > 1:
            self.y = np.append(self.y, w, axis=0)
            i-=1

    @staticmethod
    def one_hot_encoding(y):
        y_encoded = np.zeros([y.shape[0], 3])
        y_encoded[np.arange(y.shape[0]), y] = 1

        return y_encoded

    def f(self, input):
        self.input = np.insert(input, input.shape[1], 1, axis=1)
        dot = np.dot(self.input, self.weights)
        self.output = OutputLayer.cross_entrophy_loss(OutputLayer.softmax(dot), self.y)
        return Promise.resolve(self.output)

    @staticmethod
    def softmax(z):
        z -= np.max(z) # Dividing large number is numerically unstable, this would help with the stability
        num = np.exp(z)
        denom = np.sum(np.exp(z), axis=1)[:,None]
        div = num/denom
        return div
        #if the z is num of samples x num of outputs
        #np.transpose(np.transpose(np.exp(z)) / np.sum(np.exp(z), axis=1))

    @staticmethod
    def cross_entrophy_loss(z, y):
        normalizing_factor = 0.01
        z_log = np.log10(z)
        z_dot = -y * z_log
        z_loss = normalizing_factor * np.sum(z_dot)
        return z_loss
