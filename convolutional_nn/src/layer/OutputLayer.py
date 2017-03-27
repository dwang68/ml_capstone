import numpy as np
from promise import Promise

from convolutional_nn.src.layer.Layer import Layer

reg_facotor = 1e-2 # regularization strength
step_size = 1e-0
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
        self.div = OutputLayer.softmax(dot)
        print(self.div)
        self.output = OutputLayer.cross_entrophy_loss(self.div, self.y, self.weights)
        return Promise.resolve(self.output)

    def b(self, gradient):
        dscores = self.div
        num_examples = self.div.shape[0]
        dscores -= self.y
        dscores /= num_examples

        dW = np.dot(self.input.T, dscores)
        dW += reg_facotor* self.weights # don't forget the regularization gradient
        self.d_input = np.dot(dscores, self.weights[0:self.weights.shape[0] - 1].T) # the gradient on the output of the previous layer
        self.update(dW, step_size)
        return Promise.resolve(self.d_input)

    @staticmethod
    def softmax(z):
        z -= np.max(z) # Dividing large number is numerically unstable, this would help with the stability
        num = np.exp(z)
        denom = np.sum(np.exp(z), axis=1)[:,None]
        div = num/denom
        return np.clip(div, 1e-10, 1)
        #if the z is num of samples x num of outputs
        #np.transpose(np.transpose(np.exp(z)) / np.sum(np.exp(z), axis=1))

    @staticmethod
    def cross_entrophy_loss(z, y, w):
        normalizing_factor = 1.0 / float(y.shape[0])

        z_log = np.log10(z)
        z_dot = -y * z_log
        data_loss = normalizing_factor * np.sum(z_dot)
        reg_loss = 0.5*reg_facotor*np.sum(w*w)
        loss = data_loss + reg_loss
        return loss
