import numpy as np

from convolutional_nn.src.NNM import NNM
from convolutional_nn.src.layer.ConvolutionLayer import ConvolutionLayer
from convolutional_nn.src.layer.HiddenLayer import HiddenLayer
from convolutional_nn.src.layer.OutputLayer import OutputLayer
from convolutional_nn.src.layer.PoolingLayer import PoolingLayer


class NNC(object):

    def __init__(self, x, y, batch_size, sigma):

        self.sigma = sigma
        self.x = x
        self.y = y
        if self.x.shape[0]%batch_size != 0:
            raise IOError("data of shape " + str(self.x.shape) +
                          " cannot be split even with given batch size " + str(batch_size))
        self.batch_size = batch_size

        self.nnm = self.initialize()

    def learn(self):
        x_arr = np.split(self.x, self.x.shape[0] / self.batch_size)
        y_arr = np.split(self.y, self.y.shape[0] / self.batch_size)

        for index, batch_input in enumerate(x_arr):
            self.nnm.fit(batch_input, y_arr[index])



    #default settings for the neural network
    def initialize(self):
        l = {}
        convo_filter_num = 2
        #Construct convolutional layer
        c = ConvolutionLayer(self.sigma * np.random.randn(9, convo_filter_num), 1)
        #Construct pooling layer
        p = PoolingLayer(None, 2)
        #Construct hidden layers
        h = HiddenLayer(self.sigma * np.random.randn(225 + 1, 150))
        #Construct output layer
        o = OutputLayer(self.sigma * np.random.randn(150 + 1, 3), convo_filter_num)
        l["convo"] = c
        l["pooling"] = p
        l["hidden"] = h
        l["output"] = o
        m = NNM(l)
        return m

    def set_layer(self, i, l):
        self.nnm.set_layer(i, l)


