import numpy as np
from promise import Promise

from convolutional_nn.src.layer.Layer import Layer


class PoolingLayer(Layer):

    def __init__(self, filters, pooling_dimension):
        Layer.__init__(self, filters)
        self.pooling_dimension = pooling_dimension

    def f(self, input):
        self.input = input
        self.max_pooling()
        return Promise.resolve(self.output)


    def max_pooling(self):

        if self.pooling_dimension > self.input.shape[1]:
            raise IOError("Pooling block dimension bigger than that of convolutional matrices")
        #if the dimension of the convolutional matrix is not even, we add padding of 1 to make it even
        if self.input.shape[1] % 2 != 0:
            #Number of values padded to the edges of each axis. ((before_1, after_1), ... (before_N, after_N))
            # unique pad widths for each axis.
            self.input = np.lib.pad(self.input, ((0,0),(1,0),(1,0)), "constant")

        for matrix in self.input:
            i = 0
            j = 0
            ip = 0
            jp = 0
            pooling = np.empty((self.input.shape[1]/self.pooling_dimension,
                                self.input.shape[2]/self.pooling_dimension))
            while i < self.input.shape[1]:
                while j < self.input.shape[2]:
                    sliced_block = matrix[i: i+self.pooling_dimension, j: j+self.pooling_dimension]
                    pooling[ip,jp] = np.amax(sliced_block)
                    j+=self.pooling_dimension
                    jp+=1
                i+=self.pooling_dimension
                ip+=1
                j = 0
                jp = 0
            if self.output is not None:
                self.output = np.append(self.output, pooling.reshape([1, pooling.shape[0], pooling.shape[1]]), axis=0)
            else:
                self.output = pooling.reshape([1, pooling.shape[0], pooling.shape[1]])
