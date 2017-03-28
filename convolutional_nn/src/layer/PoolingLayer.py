import numpy as np
from promise import Promise

from convolutional_nn.src.layer.Layer import Layer


class PoolingLayer(Layer):

    def __init__(self, filters, pooling_dimension):
        Layer.__init__(self, filters)
        self.pooling_dimension = pooling_dimension

    def f(self, input):
        self.input = self.convofy(input)
        self.output_index = np.argmax(self.input, axis=2)
        self.output_index_one_hot_encoded = (np.arange(self.output_index.max()+1) == self.output_index[:,:,None]).astype(int)
        self.output = np.amax(self.input, axis=2).reshape([self.input.shape[0],225])
        return Promise.resolve(self.output)

    def b(self, gradient):
        self.d_input = np.repeat(gradient.reshape(gradient.shape[0],gradient.shape[1],1), 4, axis=2) * self.output_index_one_hot_encoded
        self.d_input = self.d_input.reshape((gradient.shape[0]/2, 2, 900))
        self.d_input = np.transpose(self.d_input, (0,2,1))
        self.d_input = np.average(self.d_input, axis=0)
        return Promise.resolve(self.d_input)

    @staticmethod
    def one_hot_encoding(y):
        y_encoded = np.zeros([y.shape[0], 3])
        y_encoded[np.arange(y.shape[0]), y] = 1


    def convofy(self, input):
        dim = int(np.square(np.sqrt(input.shape[1])/ 2))
        r = np.zeros([input.shape[0], dim , 4])
        for num in range(0, input.shape[0], 1):
            for m in range(0, 30, 2):
                for n in range(0, 30, 2):
                    for i in range(0, 2, 1):
                        for j in range(0, 2, 1):
                            r[num, 15*(m/2) + n/2, 2 * i +j ] = input[num, 30 * (m+i) + (n+j)]
        return r






        # def max_pooling(self):
    #
    #     if self.pooling_dimension > self.input.shape[1]:
    #         raise IOError("Pooling block dimension bigger than that of convolutional matrices")
    #     #if the dimension of the convolutional matrix is not even, we add padding of 1 to make it even
    #     if self.input.shape[1] % 2 != 0:
    #         #Number of values padded to the edges of each axis. ((before_1, after_1), ... (before_N, after_N))
    #         # unique pad widths for each axis.
    #         self.input = np.lib.pad(self.input, ((0,0),(1,0),(1,0)), "constant")
    #
    #     for matrix in self.input:
    #         i = 0
    #         j = 0
    #         ip = 0
    #         jp = 0
    #         pooling = np.empty((self.input.shape[1]/self.pooling_dimension,
    #                             self.input.shape[2]/self.pooling_dimension))
    #         while i < self.input.shape[1]:
    #             while j < self.input.shape[2]:
    #                 sliced_block = matrix[i: i+self.pooling_dimension, j: j+self.pooling_dimension]
    #                 pooling[ip,jp] = np.amax(sliced_block)
    #                 j+=self.pooling_dimension
    #                 jp+=1
    #             i+=self.pooling_dimension
    #             ip+=1
    #             j = 0
    #             jp = 0
    #         if self.output is not None:
    #             self.output = np.append(self.output, pooling.reshape([1, pooling.shape[0], pooling.shape[1]]), axis=0)
    #         else:
    #             self.output = pooling.reshape([1, pooling.shape[0], pooling.shape[1]])
