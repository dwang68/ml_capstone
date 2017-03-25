import numpy as np
from promise import Promise

from convolutional_nn.src.layer.Layer import Layer


class ConvolutionLayer(Layer):

    def __init__(self, filters, stride):
        Layer.__init__(self, filters)
        self.stride = stride

    def f(self, input):
        self.input = input.reshape([input.shape[0], 32, 32])
        for filter in np.split(self.weights, self.weights.shape[0]):
            for input in np.split(self.input, self.input.shape[0]):
                convo = self.convolutionize_helper(input.reshape([input.shape[1], input.shape[2]])
                                                   , filter.reshape([filter.shape[1], filter.shape[2]]))
                if self.output is not None:
                    self.output = np.append(self.output, convo.reshape([1, convo.shape[0], convo.shape[1]]), axis=0)
                else:
                    self.output = convo.reshape([1, convo.shape[0], convo.shape[1]])
        return Promise.resolve(self.output)


    #loop over the input to create a convolutionized matrix, assuming input and filter are square blocks
    def convolutionize_helper(self, input, filter):
        #verify that the filter dimension is within the input dimension
        input_dimension = input.shape[0]
        filter_dimension = filter.shape[0]
        convo_dimension = input_dimension - filter_dimension + 1
        convo = np.empty((convo_dimension, convo_dimension))

        if input_dimension < filter_dimension:
            raise IOError("Input dimension smaller than filter dimension")

        i = 0
        j = 0
        while i <= (input_dimension - filter_dimension):
            while j <= (input_dimension - filter_dimension):
                sliced_block = input[i: i + filter_dimension, j: j + filter_dimension]
                convo[i, j] = np.multiply(sliced_block, filter).sum()
                #TODO:stride needs to be a variable instead 1, next two lines
                j+=1
            i+=1
            j=0

        return convo