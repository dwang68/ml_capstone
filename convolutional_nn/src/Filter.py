import numpy as np
import pandas as pd
import cv2

class Filter(object):

    def __init__(self, filter_metadata, stride, dimension):
        self.filter_block = np.random.rand(dimension, dimension) / dimension
        self.filter_metadata = filter_metadata
        self.stride = stride
        self.dimension = dimension

