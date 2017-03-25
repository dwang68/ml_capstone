import numpy as np
import pandas as pd
import cv2
import math

class Layer(object):

    def __init__(self, weights):
        self.weights = weights
        self.input = None
        self.output = None
        self.local_gradient = None

    def f(self, input):
        pass

    def b(self, input):
        pass