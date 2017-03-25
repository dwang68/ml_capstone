from __future__ import absolute_import
import unittest
import numpy as np
from convolutional_nn.src.Filter import Filter

class ForwardPropogationUnitTest(unittest.TestCase):

    def test_convolutionize(self):
        filter = Filter("filter", 1, 3)
        n = 1
        for item in np.nditer(filter.filter_block, op_flags=['readwrite']):
            item[...] = n
            n+=1
        print(filter.filter_block)

        pixels = np.empty((5,5))
        n = 25
        for item in np.nditer(pixels, op_flags=['readwrite']):
            item[...] = n
            n-=1
        print(pixels)
        filters = [filter]
        pooling_block_dimension = 2
        fp = ForwardPropogation(pixels, filters, pooling_block_dimension, 100)
        convo_matrix = fp.convolutionize(filter)
        target_convo_matrix = np.array([[ 759.,  714.,  669.],
                                        [ 534.,  489.,  444.],
                                       [ 309.,  264.,  219.]])
        self.assertTrue(np.array_equal(convo_matrix, target_convo_matrix))


    def test_pooling(self):
        filter = Filter("filter", 1, 1)
        pixels = np.empty((5,5))
        n = 25
        for item in np.nditer(pixels, op_flags=['readwrite']):
            item[...] = n
            n-=1
        print(pixels)
        filters = [filter]
        pooling_block_dimension = 2
        fp = ForwardPropogation(pixels, filters, pooling_block_dimension, 100)
        fp.convolution_matrices = [pixels]
        fp.max_pooling()
        target_matrix = np.array([[ 25.,  24.,  22.],
                               [ 20.,  19.,  17.],
                               [ 10.,   9.,   7.]])
        self.assertTrue(np.array_equal(fp.pooling_matrices[0], target_matrix))
        print(fp.pooling_matrices)


    def test_logistic_unit(self):
        pass


    def test_flatten(self):
        m1 = np.array([[1,2,3],[4,5,6]])
        m2 = np.array([[11,12,13],[14,15,16]])
        m_arr = [m1, m2]
        actual = ForwardPropogation.flatten(m_arr, 12)
        target = np.array([[1,2,3,4,5,6,11,12,13,14,15,16,1]])
        self.assertTrue(np.array_equal(actual, target))




if __name__ == '__main__':
    unittest.main()
