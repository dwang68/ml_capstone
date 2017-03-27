import copy_reg
import types
from multiprocessing import Pool

import numpy as np

from convolutional_nn.src.layer.ConvolutionLayer import ConvolutionLayer
from convolutional_nn.src.layer.PoolingLayer import PoolingLayer


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)



ar = np.array([[[1,2,3],[4,5,6]], [[11,12,13],[14,15,16]]])
print(ar.shape)

arr = ar.reshape(2,6)
print(arr.shape)

####################################
from itertools import izip_longest

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

##################################
pl = PoolingLayer(None)
cl = ConvolutionLayer(None, 1)
ls = [pl, cl]

for l in ls:
    print(isinstance(l, ConvolutionLayer))

################################
from promise import Promise

m = lambda x, y : x * y

print(m(2,5))

p = Promise.all([Promise.resolve('a'), 'b', Promise.resolve('c')]) \
    .then(lambda res: res == ['d', 'e', 'f'])

assert p.value is False


class P1(object):

    def __init__(self):
        pass

    def p1(self, input):
        pool = Pool(processes=4)
        p3a = pool.apply_async(P3().p3, input)
        p3b = pool.apply_async(P3().p3, input)
        p3a.get()
        p3b.get()
        print("in p1")
        return Promise.resolve(input)


class P2(object):

    def __init__(self):
        pass

    def p2(self, input):
        return Promise.resolve("p2" + input + "p2")

class P3(object):

    def __init__(self):
        pass

    @staticmethod
    def p3(input):
        print("in p3")
        return Promise.resolve("p3" + input + "p3")
p2 = P2()
print(P1().p1("p1").then(p2.p2).value)

# pm1 = P1().p1("p1")
# pm2 = pm1.then(p2.p2)
# print(pm2.then(lambda i : Promise.resolve(i + "p3")).value)

########################################
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
w = np.array([[9,10],[11,12]])
x = x.reshape([1,2,2])
y = y.reshape([1,2,2])
w = w.reshape([1,2,2])
x = np.append(x,y,axis=0)
x = np.append(x,w, axis=0)

#print(x)





