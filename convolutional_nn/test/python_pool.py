from multiprocessing import Pool
import copy_reg
import types

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

class python_pool(object):

    def __init__(self):
        self.r = []

    def f(self, x):
        self.r.append(x*x)
        print("non-blocking")
        return x*x

    def get_r(self):
        return self.r

if __name__ == '__main__':
    p = Pool(5)
    pp = python_pool()
    print(p.map(pp.f, [1, 2, 3]))
    print(pp.get_r())