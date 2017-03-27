import numpy as np

def convo():
    arr = np.arange(1, 1025)
    arr = arr.reshape((32, 32))
    r = np.zeros([900, 9])
    stride = 1
    i = 0
    j = 0
    m = 0
    n = 0
    while m < 30:
        while n < 30:
            while i < 3:
                while j < 3:
                    r[m*30+n , 3 * i + j] = arr[m+i,n+j]
                    j+=1
                i+=1
                j=0
            n+=1
            i=0
        m+=1
        n=0
    print r

convo()