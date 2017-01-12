
    
import sys
import os

sys.path.append('shapeset1_1cs_2p_3o.10000.train_code')

from image_config import *

d = dataset(fixed='orientation',shapeset=1,tag='train',free='color,size',seed=1,constrained='position',n_examples=10000)
d.data_directory = "."

try:
    task = sys.argv[1]
    args = sys.argv[2:]

    fn = getattr(d, task)
    
except Exception, e:
    print 'Usage: %(prog)s write_formats [format]*\n       %(prog)s view' % {'prog': sys.argv[0]}
    sys.exit()

fn(*args)

