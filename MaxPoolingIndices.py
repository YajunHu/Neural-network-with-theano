import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d


inp = T.tensor4()

f = theano.function([inp], [T.grad(T.sum(max_pool_2d(inp, ds=(2,2))), wrt=inp)])
a = np.arange(16).reshape(1, 1, 4, 4)