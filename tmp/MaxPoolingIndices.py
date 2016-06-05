import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d,max_pool_2d_same_size


input = T.tensor4()
pool_out = max_pool_2d(input, ds=(2,2))
method = 1

if method==1:	
	indices = T.grad(None, wrt=input, known_grads={pool_out: T.ones_like(pool_out)})
		
elif method==2:
	indices,pool_out = max_pool_2d(input, ds=(2,2)) #modifiy the code of max_pool_2d, to be completed
	
elif method==3:
	pool_same_size = max_pool_2d_same_size(input,ds=(2,2))
	indices = pool_same_size>0

f = theano.function([input],indices)
a = np.arange(16).reshape(1, 1, 4, 4)
print f(a)	

