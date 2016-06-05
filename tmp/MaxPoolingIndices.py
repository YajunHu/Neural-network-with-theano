import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.signal.pool import max_pool_2d_same_size


input = T.tensor4()
pool_out = max_pool_2d(input, ds=(2,2))
method = 1

if method==1:	
	indices = T.grad(None, wrt=input, known_grads={pool_out: T.ones_like(pool_out)})
		
elif method==2:
	indices,pool_out = max_pool_2d(input, ds=(2,2)) #modifiy the code of max_pool_2d, to be completed
	
elif method==3:
	pool_same_size = max_pool_2d_same_size(input,(2,2))
	#indices = pool_same_size>0 #get 0/1 indicating non/argmax
	indices = pool_same_size.nonzero() #get an array of the cordinates of argmax	
	unpool = T.set_subtensor(T.zeros_like(input)[pool_same_size.nonzero()],T.flatten(pool_out))

f_index = theano.function([input],indices)
f_unpool = theano.function([input],unpool)
a = np.arange(16).reshape(1, 1, 4, 4)

print f(a)	

