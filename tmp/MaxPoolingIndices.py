import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal.pool import max_pool_2d_same_size,pool_2d,MaxPoolGrad


input = T.tensor4()
pool_out = pool_2d(input, ds=(2,2),ignore_border=True,mode='max')
method = 1

if method==1:	
	#indices = T.grad(None, wrt=input, known_grads={pool_out: T.ones_like(pool_out)})
	indices = MaxPoolGrad((2,2), True)(input, pool_out, T.ones_like(pool_out))	
	unpool = indices*pool_out.repeat(2, axis=2).repeat(2, axis=3)
		
elif method==2:
	indices,pool_out = pool_2d(input, ds=(2,2),ignore_border=True,mode='max') #modifiy the code of max_pool_2d, to be completed
	
elif method==3:
	pool_same_size = max_pool_2d_same_size(input,(2,2))
	#indices = pool_same_size>0 #get 0/1 indicating non/argmax
	#unpool = T.set_subtensor(T.flatten(T.zeros_like(input))[T.flatten(indices)],T.flatten(pool_out)).reshape(input.shape)
	
	
	#indices = pool_same_size.nonzero() #get an array of the cordinates of argmax	
	#unpool = T.set_subtensor(T.zeros_like(input)[indices],T.flatten(pool_out))
f_pool = theano.function([input],pool_out)
f_pool_out_same_size = theano.function([input],pool_out_same_size)
f_index = theano.function([input],indices)
f_unpool = theano.function([input],unpool)
a = np.arange(16).reshape(1, 1, 4, 4)

print f(a)	

