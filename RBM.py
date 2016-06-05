import numpy,theano,os
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  #use GPU random to accelerate
from tools import *
from theano.compat import OrderedDict
import timeit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb

class RBM(object):
	# General RBM class, surpport GBRBM BBRBM BGRBM
	def __init__(
		self,
		input=None,
		n_visible=1024,
		n_hidden=1024,
		type='BB', #type 0 1 2 represents BBRBM GBRBM BGRBM respectly
		modelFile=None,		
	):
		"""
		RBM constructor. Defines the parameters of the model along with
		basic operations for inferring hidden from visible (and vice-versa),
		as well as for performing CD updates.
		"""

		self.n_visible = n_visible
		self.n_hidden = n_hidden
				
		#set RBM type
		if type=='BB':
			self.type = 0
		elif type=='GB':
			self.type = 1
		elif type=='BG':
			self.type = 2
		else:
			print 'RBM type %s error, use any of "BB GB BG" instead'%(type)
			exit()
			
		# create a number generator
		numpy_rng = numpy.random.RandomState(1234)

		
		theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		#initial_W = numpy.asarray(0.1*numpy_rng.randn(n_visible,n_hidden),dtype=theano.config.floatX)
		initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
		# theano shared variables for weights and biases
		W = theano.shared(value=initial_W, name='W', borrow=True)
		
		# create shared variable for hidden units bias
		hbias = theano.shared(
			value=numpy.zeros(
				n_hidden,
				dtype=theano.config.floatX
			),
			name='hbias',
			borrow=True
		)
		
		# create shared variable for visible units bias
		vbias = theano.shared(
			value=numpy.zeros(
				n_visible,
				dtype=theano.config.floatX
			),
			name='vbias',
			borrow=True
		)
	
		w_inc = theano.shared(
				value=numpy.zeros(
					[n_visible, n_hidden],
					dtype=theano.config.floatX
				),
				name='w_inc',
				borrow=True
		)
		
		hbias_inc = theano.shared(
				value=numpy.zeros(
					n_hidden,
					dtype=theano.config.floatX
				),
				name='hbias_inc',
				borrow=True
		)
		
		vbias_inc = theano.shared(
				value=numpy.zeros(
					n_visible,
					dtype=theano.config.floatX
				),
				name='vbias_inc',
				borrow=True
		)
		
		# initialize input layer for standalone RBM or layer0 of DBN
		self.input = input
		if not input:
			self.input = T.matrix('input')

		self.W = W
		self.hbias = hbias
		self.vbias = vbias
		self.w_inc = w_inc
		self.hbias_inc = hbias_inc
		self.vbias_inc = vbias_inc
		self.theano_rng = theano_rng 
		
		#default RBM training config
		self.lr = theano.shared(value=np.array(0.0001,dtype=theano.config.floatX))
		self.momentum = theano.shared(value=np.array(0.9,dtype=theano.config.floatX))
		self.weightCost = theano.shared(value=np.array(0.0002,dtype=theano.config.floatX))
	 
		self.params = [self.W, self.hbias, self.vbias]   
		self.params_inc = [self.w_inc, self.hbias_inc, self.vbias_inc]
		
		self.output = self.propup(self.input)[1]
		self.recover = self.propdown(self.output)[1]
		
		if modelFile!=None:
			self.loadModel(modelFile)
		
	def loadModel(self,modelFile):
		if not os.path.exists(modelFile):
			print modelFile,' not exist'
			exit()
		[weights,v_bias,h_bias] = ReadRBMModel(modelFile,self.n_visible,self.n_hidden)
		self.W.set_value(np.asarray(weights,dtype=theano.config.floatX))
		self.vbias.set_value(np.asarray(v_bias,dtype=theano.config.floatX))
		self.hbias.set_value(np.asarray(h_bias,dtype=theano.config.floatX))		

	def propup(self, vis):
		'''This function propagates the visible units activation upwards to
		the hidden units. Note that we return also the pre-sigmoid activation of the
		layer. As it will turn out later, due to how Theano deals with
		optimizations, this symbolic variable will be needed to write
		down a more stable computational graph (see details in the
		reconstruction cost function)
		'''		
		pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
		if self.type==2:
			return [pre_sigmoid_activation, pre_sigmoid_activation]
		else:
			return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_h_given_v(self, v0_sample):
		# This function infers state of hidden units given visible units 
		# compute the activation of the hidden units given a sample of
		# the visibles
		pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
		# get a sample of the hiddens given their activation
		# Note that theano_rng.binomial returns a symbolic sample of dtype
		# int64 by default. If we want to keep our computations in floatX
		# for the GPU we need to specify to return the dtype floatX
		
		if self.type==2:
			return [pre_sigmoid_h1, h1_mean, h1_mean]
		else:
			h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
												 n=1, p=h1_mean,
												 dtype=theano.config.floatX)
			return [pre_sigmoid_h1, h1_mean, h1_sample]
		
	def propdown(self, hid):
		#This function propagates the hidden units activation downwards to the visible units 
		pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
		
		if self.type==1:
			return [pre_sigmoid_activation, pre_sigmoid_activation]
		else:
			return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
		
	def sample_v_given_h(self, h0_sample):
		# This function infers state of visible units given hidden units 
		# compute the activation of the visible given the hidden sample
		pre_sigmoid_v1,v1_mean = self.propdown(h0_sample)		
				
		if self.type==1:
			return [pre_sigmoid_v1, v1_mean, v1_mean]
		else:
			v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
												 n=1, p=v1_mean,
												 dtype=theano.config.floatX)
			return [pre_sigmoid_v1, v1_mean, v1_sample]		
		
	def gibbs_hvh(self, h0_sample):
		# This function implements one step of Gibbs sampling, starting from the hidden state
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [pre_sigmoid_v1, v1_mean, v1_sample,
				pre_sigmoid_h1, h1_mean, h1_sample]

	def gibbs_vhv(self, v0_sample):
		# This function implements one step of Gibbs sampling, starting from the visible state
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return [pre_sigmoid_h1, h1_mean, h1_sample,
				pre_sigmoid_v1, v1_mean, v1_sample]
	
	def gradient(self, nh0_means, nv_means, nh_means):		
		w_grad = (T.dot(self.input.T, nh0_means) - T.dot(nv_means.T, nh_means))/T.cast(self.input.shape[0], dtype=theano.config.floatX)
		vbias_grad = T.mean(self.input-nv_means,0)
		hbias_grad = T.mean(nh0_means-nh_means,0)
		return [-w_grad, -hbias_grad, -vbias_grad]
		
	def get_cost_updates(self, k=1):
		"""This functions implements one step of CD-k 
		
		:param k: number of Gibbs steps to do in CD-k

		Returns a proxy for the cost and the updates dictionary. The
		dictionary contains the update rules for weights and biases but
		also an update of the shared variable used to store the persistent
		chain, if one is used.

		"""

		# compute positive phase
		pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

		# decide how to initialize persistent chain:
		# for CD, we use the newly generate hidden sample		
		
		chain_start = ph_sample
	   
		# end-snippet-2
		# perform actual negative phase
		# in order to implement CD-k/PCD-k we need to scan over the
		# function that implements one gibbs step k times.
		# Read Theano tutorial on scan for more information :
		# http://deeplearning.net/software/theano/library/scan.html
		# the scan will return the entire Gibbs chain
		if k>1:
			(
				[
					pre_sigmoid_nvs,
					nv_means,
					nv_samples,
					pre_sigmoid_nhs,
					nh_means,
					nh_samples
				],
				updates
			) = theano.scan(
				self.gibbs_hvh,
				# the None are place holders, saying that
				# chain_start is the initial state corresponding to the
				# 6th output
				outputs_info=[None, None, None, None, None, chain_start],
				n_steps=k-1
			)
			n_visMean = self.propdown(nh_samples[-1])[1]
			n_hidMean = self.propup(n_visMean)[1]
		elif k==1:
			n_visMean = self.propdown(chain_start)[1]						
			n_hidMean = self.propup(n_visMean)[1]
			
			#pre_sigmoid_v, n_visMean, v_sample = self.sample_v_given_h(chain_start)
			#n_hidMean = self.propup(v_sample)[1]
			
			updates=OrderedDict()
		else:
			print 'cd_steps wrong'
			exit()
			
			
		# start-snippet-3
		# determine gradients on RBM parameters
		# note that we only need the sample at the end of the chain
		#chain_end = nv_samples[-1]

		#cost = T.mean(self.free_energy(self.input)) - T.mean(
		#	self.free_energy(chain_end))
		# We must not compute the gradient through the gibbs sampling
		#gparams = T.grad(cost, self.params, consider_constant=[chain_end])		
		# end-snippet-3 start-snippet-4
		# constructs the update dictionary
		#pdb.set_trace()
		gparams = self.gradient( ph_mean, n_visMean, n_hidMean)
		
		# set the computational graph, real training config parameters will be imported in the training function
		updates[self.params_inc[0]] = self.params_inc[0]*self.momentum - (gparams[0]  + self.W*self.weightCost)* self.lr
		updates[self.params_inc[1]] = self.params_inc[1]*self.momentum - gparams[1] * self.lr   
		updates[self.params_inc[2]] = self.params_inc[2]*self.momentum - gparams[2] * self.lr					   
		for inc, param in zip(self.params_inc, self.params):
			updates[param] = param + inc				   
		
		# reconstruction cross-entropy is a better proxy for CD
		monitoring_cost = self.get_reconstruction_cost(n_visMean)

		return monitoring_cost, updates
		# end-snippet-4		
	def get_reconstruction_cost(self, nv_mean):
		#mean_square_error
		mean_square_error = T.mean(
			T.sum((self.input-nv_mean)**2, axis=1)
		)		
		return mean_square_error
		
	def write_params(self, file):
		weights = self.W.get_value()		
		vbias = self.vbias.get_value()
		hbias = self.hbias.get_value()
		with open(file,'wb') as fout:  
			weights_T = weights.T.copy(order='C')
			fout.write(weights_T)
			fout.write(hbias)
			fout.write(vbias)			
	
	def mapping(self,inDir,outDir,binary):
		print 'mapping %d:\n    %s\n    %s\n'%(binary,inDir,outDir)
		SaveMkdir(outDir)		
		vis = T.matrix('vis')
		[pre_sigmoid_activation, hid_mean] = self.propup(vis)		
		f = theano.function([vis],hid_mean)
		
		for file in sorted(os.listdir(inDir)):			
			infile = inDir+os.sep+file
			outfile = outDir+os.sep+file
			if os.path.exists(outfile):continue
			inData = ReadFloatRawMat(infile,self.n_visible)
			outData = f(np.asarray(inData,dtype=theano.config.floatX))
			if type!=2 and binary==True:
				outData[outData>=0.5]=1.0
				outData[outData<0.5]=0.0
			WriteArrayFloat(outfile,outData)
	
	def checkHidProbs(self,inDir,number=0):
		vis = T.matrix('vis')
		[pre_sigmoid_activation, hid_mean] = self.propup(vis)		
		f = theano.function([vis],hid_mean)		
		hist=np.zeros(10)
		level = 0
		count = 0
		fileNum = 0
		fileList = os.listdir(inDir)
		np.random.shuffle(fileList)
		if number==0:
			number=len(fileList)
		for file in fileList: 
			#print file
			infile = inDir+os.sep+file
			inData = ReadFloatRawMat(infile,self.n_visible)
			outData = f(np.asarray(inData,dtype=theano.config.floatX))
			
			[n_hist,n_value] = np.histogram(outData,0.1*np.arange(11))
			hist += n_hist
			level += np.sum(outData)
			count += np.size(outData)
			fileNum += 1
			
			if fileNum>number:
				break
				
		return hist/float(np.sum(hist)),level/count		

	def PrintTrainingParameters(self,dataDir,modelFile,learnRate=0.001,batch_size=128,initialMomentum=0.5, finalMomentum=0.9, weightCost=0.0002, layerEpochs=100,CDstep=1):
		print '    Training parameters:'
		print '        dataDir: ',dataDir
		print '        modelFile: ',modelFile
		print '        learnRate: ',learnRate
		print '        batch_size: ',batch_size
		print '        initialMomentum: ',initialMomentum
		print '        finalMomentum: ',finalMomentum
		print '        weightCost: ',weightCost
		print '        layerEpochs: ',layerEpochs
		print '        CDstep: ',CDstep
		
	def train(self,dataDir,modelFile,learnRate=0.001,batch_size=128,initialMomentum=0.5, finalMomentum=0.9, weightCost=0.0002, layerEpochs=100):
		CDstep=1
		#pdb.set_trace()
		self.PrintTrainingParameters(dataDir,modelFile,learnRate=learnRate,batch_size=batch_size,initialMomentum=initialMomentum, finalMomentum=finalMomentum, weightCost=weightCost, layerEpochs=layerEpochs,CDstep=CDstep)
		
		cost,updates = self.get_cost_updates(CDstep)  
		train = theano.function(
			[self.input],
			cost,
			updates=updates,		
			name='train'
		)	
		
		self.lr.set_value(learnRate) 
		self.weightCost.set_value(weightCost) 
		
		for epoch in range(layerEpochs):			  
			epoch_start = timeit.default_timer()
			print '    Training epoch %d ......'%(epoch),				
			if epoch<5:
				self.momentum.set_value(initialMomentum)	 
			else:
				self.momentum.set_value(finalMomentum)
			mean_cost = [] 
			fileList = os.listdir(dataDir)
			np.random.shuffle(fileList)	
			
			for item in fileList:  
				file = dataDir+os.sep+item	
				data = ReadFloatRawMat(file,self.n_visible) 
				n_train_batches = data.shape[0] // batch_size
				np.random.shuffle(data)	
				data = np.asarray(data,dtype=theano.config.floatX)				
				for index in range(n_train_batches):
					mean_cost += [train(data[index * batch_size: (index + 1) * batch_size])]				
				#print '\n%f'%(mean_cost[-1])
			epoch_end = timeit.default_timer()
			#self.lr.set_value(np.asarray(self.lr.get_value()*0.9,dtype=theano.config.floatX))
			
			hist,level = self.checkHidProbs(dataDir,100)		
			print ' average cost per sample is %f, epoch time: %f seconds'%(numpy.mean(mean_cost),epoch_end - epoch_start)
			print '        activation level is %.2f, hist on 100 random files: %s'%(level,' '.join(map(lambda x: '%.2f'%x, hist)))
			modelFile_tmp = '%s.%d'%(modelFile,epoch)
			#if np.mod(epoch,10)==0:
			#	self.write_params(modelFile_tmp)  
		self.write_params(modelFile)			
		hist,level = self.checkHidProbs(dataDir)
		print '        activation level is %.2f, hist on whole dataset: %s'%(level,' '.join(map(lambda x: '%.2f'%x, hist)))	
		
		
if __name__=='__main__': 
	
	
	
	modelFile = r'/home/yjhu/theano/DBN/Model_RBM/SLT_1000_RBM_binaryHidConstraint_1_layer1RBM_513-1024.rbm'
	dataDir = r'/home/yjhu/theano/data/logSPE_0Sil_normalization'
	print modelFile
	#log = 'Model/'+os.path.basename(modelFile).split('.')[0]+'.log'
	#sys.stdout = Logger(log) 
		
	rbm = RBM(input=None, n_visible=513, n_hidden=1024, type='GB', dropout=0.0)
	rbm.loadModel(modelFile)
	#rbm.train(dataDir,modelFile,initialMomentum=0.9)
	
	#sys.stdout.flush()	
	
	hist,level = rbm.checkHidProbs(dataDir)
	print level
	print hist
	#plt.bar(range(len(hist)),hist,width=1)
	#plt.title(os.path.basename(modelFile).split('.')[0])
	#plt.savefig('HiddenHistFigure/'+os.path.basename(modelFile).split('.')[0]+'.jpg')	
	#plt.close()
	
	
	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		






		