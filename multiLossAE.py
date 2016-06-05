import numpy,theano,os
import theano.tensor as T
import theano.misc.pkl_utils as pkl_utils
from tools import *
from DBN import *
import timeit
import shutil
from logistic_sgd import LogisticRegression
from collections import OrderedDict

class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
				 activation=T.nnet.sigmoid):	   
		self.input = input
		
		if W is None:
			W_values = numpy.asarray(
				rng.uniform(
					low=-numpy.sqrt(6. / (n_in + n_out)),
					high=numpy.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b
		self.W_prime = self.W.T

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		# parameters of the model
		self.params = [self.W, self.b]
		
		# paramters for momentun during training
		w_inc = numpy.zeros((n_in, n_out), dtype=theano.config.floatX)
		b_inc = numpy.zeros((n_out,), dtype=theano.config.floatX)
		self.W_inc = theano.shared(value=w_inc, name='W_inc', borrow=True)
		self.b_inc = theano.shared(value=b_inc, name='b_inc', borrow=True)
		self.params_inc = [self.W_inc, self.b_inc]
		

class MultiLossAE(object):
	"""
	auto-encoder with multiple loss functions as Stacked What Where Auto-encoder Class	
	"""
	def __init__(self, rng, input, AELayerSizes, classifyLayerSizes):
		self.input = input
		self.label = T.ivector('label')
		self.params = []
		self.AEparams = []
		self.params_inc = []
		
		self.AELayerSizes = AELayerSizes + AELayerSizes[::-1][1:]
		self.AELayerNum = len(self.AELayerSizes)		
		self.AELayers=[input]
		

		for i in range(1,self.AELayerNum):
			if i==1:
				self.AELayers.append(HiddenLayer(rng, self.input, self.AELayerSizes[0], self.AELayerSizes[1]))
			elif i!=self.AELayerNum-1:	
				self.AELayers.append(HiddenLayer(rng, self.AELayers[i-1].output, self.AELayerSizes[i-1], self.AELayerSizes[i]))
			else:	#last layer: linear output
				self.AELayers.append(HiddenLayer(rng, self.AELayers[i-1].output, self.AELayerSizes[i-1], self.AELayerSizes[i], activation=None))
			self.params += self.AELayers[i].params
			self.AEparams += self.AELayers[i].params
			self.params_inc += self.AELayers[i].params_inc

			
		self.classifyLayerSizes = classifyLayerSizes
		self.classifyLayerNum = len(self.classifyLayerSizes)
		self.classifyLayers=[]
		for i in range(self.classifyLayerNum):
			if i==0:
				mid_layer = len(AELayerSizes)-1
				last_input = self.AELayers[mid_layer].output
			else:
				last_input = self.classifyLayers[i-1].output
			
			if i==0:
				self.classifyLayers.append(HiddenLayer(rng, last_input, AELayerSizes[-1], self.classifyLayerSizes[i]))
			elif i!=self.classifyLayerNum-1:
				self.classifyLayers.append(HiddenLayer(rng, last_input, self.classifyLayerSizes[i-1], self.classifyLayerSizes[i]))
			else:
				self.classifyLayers.append(LogisticRegression(last_input, self.classifyLayerSizes[i-1], self.classifyLayerSizes[i]))
				
			self.params += self.classifyLayers[i].params
			self.params_inc += self.classifyLayers[i].params_inc
			
		self.loss_NLL = (self.classifyLayers[-1].negative_log_likelihood)
		self.loss_L2rec = T.mean(T.sum(T.power((self.input-self.AELayers[-1].output),2), axis=1))
		self.loss_L2M = []
		for i in range(1,self.AELayerNum/2):
			self.loss_L2M.append(T.mean(T.sum(T.power((self.AELayers[i].output-self.AELayers[-i-1].output),2), axis=1)))
		
		self.errors = self.classifyLayers[-1].errors
	
	def SaveModel(self,file):	
		print 'Saving model parameters'
		model = []
		for param in self.params:
			model.append(param.get_value())			
		with open(file,'wb') as f:
			pkl_utils.dump(model,f)
	
	def DBNPretraining(self, dataDir):
		modelDir = os.path.dirname(self.modelFile)
		title = os.path.basename(self.modelFile).split('.')[0]
		layerSizes = self.AELayerSizes[:len(self.AELayerSizes)/2+1]
		types = ['GB']+['BB']*(len(layerSizes)-2)		

		dbn = DBN(title,layerSizes,types,modelDir)
		dbn.DBNTrain(dataDir)
	
	def loadDBNModel(self,title=None,modelDir=None,layerSizes=None):
		print 'Loading DBN pretrain parameters'
		if title==None or modelDir==None or layerSizes==None:
			modelDir = os.path.dirname(self.modelFile)
			title = os.path.basename(self.modelFile).split('.')[0]
			layerSizes = self.AELayerSizes[:len(self.AELayerSizes)/2+1]
		
		for i in range(len(layerSizes)-1):
			file = modelDir+os.sep+'%s_layer%dRBM_%d-%d.rbm'%(title,i+1,layerSizes[i],layerSizes[i+1])
			
			[weights,v_bias,h_bias] = ReadRBMModel(file,layerSizes[i],layerSizes[i+1])
			
			self.AELayers[i+1].params[0].set_value(np.asarray(weights,dtype=theano.config.floatX))
			self.AELayers[i+1].params[1].set_value(np.asarray(h_bias,dtype=theano.config.floatX))
			self.AELayers[-i-1].params[0].set_value(np.asarray(weights.T,dtype=theano.config.floatX))
			self.AELayers[-i-1].params[1].set_value(np.asarray(v_bias,dtype=theano.config.floatX))
		
	def LoadModel(self,file):
		print 'Loading model paramters'
		with open(file,'rb') as f:
			model = pkl_utils.load(f)
		for value,param in zip(model,self.params):
			param.set_value(np.asarray(value,dtype=theano.config.floatX))
	
	def ShowResults(self,dataDir,labelDir):  # show errors
		start_time = timeit.default_timer()
		errors = self.errors(self.label)		
		
		if len(self.loss_L2M)==1:
			loss_L2M = self.loss_L2M[0]
		else:
			loss_L2Ms = T.stack(self.loss_L2M)
			sums,updates_loss_L2M = theano.scan(lambda x,z:x+z,outputs_info=loss_L2Ms[0],sequences=loss_L2Ms[1:])
			loss_L2M = sums[-1]/len(self.loss_L2M)		
		
		test = theano.function(inputs=[self.input,self.label], outputs=[errors,self.loss_L2rec,loss_L2M], name='test')
		
		batch_size = 50
		classifyError = []
		reconstructionError = []
		midReconstructionError = []
		for item in os.listdir(dataDir):
			file = dataDir+os.sep+item	
			labFile = labelDir+os.sep+item
			data = ReadFloatRawMat(file,self.AELayerSizes[0]) 
			labelData = ReadFloatRawMat(labFile,1) 
			data = np.asarray(data,dtype=theano.config.floatX)		  
			labelData = np.asarray(np.round(labelData),dtype=np.int32).reshape(len(labelData),)	  
			
			n_train_batches = data.shape[0] // batch_size  
			for index in range(n_train_batches):
				err = test(data[index * batch_size: (index + 1) * batch_size], labelData[index * batch_size: (index + 1) * batch_size])
				classifyError += [err[0]]				
				reconstructionError += [err[1]]				
				midReconstructionError += [err[2]]		
		end_time = timeit.default_timer()
		#print '(time used %.2f seconds):'%(end_time - start_time)
		print 'Testing results of data in %s'%dataDir
		print 'classfication error: %f'%(np.mean(classifyError))
		print 'reconstruction error: %f'%(np.mean(reconstructionError))
		print 'middle layer reconstruction error: %f'%(np.mean(midReconstructionError))
	
	def SPEReconstruction(self, dataDir, outDir, meanFile):		
		reconstruction = theano.function(inputs=[self.input], outputs=[self.AELayers[-1].output], name='reconstruction')		
		
		SaveMkdir(outDir)
		mean_std = ReadFloatRawMat(meanFile,self.AELayerSizes[0])		
		for file in sorted(os.listdir(dataDir)):			
			inFile = dataDir+os.sep+file
			outFile = outDir+os.sep+file			
			print 'Reconstrut %s'%outFile
			
			rawData = ReadFloatRawMat(inFile,self.AELayerSizes[0])			
			inData = (np.log(rawData) - mean_std[0])/mean_std[1]
			outData = reconstruction(np.asarray(inData,dtype=theano.config.floatX))
			recData = np.exp(outData*mean_std[1]+mean_std[0])
			WriteArrayFloat(outFile,recData)	

	def GenHidReconstruction(self, dataDir, outDir, meanFile):
		genHid = theano.function(inputs=[self.input],outputs=[self.AELayers[self.AELayerNum/2].output],name='genHid')
		recInput = theano.function(inputs=[self.AELayers[self.AELayerNum/2].output],outputs=[self.AELayers[-1].output],name='recInput')
		
		SaveMkdir(outDir)
		mean_std = ReadFloatRawMat(meanFile,self.AELayerSizes[0])		
		for file in sorted(os.listdir(dataDir)):			
			inFile = dataDir+os.sep+file
			outFile = outDir+os.sep+file			
			print 'GenHidReconstruction %s'%outFile
			
			rawData = ReadFloatRawMat(inFile,self.AELayerSizes[0])			
			inData = ((rawData) - mean_std[0])/mean_std[1]
			
			[hid] = genHid(np.asarray(inData,dtype=theano.config.floatX))
			hid = np.mean(hid,0).reshape(1,self.AELayerSizes[self.AELayerNum/2])				
			
			outData = recInput(hid)
			recData = outData*mean_std[1]+mean_std[0]
			WriteArrayFloat(outFile,recData)	

		
	def Train(self, dataDir, labelDir, modelFile, lambdas, learning_rate=0.0001, batch_size=20, layerEpochs=100, DBNPretrain=False):			
		self.modelFile = modelFile
		self.modelName = os.path.basename(modelFile).split('.')[0]
		if DBNPretrain==True:
			print 'DBN pretraining'
			self.DBNPretraining(dataDir)
			self.loadDBNModel()
		
		print 'Training multi-loss function auto-encoder: %s, error weights: %f %f %f, learning_rate: %f'%(self.modelName, lambdas[0],lambdas[1],lambdas[2],learning_rate)
		
		if len(self.loss_L2M)==1:
			cost = lambdas[0]*self.loss_NLL(self.label) + lambdas[1] * self.loss_L2M[0] + lambdas[2] * self.loss_L2rec	
			loss_L2M = self.loss_L2M[0]
		elif len(self.loss_L2M)==0:
			cost = lambdas[0]*self.loss_NLL(self.label) + lambdas[2] * self.loss_L2rec
			loss_L2M = theano.shared(np.asarray(0.0,dtype=theano.config.floatX))	
		else:
			loss_L2Ms = T.stack(self.loss_L2M)
			sums,updates_loss_L2M = theano.scan(lambda x,z:x+z,outputs_info=loss_L2Ms[0],sequences=loss_L2Ms[1:])
			loss_L2M = sums[-1]/len(self.loss_L2M)
			cost = lambdas[0]*self.loss_NLL(self.label) + lambdas[1] * loss_L2M + lambdas[2] * self.loss_L2rec
		
		errors = self.errors(self.label)		
		
		self.learning_rate = theano.shared(np.asarray(learning_rate,dtype=theano.config.floatX))		
		gparams = [T.grad(cost, param) for param in self.params]		
		updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
		
		''' # tried momentum, failed --> lead to worse training curve
		momentum = 0.0
		self.momentum = theano.shared(np.asarray(momentum,dtype=theano.config.floatX))
		updates = OrderedDict()		
		for inc, param, gparam in zip(self.params_inc, self.params, gparams):
			updates[inc] = inc*self.momentum - gparam*self.learning_rate
			updates[param] = param + inc  
		'''	
			
		train_model = theano.function(inputs=[self.input,self.label], outputs=[cost,errors,self.loss_L2rec,loss_L2M], updates=updates, name='train_model')
		
		trianing_start = timeit.default_timer()	
		for epoch in range(layerEpochs):			  
			epoch_start = timeit.default_timer()
			print 'Training epoch %d ......'%(epoch),		
			mean_cost = [] 
			classifyError = []
			reconstructionError = []
			midReconstructionError = []
			fileList = os.listdir(dataDir)
			np.random.shuffle(fileList)	
			for item in fileList:  
				file = dataDir+os.sep+item	
				labFile = labelDir+os.sep+item
				data = ReadFloatRawMat(file,self.AELayerSizes[0]) 
				labelData = ReadFloatRawMat(labFile,1) 
				n_train_batches = data.shape[0] // batch_size  
				
				combined = zip(data,labelData)
				np.random.shuffle(combined)				
				data, labelData=zip(*combined)			
				  
				data = np.asarray(data,dtype=theano.config.floatX)		  
				labelData = np.asarray(np.round(labelData),dtype=np.int32).reshape(len(labelData),)	   				
				for index in range(n_train_batches):
					err = train_model(data[index * batch_size: (index + 1) * batch_size], labelData[index * batch_size: (index + 1) * batch_size])
					mean_cost += [err[0]]
					classifyError += [err[1]]
					reconstructionError += [err[2]]
					midReconstructionError += [err[3]]
					
					#print '\n%f'%(mean_cost[-1])
					
			epoch_end = timeit.default_timer()
			# reduce learning_rate by a factor of 0.9
			self.learning_rate.set_value(np.asarray(np.max([self.learning_rate.get_value()*0.9, 0.0001]),dtype=theano.config.floatX))			   
			
			print 'weighted cost per sample on training set is %f, epoch time: %.2f seconds'%(numpy.mean(mean_cost),epoch_end - epoch_start)			
			print '    classfication error: %f'%(np.mean(classifyError))
			print '    reconstruction error: %f'%(np.mean(reconstructionError))
			print '    middle layer reconstruction error: %f'%(np.mean(midReconstructionError))
			
		trianing_end = timeit.default_timer()	
		print 'Total training time: %.2f hours'%((trianing_end - trianing_start)/3600.0)
			
			
			
				
		self.SaveModel(self.modelFile)

def testMLAE():
	dataDir = r'/home/yjhu/theano/data/logSPE_0Sil_normalization'
	labelDir = r'/home/yjhu/theano/data/phoneme_label_0Sil'
	
	learning_rate=0.001
	batch_size=20
	layerEpochs=100	
	AELayerSizes = [513,1024]
	classifyLayerSizes = [256,41]
	
	'''	
	
	lambdas = [0,1,1]	
	rng = numpy.random.RandomState(1234)
	input = T.matrix('input')	
	modelFile = r'/home/yjhu/theano/SWWAE/model/MLAE_L2M_L2rec_lr_001.model'
	log = modelFile.split('.')[0]+'.log'
	sys.stdout = Logger(log)
	MLAE = MultiLossAE(rng, input, AELayerSizes, classifyLayerSizes)	
	MLAE.loadDBNModel(title='MLAE_autoEncoder_DBNPretrain',modelDir=os.path.dirname(modelFile),layerSizes=AELayerSizes)	
	MLAE.Train(dataDir, labelDir,  modelFile, lambdas, learning_rate, batch_size, layerEpochs)
	sys.stdout.flush()	
	'''
	'''
	lambdas = [10,1,1]	
	rng = numpy.random.RandomState(1234)
	input = T.matrix('input')	
	modelFile = r'/home/yjhu/theano/SWWAE/model/MLAE_L2M_L2rec_L2NLL_weights1011.model'
	log = modelFile.split('.')[0]+'.log'
	sys.stdout = Logger(log)
	MLAE = MultiLossAE(rng, input, AELayerSizes, classifyLayerSizes)	
	MLAE.loadDBNModel(title='MLAE_autoEncoder_DBNPretrain',modelDir=os.path.dirname(modelFile),layerSizes=AELayerSizes)	
	MLAE.Train(dataDir, labelDir,  modelFile, lambdas, learning_rate, batch_size, layerEpochs)
	sys.stdout.flush()
	
	
	lambdas = [0.1,1,1]	
	rng = numpy.random.RandomState(1234)
	input = T.matrix('input')	
	modelFile = r'/home/yjhu/theano/SWWAE/model/MLAE_L2M_L2rec_L2NLL_weights0111.model'
	log = modelFile.split('.')[0]+'.log'
	sys.stdout = Logger(log)
	MLAE = MultiLossAE(rng, input, AELayerSizes, classifyLayerSizes)	
	MLAE.loadDBNModel(title='MLAE_autoEncoder_DBNPretrain',modelDir=os.path.dirname(modelFile),layerSizes=AELayerSizes)	
	MLAE.Train(dataDir, labelDir,  modelFile, lambdas, learning_rate, batch_size, layerEpochs)
	sys.stdout.flush()
	
	lambdas = [100,1,1]	
	rng = numpy.random.RandomState(1234)
	input = T.matrix('input')	
	modelFile = r'/home/yjhu/theano/SWWAE/model/MLAE_L2M_L2rec_L2NLL_weights10011.model'
	log = modelFile.split('.')[0]+'.log'
	sys.stdout = Logger(log)
	MLAE = MultiLossAE(rng, input, AELayerSizes, classifyLayerSizes)	
	MLAE.loadDBNModel(title='MLAE_autoEncoder_DBNPretrain',modelDir=os.path.dirname(modelFile),layerSizes=AELayerSizes)	
	MLAE.Train(dataDir, labelDir,  modelFile, lambdas, learning_rate, batch_size, layerEpochs)
	sys.stdout.flush()
	
	lambdas = [0.01,1,1]	
	rng = numpy.random.RandomState(1234)
	input = T.matrix('input')	
	modelFile = r'/home/yjhu/theano/SWWAE/model/MLAE_L2M_L2rec_L2NLL_weights00111.model'
	log = modelFile.split('.')[0]+'.log'
	sys.stdout = Logger(log)
	MLAE = MultiLossAE(rng, input, AELayerSizes, classifyLayerSizes)	
	MLAE.loadDBNModel(title='MLAE_autoEncoder_DBNPretrain',modelDir=os.path.dirname(modelFile),layerSizes=AELayerSizes)	
	MLAE.Train(dataDir, labelDir,  modelFile, lambdas, learning_rate, batch_size, layerEpochs)
	sys.stdout.flush()
	
	lambdas = [1,0,0]	
	rng = numpy.random.RandomState(1234)
	input = T.matrix('input')	
	modelFile = r'/home/yjhu/theano/SWWAE/model/MLAE_phoneme_classification.model'
	log = modelFile.split('.')[0]+'.log'
	sys.stdout = Logger(log)
	MLAE = MultiLossAE(rng, input, AELayerSizes, classifyLayerSizes)	
	MLAE.loadDBNModel(title='MLAE_autoEncoder_DBNPretrain',modelDir=os.path.dirname(modelFile),layerSizes=AELayerSizes)	
	MLAE.Train(dataDir, labelDir,  modelFile, lambdas, learning_rate, batch_size, layerEpochs)
	sys.stdout.flush()
	'''
	'''
	learning_rate=0.0001
	lambdas = [0,0,1]
	AELayerSizes = [513,1024,1024]
	classifyLayerSizes = [256,41]
	rng = numpy.random.RandomState(1234)
	input = T.matrix('input2')
	DBNPretrain = False
	modelFile = r'/home/yjhu/theano/SWWAE/model/MLAE_autoEncoder_noPretrain_lr001.model'
	log = modelFile.split('.')[0]+'.log'
	sys.stdout = Logger(log)	
	MLAE2 = MultiLossAE(rng, input, AELayerSizes, classifyLayerSizes)	
	MLAE2.Train(dataDir, labelDir, modelFile, lambdas, learning_rate, batch_size, layerEpochs, DBNPretrain)
	sys.stdout.flush()
	'''
	
if __name__=='__main__':
	testMLAE()
	'''
	rng = numpy.random.RandomState(1234)
	input = T.matrix('input')	
	AELayerSizes = [513,1024,1024]
	classifyLayerSizes = [256,41]
	
	modelFile = r'/home/yjhu/theano/SWWAE/model/MLAE_L2M_L2rec_L2NLL_weights1011.model'
	MLAE = MultiLossAE(rng, input, AELayerSizes, classifyLayerSizes)
	MLAE.LoadModel(modelFile)
	
	DataDir = r'/home/yjhu/theano/testData/state_logSPE_0sil'	
	meanFile = r'/home/yjhu/theano/testData/mean_std.data'
	OutDataDir = r'/home/yjhu/theano/testData/Hid_recSPE_MLAE_1011'
	MLAE.GenHidReconstruction(DataDir,OutDataDir,meanFile)
	'''
	
	
	'''
	testDataDir = r'/home/yjhu/theano/testData/logspg_nosil_normalization'
	testLabelDir = r'/home/yjhu/theano/testData/phoneme_label_0Sil'
	MLAE.ShowResults(testDataDir,testLabelDir)
	'''
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		