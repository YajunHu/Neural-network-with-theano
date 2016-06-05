import numpy,theano,os
import theano.tensor as T
import theano.misc.pkl_utils as pkl_utils
from tools import *
import timeit
#import pdb

class AELayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, bhid=None, bvis=None,
				 up_activation=T.nnet.sigmoid, down_activation=T.nnet.sigmoid):	   
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
			if up_activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			W = theano.shared(value=W_values, name='W', borrow=True)

		if bhid is None:
			bhid_values = numpy.zeros(n_out, dtype=theano.config.floatX)
			bhid = theano.shared(value=bhid_values, name='bhid', borrow=True)
		if bvis is None:
			bvis_values = numpy.zeros(n_in, dtype=theano.config.floatX)
			bvis = theano.shared(value=bvis_values, name='bvis', borrow=True)	

		self.W = W
		self.b = bhid
		self.W_prime = self.W.T
		self.b_prime = bvis
		
		# parameters of the model
		self.params = [self.W, self.b, self.b_prime]
		
		if input is None:
			self.input = T.matrix(name='input')
		else:
			self.input = input
		
		self.up_activation = up_activation		
		self.down_activation = down_activation		
		
	def get_hidden(self, input):	
		lin_output = T.dot(input, self.W) + self.b
		output = (
			lin_output if self.up_activation is None
			else self.up_activation(lin_output)
		)
		return output
	
	def get_reconstruction(self,hidden):
		lin_output = T.dot(hidden, self.W_prime) + self.b_prime
		output = (
			lin_output if self.down_activation is None
			else self.down_activation(lin_output)
		)
		return output	
	
class AutoEncoder(object):
	def __init__(self, rng, layerSizes):
		self.AELayers=[]
		self.ups = []
		self.downs = []		
		self.params = []				
		
		self.layerSizes = layerSizes
		self. n_layers = len(layerSizes)-1		
		assert self.n_layers>0
		
		self.input = T.matrix('AE_Input')
		self.ups.append(self.input)
		for i in range(self.n_layers):
			if i==0:
				self.AELayers.append(AELayer(rng, self.ups[i], self.layerSizes[i],self.layerSizes[i+1],down_activation=None))				
			else:
				self.AELayers.append(AELayer(rng, self.ups[i], self.layerSizes[i],self.layerSizes[i+1]))
			self.params += (self.AELayers[i].params)
			self.ups.append(self.AELayers[i].get_hidden(self.ups[i]))
		
		self.downs.append(self.ups[-1])
		for i in range(self.n_layers-1,-1,-1):
			self.downs.append(self.AELayers[i].get_reconstruction(self.downs[self.n_layers-1-i]))
		
		self.loss_rec = T.mean(T.sum(T.power((self.input-self.downs[-1]),2), axis=1))
	
	def DBNPretraining(self, dataDir):
		modelDir = os.path.dirname(self.modelFile)
		title = os.path.basename(self.modelFile).split('.')[0]
		layerSizes = self.layerSizes
		types = ['GB']+['BB']*(len(layerSizes)-2)
		dbn = DBN(title,layerSizes,types,modelDir)
		dbn.DBNTrain(dataDir)
		
		self.loadDBNModel(title,modelDir,layerSizes)
		
	def LoadDBNModel(self,title=None,modelDir=None,layerSizes=None):
		print 'Loading DBN pretrain parameters'
		if title==None or modelDir==None or layerSizes==None:
			modelDir = os.path.dirname(self.modelFile)
			title = os.path.basename(self.modelFile).split('.')[0]
			layerSizes = self.layerSizes
		for i in range(len(layerSizes)-1):
			file = modelDir+os.sep+'%s_layer%dRBM_%d-%d.rbm'%(title,i+1,layerSizes[i],layerSizes[i+1])			
			[weights,v_bias,h_bias] = ReadRBMModel(file,layerSizes[i],layerSizes[i+1])			
			self.AELayers[i].W.set_value(np.asarray(weights,dtype=theano.config.floatX))
			self.AELayers[i].b.set_value(np.asarray(h_bias,dtype=theano.config.floatX))
			self.AELayers[i].b_prime.set_value(np.asarray(v_bias,dtype=theano.config.floatX))			
			
	def SaveModel(self,file):	
		print 'Saving model parameters to %s'%file
		model = []
		for param in self.params:
			model.append(param.get_value())			
		with open(file,'wb') as f:
			pkl_utils.dump(model,f)

	def LoadModel(self,file):
		print 'Loading model paramters from %s'%file
		with open(file,'rb') as f:
			model = pkl_utils.load(f)
		for value,param in zip(model,self.params):
			param.set_value(np.asarray(value,dtype=theano.config.floatX))
	
	def SPEReconstruction(self, dataDir, outDir, meanFile):		
		reconstruction = theano.function(inputs=[self.input], outputs=[self.downs[-1]], name='reconstruction')		
		
		SaveMkdir(outDir)
		mean_std = ReadFloatRawMat(meanFile,self.layerSizes[0])		
		for file in sorted(os.listdir(dataDir)):			
			inFile = dataDir+os.sep+file
			outFile = outDir+os.sep+file			
			print 'SPEReconstruction %s'%outFile
			
			rawData = ReadFloatRawMat(inFile,self.layerSizes[0])			
			inData = (np.log(rawData) - mean_std[0])/mean_std[1]
			outData = reconstruction(np.asarray(inData,dtype=theano.config.floatX))
			recData = np.exp(outData*mean_std[1]+mean_std[0])
			WriteArrayFloat(outFile,recData)	

	def GenHidFeature(self,dataDir,outDir,meanFile):
		genHid = theano.function(inputs=[self.ups[0]], outputs=[self.ups[1]], name='genHid')	
		
		SaveMkdir(outDir)
		mean_std = ReadFloatRawMat(meanFile,self.layerSizes[0])		
		for file in sorted(os.listdir(dataDir)):			
			inFile = dataDir+os.sep+file
			outFile = outDir+os.sep+file			
			print 'GenHidFeature %s'%outFile
			
			rawData = ReadFloatRawMat(inFile,self.layerSizes[0])			
			inData = (np.log(rawData) - mean_std[0])/mean_std[1]
			outData = genHid(np.asarray(inData,dtype=theano.config.floatX))			
			WriteArrayFloat(outFile,outData)	

	def SPE_Hid_Reconstruction(self,dataDir,outDir,meanFile):
		genHid = theano.function(inputs=[self.ups[0]], outputs=[self.ups[1]], name='genHid')	
		hidRec = theano.function(inputs=[self.downs[0]], outputs=[self.downs[-1]], name='hidRec')	
		
		SaveMkdir(outDir)
		mean_std = ReadFloatRawMat(meanFile,self.layerSizes[0])		
		for file in sorted(os.listdir(dataDir)):			
			inFile = dataDir+os.sep+file
			outFile = outDir+os.sep+file			
			print 'SPE_Hid_Reconstruction %s'%outFile			
			rawData = ReadFloatRawMat(inFile,self.layerSizes[0])			
			inData = (np.log(rawData) - mean_std[0])/mean_std[1]
			hidData = genHid(np.asarray(inData,dtype=theano.config.floatX))[0]			
			
			#############	
			main = np.zeros(hidData.shape)
			remain = np.zeros(hidData.shape)
			main[hidData>=0.5]=1.0
			main[hidData<0.5]=0.0	
			remain = hidData-main
			#############
			data = remain
			
			rec = hidRec(np.asarray(data,dtype=theano.config.floatX))
			outData = np.exp(rec*mean_std[1]+mean_std[0])
			WriteArrayFloat(outFile,outData)
			
	def HidReconstruction(self,dataDir,outDir,meanFile):
		hidRec = theano.function(inputs=[self.downs[0]], outputs=[self.downs[-1]], name='hidRec')	
		
		SaveMkdir(outDir)
		mean_std = ReadFloatRawMat(meanFile,self.layerSizes[0])		
		for file in sorted(os.listdir(dataDir)):			
			inFile = dataDir+os.sep+file
			outFile = outDir+os.sep+file			
			print 'HidReconstruction %s'%outFile
			
			inData = ReadFloatRawMat(inFile,self.layerSizes[0])			
			outData = hidRec(np.asarray(inData,dtype=theano.config.floatX))			
			recData = outData*mean_std[1]+mean_std[0]
			WriteArrayFloat(outFile,recData)
	
	def PrintTrainingInfo(self, dataDir, modelFile, learning_rate=0.001, batch_size=20, layerEpochs=100)
		print 'Training auto-encoder: %s'%(self.modelName)
		print 'Training parameters:'
		print '    dataDir: ',dataDir
		print '    modelFile: ',modelFile
		print '    learning_rate: ',learning_rate
		print '    batch_size: ',batch_size
		print '    layerEpochs: ',layerEpochs
	
	def Train(self, dataDir, modelFile, learning_rate=0.001, batch_size=20, layerEpochs=100):	
		self.modelFile = modelFile
		self.modelName = os.path.basename(modelFile).split('.')[0]
		
		self.PrintTrainingInfo(dataDir, modelFile, learning_rate=learning_rate, batch_size=batch_size, layerEpochs=layerEpochs)
		
		cost = self.loss_rec
		
		self.learning_rate = theano.shared(np.asarray(learning_rate,dtype=theano.config.floatX))	
		#pdb.set_trace()
		gparams = [T.grad(cost, param) for param in self.params]		
		updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
		
		train_model = theano.function(inputs=[self.input], outputs=[cost], updates=updates, name='train_model')
		
		trianing_start = timeit.default_timer()	
		for epoch in range(layerEpochs):			  
			epoch_start = timeit.default_timer()
			print 'Training epoch %d ......'%(epoch),		
			mean_cost = [] 			
			fileList = os.listdir(dataDir)
			np.random.shuffle(fileList)	
			for item in fileList:  
				file = dataDir+os.sep+item					
				data = ReadFloatRawMat(file,self.layerSizes[0]) 				
				n_train_batches = data.shape[0] // batch_size				
				np.random.shuffle(data)			  
				data = np.asarray(data,dtype=theano.config.floatX)					
				for index in range(n_train_batches):
					err = train_model(data[index * batch_size: (index + 1) * batch_size])
					mean_cost += [err[0]]					
					#print '\n%f'%(mean_cost[-1])
					
			epoch_end = timeit.default_timer()
			# reduce learning_rate by a factor of 0.9
			#self.learning_rate.set_value(np.asarray(np.max([self.learning_rate.get_value()*0.95, 0.0001]),dtype=theano.config.floatX))			   			
			print 'weighted cost per sample on training set is %f, epoch time: %.2f seconds'%(numpy.mean(mean_cost),epoch_end - epoch_start)			
		trianing_end = timeit.default_timer()	
		print 'Total training time: %.2f hours'%((trianing_end - trianing_start)/3600.0)
		self.SaveModel(self.modelFile)	

def train():
	dataDir = r'/home/yjhu/theano/data/logSPE_0Sil_normalization'
	layerSizes = [513,1024,1024,1024]	
	rng = numpy.random.RandomState(1234)
	modelFile = r'/home/yjhu/theano/DBN_comparison/Model_AE/SLT_1000_AE.model'
	
	log = modelFile.split('.')[0]+'.log'
	sys.stdout = Logger(log)
	AE = AutoEncoder(rng, layerSizes)		
	AE.LoadDBNModel(title='SLT_1000_GBBB',modelDir=r'/home/yjhu/theano/DBN_comparison/Model',layerSizes=layerSizes)
	AE.Train(dataDir, modelFile, learning_rate=0.001)
	sys.stdout.flush()	

def test():
	layerSizes = [513,1024]	
	rng = numpy.random.RandomState(1234)	
	modelFile = r'/home/yjhu/theano/DBN/Model_AE/AE_513_1024.model'
	
	log = modelFile.split('.')[0]+'.log'
	AE = AutoEncoder(rng, layerSizes)		
	AE.LoadModel(modelFile)
		
	meanFile = r'/home/yjhu/theano/testData/mean_std.data'
	dataDir = r'/home/yjhu/theano/testData/SPE_test'
	outDir = r'/home/yjhu/theano/testData/SPE_test_rec_AE_513_1024_tiedWeights_remain' 
	
	AE.SPE_Hid_Reconstruction(dataDir, outDir, meanFile)

	
	
if __name__=='__main__':
	if sys.argv[1]=='test':		
		test();
	elif sys.argv[1]=='train':
		train()
		
		
			
		
		


