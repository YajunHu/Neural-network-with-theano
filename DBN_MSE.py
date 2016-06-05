from RBM_MSE import *
import shutil,sys

class DBN_MSE(object):
	def __init__(self,title,layerSizes,types,modelDir,MSEType='mean'):
		#checking DBN configuration
		if not len(types)==len(layerSizes)-1:   
			print 'DBN initialize error: layerSizes types length mismatch'
			exit()
		for i in range(len(types)-1):
			if not (types[i] in ['BB','GB','BG'] and types[i+1] in ['BB','GB','BG']):
				print "DBN initialize error: RBM types error, RBM types must be ['BB','GB','BG']"
				exit()
			if not types[i][1]==types[i+1][0]:
				print 'DBN initialize error: RBM types sequence error'
				exit()		
		
		print '\nInitializing DBN: %s, layerSizes: %s, layerTypes: %s'%(title,layerSizes,types)
		self.title = title
		self.layerSizes = layerSizes
		self.layerNum = len(self.layerSizes)-1
		self.modelDir = modelDir   
	   
		self.rbms = []
		self.rbmFiles = []
		for i in range(self.layerNum):			
			self.rbms.append(RBM_MSE(input=None,n_visible=self.layerSizes[i],n_hidden=self.layerSizes[i+1],type=types[i],MSEType=MSEType))			
			self.rbmFiles.append(modelDir+os.sep+'%s_layer%dRBM_%d-%d.rbm'%(self.title,i+1,self.layerSizes[i],self.layerSizes[i+1]))			
			'''
			if os.path.exists(self.rbmFiles[i]):
				print 'Loading layer %d RBM'%(i+1)
				self.rbms[i].loadModel(self.rbmFiles[i])
			'''	
	
	def LoadModel(self):
		for i in range(self.layerNum):
			print 'loading layer %d RBM: %s'%(i+1,self.rbmFiles[i])
			self.rbms[i].loadModel(self.rbmFiles[i])
			
	def SPEupdown(self,inDir,outDir,meanStdFile,type):		
		upFuncs = []
		downFuncs = []
		for i in range(self.layerNum):
			fup = theano.function([self.rbms[i].input],self.rbms[i].output)
			fdown = theano.function([self.rbms[i].output],self.rbms[i].recover)
			upFuncs.append(fup)
			downFuncs.append(fdown)
		
		SaveMkdir(outDir)
		count=0
		meanStd = ReadFloatRawMat(meanStdFile,self.layerSizes[0])
		for item in sorted(os.listdir(inDir)):
			print item
			infile = inDir+os.sep+item
			outfile = outDir+os.sep+item
			data = ReadFloatRawMat(infile,self.layerSizes[0])			
			data = np.asarray((np.log(data)-meanStd[0])/meanStd[1],dtype=theano.config.floatX)  	
			
			if type == 'MeanField':
				for i in range(self.layerNum):
					data = upFuncs[i](data)					
				

				for i in range(self.layerNum-1,-1,-1):
					data = downFuncs[i](data)
					
			elif type=='Binary':
				for i in range(self.layerNum):
					data = upFuncs[i](data)	
					data = Binary(data)	
					
				for i in range(self.layerNum-1,-1,-1):
					data = downFuncs[i](data)
					if i!=0:
						data = Binary(data)
			
			elif type=='DBC':
				for i in range(self.layerNum):
					data = upFuncs[i](data)	
					
				data = Binary(data)	
				
				for i in range(self.layerNum-1,-1,-1):
					data = downFuncs[i](data)					
				
			recData = np.exp(data*meanStd[1]+meanStd[0])	 
			WriteArrayFloat(outfile,recData)		
			count+=1

			if count>50:
				break		
	
			
	def GenHidMeanReconstruction(self,inDir,outDir,meanStdFile,type):
		upFuncs = []
		downFuncs = []
		for i in range(self.layerNum):
			fup = theano.function([self.rbms[i].input],self.rbms[i].output)
			fdown = theano.function([self.rbms[i].output],self.rbms[i].recover)
			upFuncs.append(fup)
			downFuncs.append(fdown)
		
		print 'GenHidMeanReconstruction: %s\n    %s'%(type,outDir)
		SaveMkdir(outDir)		
		meanStd = ReadFloatRawMat(meanStdFile,self.layerSizes[0])
		for item in sorted(os.listdir(inDir)):
			
			infile = inDir+os.sep+item
			outfile = outDir+os.sep+item
			data = ReadFloatRawMat(infile,self.layerSizes[0])
			data = np.asarray((data-meanStd[0])/meanStd[1], dtype=theano.config.floatX)
			
			if type=='MeanField':
				for i in range(self.layerNum):
					data = upFuncs[i](data)					
				
				data = np.mean(data,0)	
				#data = Binary(data)	
				
				data.shape=1,data.shape[0]
				for i in range(self.layerNum-1,-1,-1):
					data = downFuncs[i](data)
					
					
			elif type=='Binary':
				for i in range(self.layerNum):
					data = upFuncs[i](data)					
					data = Binary(data)
				
				data = np.mean(data,0)	
				data = Binary(data)
				
				data.shape=1,data.shape[0]
				for i in range(self.layerNum-1,-1,-1):
					data = downFuncs[i](data)
					if i!=0:
						data = Binary(data)
						
			elif type=='DBC':
				for i in range(self.layerNum):
					data = upFuncs[i](data)				
					
				data = Binary(data)				
				data = np.mean(data,0)	
				data = Binary(data)
				
				data.shape=1,data.shape[0]
				for i in range(self.layerNum-1,-1,-1):
					data = downFuncs[i](data)
						
			else:
				print 'Gen method wrong'
				exit()
			
			recData = data*meanStd[1]+meanStd[0]	 
			WriteArrayFloat(outfile,recData)	   
	
	def GenHid(self,inDir,outDir,meanStdFile,log=True):
		upFuncs = []		
		for i in range(self.layerNum):
			fup = theano.function([self.rbms[i].input],self.rbms[i].output)			
			upFuncs.append(fup)			
		
		SaveMkdir(outDir)		
		meanStd = ReadFloatRawMat(meanStdFile,self.layerSizes[0])
		for item in sorted(os.listdir(inDir)):
			print item
			infile = inDir+os.sep+item
			outfile = outDir+os.sep+item.split('.')[0]+'.gh'
			data = ReadFloatRawMat(infile,self.layerSizes[0])
			if log==True:
				data=np.log(data)
			data = np.asarray((data-meanStd[0])/meanStd[1], dtype=theano.config.floatX)
			for i in range(self.layerNum):
				data = upFuncs[i](data)			
			WriteArrayFloat(outfile,data)	

	def HidDown(self,inDir,outDir,meanStdFile):			
		downFuncs = []
		for i in range(self.layerNum):			
			fdown = theano.function([self.rbms[i].output],self.rbms[i].recover)			
			downFuncs.append(fdown)
			
		SaveMkdir(outDir)		
		meanStd = ReadFloatRawMat(meanStdFile,self.layerSizes[0])
		for item in sorted(os.listdir(inDir)):
			print item
			infile = inDir+os.sep+item
			outfile = outDir+os.sep+item.split('.')[0]+'.spg'
			
			data = ReadFloatRawMat(infile,self.layerSizes[-1])
			#data = data[:,0:self.layerSizes[-1]]
				
			data = np.asarray(data, dtype=theano.config.floatX)
			for i in range(self.layerNum-1,-1,-1):
				data = downFuncs[i](data)
			recData = np.exp(data*meanStd[1]+meanStd[0])	 
			WriteArrayDouble(outfile,recData)	 
	
	def DBNTrain(self,inputDataDir,MSEWeight=1,binaryMapping=True,lr=0.001,batch_size=20,initialMomentum=0.5, finalMomentum=0.9, weightCost=0.0002, layerEpochs=100):
		# print more training information here
		print 'Training DBN: %s, layerSizes: %s'%(self.title,self.layerSizes)				
		#print 'Training parameters: learnRate %s, batchSize %d, initialMomentum %f, finalMomentum %f, weightCost %f, layerEpochs %d'%(' '.join(map(lambda x:str(x),lr)),batch_size,initialMomentum,finalMomentum,weightCost,layerEpochs)
		#print 'Training data: %s'%inputDataDir
		
		start_time = timeit.default_timer()

		if type(lr)==list: 
			if len(lr)!=self.layerNum:
				print 'learning rate sequence length mismatch'
				exit()
			learnRate = lr
		else: 
			learnRate = lr*np.ones(self.layerNum,dtype=theano.config.floatX)
			
		for i in range(self.layerNum):
			if i==0: 
				inDir = inputDataDir
				outDir = self.modelDir+os.sep+'%s_layer%d_%d_%d_trainingData'%(self.title,i+2,self.layerSizes[i+1],self.layerSizes[i+2])
			else:
				inDir = self.modelDir+os.sep+'%s_layer%d_%d_%d_trainingData'%(self.title,i+1,self.layerSizes[i],self.layerSizes[i+1])
				if i<(len(self.layerSizes)-2): 
					outDir = self.modelDir+os.sep+'%s_layer%d_%d_%d_trainingData'%(self.title,i+2,self.layerSizes[i+1],self.layerSizes[i+2])
									
			if os.path.exists(self.rbmFiles[i]):
				print 'Loading layer %d RBM'%(i+1)
				self.rbms[i].loadModel(self.rbmFiles[i])
				self.rbms[i].mapping(inDir,outDir,binaryMapping)
				if i>0: shutil.rmtree(inDir)
				continue
								
			print '\nTraining layer %d RBM'%(i+1)
			layer_start_time = timeit.default_timer()
			
			self.rbms[i].train(dataDir=inDir, modelFile=self.rbmFiles[i], MSEWeight=MSEWeight, learnRate=learnRate[i], batch_size=batch_size, initialMomentum=initialMomentum, finalMomentum=finalMomentum, weightCost=weightCost, layerEpochs=layerEpochs)
			
			layer_end_time = timeit.default_timer()
			print 'Training layer %d RBM complete, layer training time: %f minutes\n'%(i+1,(layer_end_time-layer_start_time)/60.0)
									
			if i<(len(self.layerSizes)-2): 
				print 'Mapping layer %d data, binary: %s'%(i+1,binaryMapping)
				self.rbms[i].mapping(inDir,outDir,binaryMapping)
			if i>0: shutil.rmtree(inDir)
			
		end_time = timeit.default_timer()	
		print 'Training DBN %s complete, total training time: %f hours'%(self.title,(end_time-start_time)/3600.0)

def test():	
	work='tts'
	if work=='tts':
		modelDir = r'/home/yjhu/theano/DBN/Model_new'
		DBNtitle = 'SLT_1000_DBN_modeRecConstraint_1_MSE_hvSample_ori'	
		for layers in [1,2,3,4]:	
			layers=3
			type='MeanField'
			layerSizes = [513]+[1024]*layers
			types = ['GB']+['BB']*(layers-1)
			
			dbn = DBN(DBNtitle,layerSizes,types,modelDir)		
			dbn.LoadModel()
			
			meanStdFile = r'/home/yjhu/theano/testData/mean_std.data'	
			inDir = r'/home/yjhu/theano/testData/state_logSPE_0sil'
			
			outDir = r'/home/yjhu/theano/testData/Hid_recSPE_DBN_modeRecConstraint_1_MSE_ori_hvSample_%dhid_gen_MeanField'%(layers)
			dbn.GenHidMeanReconstruction(inDir,outDir,meanStdFile,'MeanField')		
			exit()
			'''
			for type in ['DBC','Binary','MeanField']:			
				outDir = r'/home/yjhu/theano/testData/Hid_recSPE_DBN_modeRecConstraint_1_MSE_hvSample_%dhid_gen_%s'%(layers,type)				 
				#dbn.GenHid(inDir,outDir,meanStdFile,log=True)	
				#dbn.HidDown(inDir,inDir+'_recSPE',meanStdFile)
				dbn.GenHidMeanReconstruction(inDir,outDir,meanStdFile,type)	
			'''
			
	elif work=='ana_syn':
		modelDir = r'/home/yjhu/theano/DBN/Model_new'
		DBNtitle = 'SLT_1000_GBBBB_hSample'	
		for layers in [1,2,3,4]:		
			layerSizes = [513]+[1024]*layers
			types = ['GB']+['BB']*(layers-1)
			
			dbn = DBN(DBNtitle,layerSizes,types,modelDir)		
			dbn.LoadModel()
			
			meanStdFile = r'/home/yjhu/theano/testData/mean_std.data'	
			inDir = r'/home/yjhu/theano/testData/SPE_test'	
			for type in ['DBC','Binary','MeanField']:			
				outDir = r'/home/yjhu/theano/testData/SPE_test_rec_DBN_theano_hSample_%dhid_gen_%s'%(layers,type)
				dbn.SPEupdown(inDir,outDir,meanStdFile,type)	 
		
	
def train():
	MSEType = 'mean'
	for MSEWeight in [0.2, 0.4, 0.6, 0.8]:
		inputDataDir = r'/home/yjhu/theano/data/logSPE_0Sil_normalization'
		
		DBNtitle = 'SLT_1000_GBBB_MSE_%s_%s_2nd'%(MSEType,str(MSEWeight).replace('.',''))

		modelDir = r'/home/yjhu/theano/DBN_comparison/Model/%s'%DBNtitle
		
		#inputDataDir = r'../data/Yanping13k_logSPE_20Sil_normalization'
		#modelDir = r'/home/yjhu/theano/DBN/Model/Yanping13k_GBBB'
		
		SaveMkdir(modelDir)		
		layerSizes = [513]+[1024]*3
		types = ['GB']+['BB']*2
		
		initialMomentum=0.9
		lr = [0.0001,0.0005,0.0005]
		batch_size = 20
		layerEpochs = 100
			
		log = modelDir+os.sep+DBNtitle+'_'+'_'.join(['%d'%x for x in layerSizes])+'.log'
		sys.stdout = Logger(log) 
		dbn = DBN_MSE(DBNtitle,layerSizes,types,modelDir,MSEType=MSEType)
		dbn.DBNTrain(inputDataDir,MSEWeight=MSEWeight,lr=lr,initialMomentum=initialMomentum,batch_size=batch_size,layerEpochs=layerEpochs)
		sys.stdout.flush()	
		
if __name__ == '__main__':	
	if sys.argv[1]=='test':		
		test()
	elif sys.argv[1]=='train':
		train()
	
	
	
	
	
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			


			