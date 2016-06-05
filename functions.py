from tools import *
import re
from multiprocessing import Pool
import pdb

def RmSilence(AlignmentDir,SpeDir,spedim,SpeDir_dropSil,ratio):	
	
	SaveMkdir(SpeDir_dropSil)
	list = os.listdir(AlignmentDir)
	for file in list:
		name = file.split('.')[0]	
		spefile = SpeDir+os.sep+name+'.lspe'
		labelfile = AlignmentDir+os.sep+file
		spe_dropSil_file = SpeDir_dropSil+os.sep+name+'.dat'
		
		print 'Removing silence: file %s'%name
		
		flabel = open(labelfile,'r')
		silPos = []
		while True:
			line = flabel.readline()
			if len(line)<5:
				break
			parts = line.split(' ')
			start = round(int(parts[0])/50000.0)
			end = round(int(parts[1])/50000.0)
			if re.findall(r'-pau+',parts[2])==[]:
				continue
			silPos.append([start,end])
		flabel.close()
		
		rmSilPos = []
		for pos in silPos:
			for i in range(pos[0],pos[1]):
				if random.random()<=ratio:
					rmSilPos.append(i)
					
		speData = ReadFloatRawMat(spefile,spedim)
		newSpeData = np.delete(speData,rmSilPos,0)
		WriteArrayFloat(spe_dropSil_file,newSpeData)

def RmSilence2(AlignmentDir,SpeDir, spedim, dataDir,dataDim,ratio):	
	SpeDir_dropSil = SpeDir+'_dropSil%d'%(int(ratio*100))
	data_dropSil = dataDir+'_dropSil%d'%(int(ratio*100))
	SaveMkdir(SpeDir_dropSil)
	SaveMkdir(data_dropSil)
	list = os.listdir(AlignmentDir)
	for file in list:
		name = file.split('.')[0]	
		spefile = SpeDir+os.sep+name+'.dat'
		labelfile = AlignmentDir+os.sep+file
		spe_dropSil_file = SpeDir_dropSil+os.sep+name+'.dat'
		
		print 'Removing silence: file %s'%name
		
		flabel = open(labelfile,'r')
		silPos = []
		while True:
			line = flabel.readline()
			if len(line)<5:
				break
			parts = line.split()
			start = int(round(int(parts[0])/50000.0))
			end = int(round(int(parts[1])/50000.0))
			if re.findall(r'-sil+',parts[2])==[]:
				continue
			silPos.append([start,end])
		flabel.close()
		
		rmSilPos = []
		for pos in silPos:
			for i in range(pos[0],pos[1]):
				if random.random()<=ratio:
					rmSilPos.append(i)
		
		######## removing silence from two files
		dataFile = dataDir+os.sep+name+'.dat'
		data_dropSil_file = data_dropSil+os.sep+name+'.dat'
		data = ReadFloatRawMat(dataFile,dataDim)
		newData = np.delete(data,rmSilPos,0)
		WriteArrayFloat(data_dropSil_file,newData)		
		
		speData = ReadFloatRawMat(spefile,spedim)
		newSpeData = np.delete(speData,rmSilPos,0)
		WriteArrayFloat(spe_dropSil_file,newSpeData)

def RmSilence2_Parallel(AlignmentDir,SpeDir, spedim, dataDir,dataDim,ratio, cores = 5):
	SpeDir_dropSil = SpeDir+'_dropSil%d'%(int(ratio*100))
	data_dropSil = dataDir+'_dropSil%d'%(int(ratio*100))
	SaveMkdir(SpeDir_dropSil)
	SaveMkdir(data_dropSil)	
	print 'Removing silence'
	argList = []
	for file in os.listdir(AlignmentDir):
		name = file.split('.')[0]	
		spefile = SpeDir+os.sep+name+'.dat'
		labelfile = AlignmentDir+os.sep+file
		spe_dropSil_file = SpeDir_dropSil+os.sep+name+'.dat'		
		dataFile = dataDir+os.sep+name+'.dat'
		data_dropSil_file = data_dropSil+os.sep+name+'.dat'
		
		argList.append([labelfile,spefile, spedim, dataFile,dataDim,spe_dropSil_file,data_dropSil_file,ratio])
	pool = Pool(cores)
	pool.map(RmSilence2SingleFile,argList)
		
def RmSilence2SingleFile(args):
	[labelfile,spefile, spedim, dataFile,dataDim,spe_dropSil_file,data_dropSil_file,ratio] = args
	flabel = open(labelfile,'r')
	silPos = []
	while True:
		line = flabel.readline()
		if len(line)<5:
			break
		parts = line.split()
		start = int(round(int(parts[0])/50000.0))
		end = int(round(int(parts[1])/50000.0))
		if re.findall(r'-sil+',parts[2])==[]:
			continue
		silPos.append([start,end])
	flabel.close()
	
	rmSilPos = []
	for pos in silPos:
		for i in range(pos[0],pos[1]):
			if random.random()<=ratio:
				rmSilPos.append(i)
	
	######## removing silence from two files	
	data = ReadFloatRawMat(dataFile,dataDim)
	newData = np.delete(data,rmSilPos,0)
	WriteArrayFloat(data_dropSil_file,newData)		
	
	speData = ReadFloatRawMat(spefile,spedim)
	newSpeData = np.delete(speData,rmSilPos,0)
	WriteArrayFloat(spe_dropSil_file,newSpeData)
		
def Normalization(datadir,dim,meanFile=None):
	list = os.listdir(datadir)	
	if meanFile==None:
		filenum = len(list)
		mean_std = np.zeros([2,dim],dtype=np.float64)
		file_mean = np.zeros([filenum,dim+1],dtype=np.float64)
		file_std = np.zeros([filenum,dim+1],dtype=np.float64)
		for i in range(len(list)):		
			file = datadir+os.sep+list[i]
			data = ReadFloatRawMat(file,dim)
			file_mean[i][0] = data.shape[0]
			file_std[i][0] = data.shape[0]		
			file_mean[i][1:] = np.mean(data,0)
			file_std[i][1:] = np.mean(data**2,0)
		
		file_sum = (file_mean[:,0]*file_mean[:,1:].T).T 
		file_ssum = (file_std[:,0]*file_std[:,1:].T).T 
		
		mean_std[0] = np.sum(file_sum,0) / np.sum(file_mean[:,0])	
		mean_std[1] = np.sqrt(np.sum(file_ssum,0)/ np.sum(file_mean[:,0]) - mean_std[0]**2)
	else:
		mean_std = ReadFloatRawMat(meanFile,dim)
		
	outdir = datadir+'_normalization'	
	SaveMkdir(outdir)	
	WriteArrayFloat(datadir+'\\mean_std.data',mean_std)
	
	for line in list:
		print 'normalizing: file %s'%line
		infile = datadir+os.sep+line
		outfile = outdir+os.sep+line.split('.')[0]+'.dat'
		indata = ReadFloatRawMat(infile,dim)
		nframes = indata.shape[0]
		outdata = (indata - np.tile(mean_std[0],(nframes,1)))/(np.tile(mean_std[1],(nframes,1)))
		WriteArrayFloat(outfile,outdata)

def Normalization_nDim(datadir,dim,norm_dim,meanFile=None):
	list = os.listdir(datadir)	
	if meanFile==None:
		filenum = len(list)
		mean_std = np.zeros([2,dim],dtype=np.float64)
		file_mean = np.zeros([filenum,dim+1],dtype=np.float64)
		file_std = np.zeros([filenum,dim+1],dtype=np.float64)
		for i in range(len(list)):		
			file = datadir+os.sep+list[i]
			data = ReadFloatRawMat(file,dim)
			file_mean[i][0] = data.shape[0]
			file_std[i][0] = data.shape[0]		
			file_mean[i][1:] = np.mean(data,0)
			file_std[i][1:] = np.mean(data**2,0)
		
		file_sum = (file_mean[:,0]*file_mean[:,1:].T).T 
		file_ssum = (file_std[:,0]*file_std[:,1:].T).T 
		
		mean_std[0] = np.sum(file_sum,0) / np.sum(file_mean[:,0])	
		mean_std[1] = np.sqrt(np.sum(file_ssum,0)/ np.sum(file_mean[:,0]) - mean_std[0]**2)
	else:
		mean_std = ReadFloatRawMat(meanFile,dim)
		
	outdir = datadir+'_normalization'	
	SaveMkdir(outdir)	
	WriteArrayFloat(datadir+'\\mean_std.data',mean_std)
	
	for line in list:
		print 'normalizing: file %s'%line
		infile = datadir+os.sep+line
		outfile = outdir+os.sep+line.split('.')[0]+'.dat'
		indata = ReadFloatRawMat(infile,dim)
		nframes = indata.shape[0]
		outdata = indata		
		outdata[:,norm_dim] = (indata[:,norm_dim] - np.tile(mean_std[0][norm_dim],(nframes,1)))/(np.tile(mean_std[1][norm_dim],(nframes,1)))
		WriteArrayFloat(outfile,outdata)		

def CheckData(dataDir,dim):
	flag = True
	for name in os.listdir(dataDir):
		file = dataDir+os.sep+name
		data = ReadFloatRawMat(file,dim)
		yes = np.all(data==data)
		if yes==False:
			flag=False
			print 'some data is nan in %s'%file
	if flag==True:
		print 'Data is fine'
	else:
		print 'Data is wrong'
		
def CalDynamics(staticDir,dynamicDic,featDim):		
	SaveMkdir(dynamicDic)
	for i in sorted(os.listdir(staticDir)):
		name = i.split('.')[0]
		print 'Calculate dynamic feature, file %s'%name		
		logSPEFile = staticDir+os.sep+name+'.dat'
		logSPEDynamicFile = dynamicDic+os.sep+name+'.dat'
		
		data = ReadFloatRawMat(logSPEFile,featDim)
		dData = np.zeros(data.shape)
		ddData = np.zeros(data.shape)
		
		win1 = [-0.5,0,0.5]
		win2 = [0.25,-0.5,0.25]		
		for k in range(data.shape[1]):
			dData[:,k] = np.convolve(data[:,k],np.flipud(win1),mode='same')
			ddData[:,k] = np.convolve(data[:,k],np.flipud(win2),mode='same')
		dData[0] = dData[1]
		ddData[0] = ddData[1]
		dData[-1] = dData[-2]
		ddData[-1] = ddData[-2]	

		dynamicData = np.concatenate((data,dData,ddData),axis=1)
		
		WriteArrayFloat(logSPEDynamicFile,dynamicData)

def CalMeanVar_singleFile(dataDir,featDim):
	MeanDir = dataDir+'_meanVar'
	SaveMkdir(MeanDir)	
	for file in sorted(os.listdir(dataDir)):
		print file
		infile = dataDir+os.sep+file
		outfile = MeanDir+os.sep+file
		data = ReadFloatRawMat(infile,featDim)	
		mean = np.mean(data,0)
		var = np.std(data,0)**2
		WriteArrayFloat(outfile,np.array([mean,var]))

def MLPG(genDir,dim):
	tool = r'E:\yjhu\tools\tools\bin\mlpg_mv.exe'
	winfile = r'F:\yjhu2\2016\DBC_Baseline\win\window_d\mcep_d1.win'
	accwinfile = r'F:\yjhu2\2016\DBC_Baseline\win\window_d\mcep_d2.win'
	
	for file in sorted(os.listdir(genDir)):
		if not file.endswith('.pcp'):continue		
		name = file.split('.')[0]
		print name
		meanFile = genDir+os.sep+name+'.mean'
		stdFile = genDir+os.sep+name+'.var'
		outfile = genDir+os.sep+name+'.gh'
		
		data = ReadFloatRawMat(genDir+os.sep+file,dim*6)
		WriteArrayDouble(meanFile,data[:,0:dim*3])
		WriteArrayDouble(stdFile,data[:,dim*3:])
		
		cmd = '%s -din -order %d -dynwinf %s -accwinf %s %s %s %s'%(tool,dim,winfile,accwinfile,meanFile,stdFile,outfile);
		os.system(cmd)
		os.remove(meanFile)
		os.remove(stdFile)

def EvaluateLSD_Normalized(inDir,refDir,dim,lowF=0,highF=8000):
	frameCount = 0
	LSDSum = 0
	start = lowF/16000.*1024
	end = highF/16000.*1024+1
	
	count = 0
	list = os.listdir(refDir)
	for file in list:
		refFile = refDir+os.sep+file
		genFile = inDir+os.sep+file
		
		if not os.path.exists(genFile):break
		
		data = ReadFloatRawMat(genFile,dim)[:,start:end]
		refData = ReadFloatRawMat(refFile,dim)[:,start:end]
		
		if not data.shape==refData.shape:
			print data.shape
			print refData.shape
			print 'data mismatch %s'%file
			exit()	
		

		data = data/np.tile(np.sum(data,1),[data.shape[1],1]).T
		refData = refData/np.tile(np.sum(refData,1),[refData.shape[1],1]).T				
			
		frameCount = frameCount + data.shape[0]	
		#current_LSD = np.mean(np.sqrt(np.mean((10*np.log10(data/refData))**2,1)))
		sumDistance = np.sum((10*np.log10(data/refData))**2)
		LSDSum = LSDSum + sumDistance
		
		print 'LSD %s: %f'%(file,np.sqrt(sumDistance/data.shape[0]/data.shape[1]))		
		count+=1
		#if count>=100:break
		
	LSD = np.sqrt(LSDSum/frameCount/data.shape[1])	
	print 'Average Log Spectral Distortion is : %f, total frame: %d'%(LSD,frameCount)	

		
if __name__=='__main__':
	#CheckData(r'F:\yjhu2\2016\Yanping_13k\DNN\data\DNNOutput_mcep_lf0_uv_dropSil80_normalization',127)
	
	inDir = r'\\172.16.45.36\yjhu\theano\testData\SPE_test_rec_CNN_test'
	refDir = r'\\172.16.45.36\yjhu\theano\testData\SPE_test'
	EvaluateLSD_Normalized(inDir,refDir,513)








	