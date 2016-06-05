import os,sys
import numpy as np
#import matplotlib.pyplot as plt
import random,struct

class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, 'wt')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)  
        self.log.flush()		
	
    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
		self.log.close()
		sys.stdout=self.terminal
		pass    

def SaveMkdir(dir):
	if not os.path.exists(dir):
		os.mkdir(dir)

def Binary(data):
	outdata=data.copy()
	outdata[outdata>=0.5]=1.
	outdata[outdata<0.5]=0.
	return outdata

def ReadRBMModel(RBMFile,numdim,numhid):
	rbm_data = np.fromfile(RBMFile,dtype=np.float32)
	tvishid = rbm_data[0:numdim*numhid]
	thidbias = rbm_data[numdim*numhid:numdim*numhid+numhid]
	tvisbias = rbm_data[numdim*numhid+numhid:]
	tvishid.shape = [numhid,numdim]
	tvishid = tvishid.T
	return [tvishid.copy(),tvisbias.copy(),thidbias.copy()]

def CheckEXT(dir):
	exts = {}
	for line in os.listdir(dir):
		if not '.' in line: continue
		ext = line.split('.')[-1]
		if ext in exts:
			exts[ext]+=1
		else:
			exts[ext]=1	
	return '.'+sorted(exts,key=exts.get)[-1]
	
def ReadDNN(DNNfile,layernum):
	newDNN = []
	fdnn = open(DNNfile,'rb')		
	for i in range(layernum):
		stat = struct.unpack('5i',fdnn.read(20))	
		head = struct.unpack('%ds'%stat[4],fdnn.read(stat[4]))
		tmp = struct.unpack('%df'%(stat[2]*stat[1]),fdnn.read(stat[2]*stat[1]*4))
		weights = np.float64(tmp)
		weights.shape = stat[2],stat[1]
		
		stat = struct.unpack('5i',fdnn.read(20))	
		head = struct.unpack('%ds'%stat[4],fdnn.read(stat[4]))
		tmp = struct.unpack('%df'%(stat[2]*stat[1]),fdnn.read(stat[2]*stat[1]*4))
		bias = np.float64(tmp)
		
		newDNN.append([weights.copy(),bias.copy()])	
	fdnn.close()
	return newDNN

def ReadFloatRawMat(datafile,column):
	data = np.fromfile(datafile,dtype=np.float32)
	if len(data)%column!=0:
		print 'ReadFloatRawMat %s, column wrong!'%datafile
		exit()
	data.shape = [len(data)/column,column]
	return np.float64(data)

def ReadDoubleRawMat(datafile,column):
	data = np.fromfile(datafile,dtype=np.float64)
	if len(data)%column!=0:
		print 'ReadDoubleRawMat %s, column wrong!'%datafile
		exit()
	data.shape = [len(data)/column,column]
	return data.copy()

def WriteArrayFloat(file,data):
	tmp=np.array(data,dtype=np.float32)
	tmp.tofile(file)

def WriteArrayDouble(file,data):
	tmp=np.array(data,dtype=np.float64)
	tmp.tofile(file)


def makelist(outdir,listfile):
	list = os.listdir(outdir)
	random.shuffle(list)
	outlist = open(listfile,'w')
	for name in list:
		filepath = os.path.join(outdir,name)+'\n'
		outlist.write(filepath)
	outlist.close()
'''	
def PlotFile(file,dim):
	data = ReadFloatRawMat(file,dim)
	for i in range(100,data.shape[0]):
		plt.plot(data[i])
		plt.show()
		
def PlotData(data,time):		
	for i in range(time):		
		plt.plot(data[i])
		plt.show()

def PlotHist(indata):
	(n,bins,patches) = plt.hist(indata.flatten())
	plt.plot(bins)
	plt.axis([0,1,0,max(n) * 1.1])
	plt.show()
'''	
if __name__=='__main__':
	print sys.argv[1],int(sys.argv[2])
	PlotFile(sys.argv[1],int(sys.argv[2]))	
	
