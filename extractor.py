import numpy as np

# Used to extract data from the main creditcard.csv file.
class Extractor:
	
	def __init__(self, addr):
		self.dataFile = addr
		self.basedata = np.genfromtxt(addr, delimiter=',', skip_header=1, dtype=None, encoding=None)

	def getbasedata(self):
		return self.basedata

	def getdata(self):
		return self.data

	def savedata(self, fname):
		addr = 'data/'+fname
		np.save(addr, self.data)

	@staticmethod
	def loaddata(fname):
		addr = 'data/'+fname+'.npy'
		return np.load(addr)



	def fnfsplit(self, fraudpercent):
		frauds = []
		nonfrauds = []
		for record in self.basedata:
			if record[-1] == "\"0\"":
				nonfrauds.append(record)
			else:
				frauds.append(record)


		fratio = fraudpercent/(1-fraudpercent)
		nonfraudnum = (int)(fratio*(len(frauds)))

		np.random.shuffle(nonfrauds)
		nonfrauds = nonfrauds[:nonfraudnum]
		finaldata = np.concatenate((frauds, nonfrauds), axis=0)
		np.random.shuffle(finaldata)

		self.data = finaldata