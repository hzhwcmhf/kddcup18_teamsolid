import pickle as pkl
import numpy as np
import os
import shutil
import logging

class FeatureLoader:
	blockLen = 5000

	def __init__(self, cache_prefix):
		self.cache_prefix = cache_prefix
		self.last_cache_name = []
		self.last_cache_data = []
		#self.dic = self.loadInfo()

	def shiftOut(self):
		filename = self.last_cache_name[0]
		data = self.last_cache_data[0]
		self.last_cache_name = self.last_cache_name[1:]
		self.last_cache_data = self.last_cache_data[1:]
		if data[1]:
			logging.info("shift out")
			with open(filename + ".tmp", 'wb') as f:
				pkl.dump(data[0], f, protocol=4)
			shutil.copy(filename + ".tmp", filename)

	def flush(self):
		while len(self.last_cache_name) != 0:
			self.shiftOut()
		logging.info("flush!!")

	def loadCache(self, filename):
		if filename not in self.last_cache_name:
			if len(self.last_cache_name) >= 5:
				self.shiftOut()
			with open(filename, 'rb') as f:
				self.last_cache_data.append((pkl.load(f), False))
			self.last_cache_name.append(filename)
		return self.last_cache_data[self.last_cache_name.index(filename)][0]
		
	def saveCache(self, filename, data):
		if filename not in self.last_cache_name:
			if len(self.last_cache_name) >= 5:
				self.shiftOut()
			self.last_cache_name.append(filename)
			self.last_cache_data.append(None)
		self.last_cache_data[self.last_cache_name.index(filename)] = (data, True)

	def saveInfo(self, dic):
		self.saveCache(self.cache_prefix + "_info", dic)
		#with open(self.cache_prefix + "_info.tmp", 'wb') as f:
		#	pkl.dump(dic, f, protocol=4)
		#shutil.copy(self.cache_prefix + "_info.tmp", self.cache_prefix + "_info")

	def loadInfo(self):
		return self.loadCache(self.cache_prefix + "_info")
		#with open(self.cache_prefix + "_info", 'rb') as f:
		#	return pkl.load(f)

	def loadBlock(self, block_id, idx, idf, s='X'):
		X, XValid = self.loadCache(self.cache_prefix + "_%s_%d" % (s,block_id))
		idx = list(map(lambda x: x - block_id * self.blockLen, idx))
		print("load block %d" % block_id)
		return X[np.ix_(idx, idf)], XValid[np.ix_(idx, idf)]

	def load(self, idx, idf, s='X'):
		idx = list(idx)
		idf = np.array(list(idf))
		sorte = all(idx[i] <= idx[i+1] for i in range(len(idx)-1))
		assert sorte
		X = np.zeros((len(idx), len(idf)))
		XValid = np.zeros((len(idx), len(idf)))

		last = 0
		for i,j in enumerate(idx):
			if idx[last] // self.blockLen < j // self.blockLen:
				X[last:i], XValid[last:i] = self.loadBlock(idx[last] // self.blockLen, idx[last:i], idf, s = s)
				last = i
		X[last:], XValid[last:] = self.loadBlock(idx[last] // self.blockLen, idx[last:], idf, s = s)
		
		return X, XValid

	def saveBlock(self, block_id, idx, idf, newX, newXValid, s='X'):
		cache_name = self.cache_prefix + "_%s_%d" % (s, block_id)
		if os.path.exists(cache_name):
			X, XValid = self.loadCache(self.cache_prefix + "_%s_%d" % (s,block_id))
		else:
			X = np.zeros((0,0))
			XValid = np.ones((0,0), dtype=bool)
		idx = list(map(lambda x: x - block_id * self.blockLen, idx))
		if max(idx) >= X.shape[0] or max(idf) >= X.shape[1]:
			Xn = np.zeros((max(idx[-1]+1, X.shape[0]), max(idf + [X.shape[1]-1]) + 1))
			XValidn = np.ones((max(idx[-1]+1, X.shape[0]), max(idf + [X.shape[1]-1]) + 1), dtype=bool)
			Xn[:X.shape[0], :X.shape[1]] = X
			XValidn[:X.shape[0], :X.shape[1]] = XValid
			X = Xn
			XValid = XValidn
		X[np.ix_(idx, idf)] = newX
		XValid[np.ix_(idx, idf)] = newXValid.astype(bool)
		self.saveCache(cache_name, [X, XValid])
		#with open(cache_name + ".tmp", 'wb') as f:
		#    pkl.dump([X, XValid], f, protocol=4)
		#shutil.copy(cache_name + ".tmp", cache_name)
		print("save block %d" % block_id)

	def save(self, idx, idf, X, XValid=None, s='X'):
		idx = list(idx)
		idf = list(idf)
		sorte = all(idx[i] <= idx[i+1] for i in range(len(idx)-1))
		assert sorte

		if XValid is None:
			XValid = np.ones((X.shape[0], X.shape[1]), dtype=bool)

		last = 0
		for i,j in enumerate(idx):
			if idx[last] // self.blockLen < j // self.blockLen:
				#print(i)
				#print(last)
				#print(X[last:i].shape)
				self.saveBlock(idx[last] // self.blockLen, idx[last:i], idf, X[last:i], XValid[last:i], s=s)
				last = i
		self.saveBlock(idx[last] // self.blockLen, idx[last:], idf, X[last:], XValid[last:], s=s)

	def delBlock(self, idx, s='X'):
		block_id = idx // self.blockLen
		cache_name = self.cache_prefix + "_%s_%d" % (s, block_id)
		X, XValid = self.loadCache(self.cache_prefix + "_%s_%d" % (s,block_id))
		idx -= self.blockLen * block_id
		Xn = np.zeros((idx, X.shape[1]))
		XValidn = np.ones((idx, X.shape[1]), dtype=bool)
		Xn[:idx] = X[:idx]
		XValidn[:idx] = XValid[:idx]
		
		self.saveCache(cache_name, [Xn, XValidn])
		print("del block %d" % block_id)