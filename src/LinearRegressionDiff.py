from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, Lasso
import numpy as np
import time
import pickle as pkl
import logging
#import matplotlib.pyplot as plt  
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import random
import shutil
from itertools import chain
import multiprocessing as mp
import os

class ScaleModule:
	def __init__(self, low = 25, high = 75, factor = 2.5, nomove = False):
		self.low = low
		self.high = high
		self.factor = factor
		self.nomove = nomove

	def fit(self, X, Findex):
		if self.nomove:
			return X
		
		self.idxdic = {}
		for i, j in enumerate(Findex):
			self.idxdic[j] = i
		l, h = np.percentile(X, [self.low, self.high], axis=0)
		self.lbound = l - (h - l) * self.factor
		self.ubound = h + (h - l) * self.factor

		X = np.maximum(X, self.lbound)
		X = np.minimum(X, self.ubound)
		self.mean = np.mean(X, axis = 0)
		self.std = np.maximum(np.std(X, axis = 0), 0.01)
		
	def forward(self, X, nFindex):
		if self.nomove or len(nFindex) == 0:
			return X
		idx = np.array([self.idxdic[i] for i in nFindex])

		lbound = self.lbound[idx]
		ubound = self.ubound[idx]
		mean = self.mean[idx]
		std = self.std[idx]

		X = np.maximum(X, lbound)
		X = np.minimum(X, ubound)
		X -= mean
		X /= std
		return X
		
class linearModel:
	def __init__(self, data, feature, args, cache, test=False):
		self.data = data
		self.feature = feature
		self.cache_prefix = cache
		self.test = test
		self.args = args

	def showFeatureName(self, fname, Findex):
		with open(fname, 'w') as f:
			for i in Findex:
				f.write(self.feature.feature_name[i] + '\n')
		logging.info("write feature to %s", fname)

	def fit(self):
		if self.test:
			train_Xindex = list(range(10000))
		else:
			train_Xindex = self.feature.filter(lambda n, a: "unknown" not in a and "recently" not in a and a[0] < 10312, None, "sample")
		
		all_Findex = list(range(len(self.feature.feature_name)))
		logging.info("X:%d", len(train_Xindex))
		logging.info("F:%d", len(all_Findex))

		self.importantYindex = []
		for i in chain(range(12), range(12, 24, 2), range(24, 48, 4), range(48, 72, 6)):
			s = "_%d" % i
			if self.test:
				self.importantYindex += self.feature.filter(lambda n, a: n[-len(s):]==s and "PM25" in n, None, "output")
			else:
				self.importantYindex += self.feature.filter(lambda n, a: n[-len(s):]==s, None, "output")
		logging.info("Y:%d", len(self.importantYindex))

		self.rimportantYindex = []
		for i in [3, 12, 24, 48]:
			s = "_%d" % i
			if self.test:
				self.rimportantYindex += self.feature.filter(lambda n, a: n[-len(s):]==s and "PM25" in n, None, "output")
			else:
				self.rimportantYindex += self.feature.filter(lambda n, a: n[-len(s):]==s, None, "output")

		# step 0: ignore extreme sample
		try:
		    train_Xindex = self.loadModule("XindexCache")
		except:
		    train_Xindex = self.ignoreExtremeSample(train_Xindex)
		    self.saveModule("XindexCache", train_Xindex)
		logging.info("X:%d", len(train_Xindex))
		logging.info("step0")

		self.train_Xindex = train_Xindex

		# step 1: fit a normalization model
		if not self.restoreFeatureNormalize():
			Xindex = self.randomSelect(train_Xindex, 6500)
			logging.info("startload")
			XFXY = self.passNone(Xindex, all_Findex)
			logging.info("stopload")
			self.fitFeatureNormalize(*XFXY)
			self.saveFeatureNormalize()
		logging.info("step1")

		# step 2: select feature by corre
		if not self.restoreSelectByCor():
			Xindex = self.randomSelect(train_Xindex, 6500)
			logging.info("startload")
			XFXY = self.passFeatureNormalize(Xindex, all_Findex)
			logging.info("stopload")
			Findex1 = self.fitSelectByCor(*XFXY)
			self.saveSelectByCor()
		else:
			Findex1 = self.selectByCor
		self.showFeatureName("./cache/findex1", Findex1)
		logging.info("F:%d", len(Findex1))
		logging.info("step2")
		self.Findex1 = Findex1

		# step 3: select feature by L1model
		if not self.restoreSelectByL1():
			if self.test:
				Xindex = self.randomSelect(train_Xindex, 50)
			else:
				Xindex = self.randomSelect(train_Xindex, 10000)
			logging.info("startload")
			XFXY = self.passSelectByCor(Xindex, Findex1)
			logging.info("stopload")
			Findex2 = self.fitSelectByL1(*XFXY)
			self.saveSelectByL1()
		else:
			Findex2 = self.selectByL1['c']
		self.showFeatureName('./cache/findex2', Findex2)
		logging.info("F:%d", len(Findex2))
		logging.info("step3")
		self.Findex2 = Findex2

		# step 4: PCA
		if not self.restorePCA():
			Xindex = self.randomSelect(train_Xindex, 20000)
			logging.info("startload")
			XFXY = self.passSelectByL1(Xindex, Findex2)
			logging.info("stopload")
			self.fitPCA(*XFXY)
			self.savePCA()
		logging.info("step4")

		# step 5: train
		#if not self.restoreModel():
		#	logging.info("startload")
		#	XFXY = self.passPCA(train_Xindex, Findex2)
		#	logging.info("stopload")
		#	#with open("./cache/orz",'wb') as f:
		#	#	pkl.dump(XFXY, f, protocol=4)
		#	#with open("./cache/orz",'rb') as f:
		#	#	XFXY = pkl.load(f)
		#	self.fitModel(*XFXY)
		#	self.saveModel()
		#logging.info("step5")

	def saveModule(self, modulename, dic):
		fname = self.cache_prefix + "_%s" % modulename
		with open(fname + ".tmp", 'wb') as f:
			pkl.dump(dic, f, protocol=4)
		shutil.copy(fname + ".tmp", fname)

	def loadModule(self, modulename):
		fname = self.cache_prefix + "_%s" % modulename
		with open(fname, 'rb') as f:
			return pkl.load(f)

	def randomSelect(self, idx, k):
		res = random.sample(idx, k=k)
		res.sort()
		return res

	def divide(self, idx, blockLen=5000):
		last = 0
		for i,j in enumerate(idx):
			if idx[last] // blockLen < j // blockLen:
				yield idx[last:i]
				last = i
		yield idx[last:]

	def ignoreExtremeSample(self, Xindex):
		feature = self.feature

		X_tmp = self.randomSelect(Xindex, 10000)
		_, _, _, Y, _ = self.passNone(X_tmp, [], True)
		YLow, YHigh = np.percentile(Y, [5, 95], axis=0)
		YIQR = YHigh - YLow

		res = []
		for now_idx in self.divide(Xindex):
			_, _, _, Y, _ = self.passNone(now_idx, [], True)
			YOut = ((Y - YHigh) / YIQR > 2.5) | ((YLow - Y) / YIQR > 2.5)
			YOut = np.sum(YOut, axis=1)
			for i, j in enumerate(now_idx):
				if YOut[i] <= 1:
					res.append(j)

		logging.info("ignore Extreme Sample: %d / %d", len(res), len(Xindex))
		
		return res

	def minusLast(self, lastX, Y):
		if self.args["location"] == "bj":
			Y1 = Y[:, :72] - lastX[:, 0:1]
			Y2 = Y[:, 72 : 72*2] - lastX[:, 1:2]
			Y3 = Y[:, 72*2 : 72*3] - lastX[:, 2:3]
			return np.concatenate([Y1, Y2, Y3], 1)
		else:
			Y1 = Y[:, :72] - lastX[:, 0:1]
			Y2 = Y[:, 72 : 72*2] - lastX[:, 1:2]
			return np.concatenate([Y1, Y2], 1)

	def addLast(self, lastX, Y):
		if self.args["location"] == "bj":
			Y1 = Y[:, :72] + lastX[:, 0:1]
			Y2 = Y[:, 72 : 72*2] + lastX[:, 1:2]
			Y3 = Y[:, 72*2 : 72*3] + lastX[:, 2:3]
			return np.concatenate([Y1, Y2, Y3], 1)
		else:
			Y1 = Y[:, :72] + lastX[:, 0:1]
			Y2 = Y[:, 72 : 72*2] + lastX[:, 1:2]
			return np.concatenate([Y1, Y2], 1)

	def passNone(self, xidx, fidx, checkValid=True):
		logging.info("load %d %d", len(xidx), len(fidx))
		assert(len(xidx) * len(fidx) < 50000 * 2000)
		
		if self.args["location"] == "bj":
			a = self.feature.feature_name.index("AQ_TS_o_PM25_1_mean")
			b = self.feature.feature_name.index("AQ_TS_o_PM10_1_mean")
			c = self.feature.feature_name.index("AQ_TS_o_O3_1_mean")
			X, Y = self.feature.select(xidx, fidx + [a, b, c], True, checkValid=checkValid)
			X, lastX = X[:, :-3], X[:, -3:]
			Y = self.minusLast(lastX, Y)
		else:
			assert self.args['location'] == 'ld'
			a = self.feature.feature_name.index("AQ_TS_o_PM25_1_mean")
			b = self.feature.feature_name.index("AQ_TS_o_PM10_1_mean")
			X, Y = self.feature.select(xidx, fidx + [a, b], True, checkValid=checkValid)
			X, lastX = X[:, :-2], X[:, -2:]
			Y = self.minusLast(lastX, Y)

		return xidx, fidx, X, Y, lastX

	def restoreFeatureNormalize(self):
		try:
			self.featureNormalizeCache = self.loadModule("FeatureNormalize")
			return True
		except:
			return False

	def fitFeatureNormalize(self, Xindex, Findex, X, Y, _):
		self.featureNormalizeCache = {}
		c = self.featureNormalizeCache
		c['scale'] = []

		feature = self.feature
		Y = None

		# label: don't adjust
		# diff\acce: scale min_max 25/75
		# WATS_seg: scale min_max 5/95
		labelFIndex = feature.filterIdx(lambda n,a : "label" in a, Findex)
		WATSFIndex = feature.filterIdx(lambda n,a : "W_ATS_" in n, Findex)
		normalFIndex = feature.filterIdx(lambda n, a: "label" not in a and "W_ATS_" not in n, Findex)
		
		for (FII, FI), mod in \
				zip([labelFIndex, WATSFIndex, normalFIndex], \
				[ScaleModule(None, None, None, True), ScaleModule(5, 95, 1.5), ScaleModule(25,75,2.5)]):
			#nX = feature.select(Xindex, FI)
			nX = X[:, FII]
			c['scale'].append(mod)
			mod.fit(nX, FI)

		logging.info("featureNormalize")

	def saveFeatureNormalize(self):
		self.saveModule("FeatureNormalize", self.featureNormalizeCache)

	def passFeatureNormalize(self, Xindex, Findex, **kwargs):
		Xindex, Findex, X, Y, lastX = self.passNone(Xindex, Findex, **kwargs)

		feature = self.feature
		c = self.featureNormalizeCache
		labelFIndex = feature.filterIdx(lambda n,a : "label" in a, Findex)
		WATSFIndex = feature.filterIdx(lambda n,a : "W_ATS_" in n, Findex)
		normalFIndex = feature.filterIdx(lambda n, a: "label" not in a and "W_ATS_" not in n, Findex)
		
		res = []
		resFI = []
		for i, ((FII, FI), mod) in \
				enumerate(zip([labelFIndex, WATSFIndex, normalFIndex], c['scale'])):
			nX = X[:, FII]
			sX = c['scale'][i].forward(nX, FI)
			res.append(sX)
			resFI += FI
		return Xindex, resFI, np.concatenate(res, 1), Y, lastX

	def restoreSelectByCor(self):
		try:
			self.selectByCor = self.loadModule("SelectByCor")
			return True
		except:
			return False

	def fitSelectByCor(self, Xindex, Findex, X, Y, _):
		def multivariate_pearsonr(X, y):
			scores, pvalues = [], []
			for column in range(X.shape[1]):
				cur_score, cur_p = pearsonr(X[:,column], y)
				scores.append(abs(cur_score))
				pvalues.append(cur_p)
			return (np.array(scores), np.array(pvalues))
		
		self.selectByCor = []        
		c = self.selectByCor
		feature = self.feature

		s = np.zeros((X.shape[1]), dtype=bool)
		
		label = feature.filter(lambda n,a : "label" in a, Findex)
		for i in label:
			s[Findex.index(i)] = True

		for j in self.importantYindex:
			scores, p = multivariate_pearsonr(X, Y[:, j])
			s |= (scores > self.args['corLowLimit'])
			logging.info("pearson %d", j)

		for i in range(X.shape[1]):
			if s[i]:
				c.append(Findex[i])

		logging.info("selectByCor: %d/%d" % (len(c), len(Findex)))
		return c

	def saveSelectByCor(self):
		self.saveModule("SelectByCor", self.selectByCor)

	def passSelectByCor(self, Xindex, Findex, **kwargs):
		Xindex, Findex, X, Y, lastX = self.passFeatureNormalize(Xindex, Findex, **kwargs)

		c = self.selectByCor

		res = []
		resFI = []
		for i, j in enumerate(Findex):
			if j in c:
				resFI.append(j)
				res.append(X[:,i])
		return Xindex, resFI, np.stack(res, 1), Y, lastX

	def restoreSelectByL1(self):
		try:
			self.selectByL1 = self.loadModule("SelectByL1")
			return True
		except:
			return False
	
	def fitSelectByL1(self, Xindex, Findex, X, Y, _):
		self.selectByL1 = {'c':[], 'lasso':[]}    
		
		c = self.selectByL1['c']
		feature = self.feature

		s = np.zeros((X.shape[1]), dtype=bool)

		logging.info("%s", X.shape)

		lassocv = LassoCV(normalize=True, n_jobs=10)
		lassocv.fit(X, Y[:, 0])
		alpha = lassocv.alpha_
		logging.info("lassocv ok")

		for yidx in self.rimportantYindex:
			lasso = Lasso(alpha=alpha, normalize=True, warm_start=True)
			lasso.fit(X, Y[:, yidx])
			logging.info("lasso %d", yidx)
			self.selectByL1['lasso'].append(lasso)
			#for i, j in enumerate(self.rimportantYindex):
			s |= (lasso.coef_ != 0)

		for i in range(X.shape[1]):
			if s[i]:
				c.append(Findex[i])

		logging.info("selectByL1: %d/%d" % (len(c), len(Findex)))
		return c
		
	def saveSelectByL1(self):
		self.saveModule("SelectByL1", self.selectByL1)

	def passSelectByL1(self, Xindex, Findex, **kwargs):
		Xindex, Findex, X, Y, lastX = self.passSelectByCor(Xindex, Findex, **kwargs)

		c = self.selectByL1['c']

		res = []
		resFI = []
		for i, j in enumerate(Findex):
			if j in c:
				resFI.append(j)
				res.append(X[:,i])
		return Xindex, resFI, np.stack(res, 1), Y, lastX

	def savePCA(self):
		self.saveModule("PCA", self.PCA)

	def restorePCA(self):
		try:
			self.PCA = self.loadModule("PCA")
			return True
		except:
			return False

	def fitPCA(self, Xindex, Findex, X, Y, _):
		self.PCA = PCA(n_components=500, whiten=True)
		X_new = self.PCA.fit_transform(X)
		logging.info("PCA oldX %s, newX %s", X.shape, X_new.shape)

	def passPCA(self, Xindex, Findex = None, **kwargs):
		if Findex is None:
			Findex = self.Findex2
		resX = []
		resY = []
		reslastX = []
		for xidx in self.divide(Xindex):
			Xi, _, X, Y, lastX = self.passSelectByCor(xidx, Findex, **kwargs)
			resX.append(self.PCA.transform(X))
			resY.append(Y)
			reslastX.append(lastX)
		return Xindex, None, np.concatenate(resX), np.concatenate(resY), np.concatenate(reslastX)

	def saveModel(self):
		self.saveModule("Model", self.model)

	def restoreModel(self):
		try:
			self.model = self.loadModule("Model")
			if len(self.model) == len(self.feature.output_name):
				return True
			else:
				return False
		except:
			return False

	def fitModel(self, Xindex, Findex, X, Y):
		if not hasattr(self, "model"):
			self.model = []

		if "ridge_alpha" in self.args:
			alpha = self.args['ridge_alpha']
		else:
			ridgecv = RidgeCV(alphas=(10., 50., 100.))
			ridgecv.fit(X, Y[:, 0])
			logging.info("Ridge cv %f", ridgecv.alpha_)
			logging.info("Ridge score %f", ridgecv.score(X, Y[:, 0]))
			alpha = ridgecv.alpha_

		global ridgeX, ridgeY, ridgeAlpha
		ridgeX = X
		ridgeY = Y
		ridgeAlpha = alpha

		for idx in self.divide(list(range(len(self.model), Y.shape[1])), 18):
			with mp.Pool(6) as pool:
				self.model += pool.map(train_ridge, idx)
			logging.info("Ridge group %d", idx[0])
			self.saveModel()
		
		logging.info("Ridge ok")

	def predictModel(self, Xindex, Findex, checkValid):
		Xindex, Findex, X, Y = self.passPCA(Xindex, Findex, checkValid=checkValid)
		res = []
		for m in self.model:
			res.append(m.predict(X))
		return Xindex, np.maximum(np.stack(res, 1), 0), Y

def train_ridge(i):
	global ridgeX, ridgeY, ridgeAlpha
	logging.info("Ridge %d start", i)
	ridge = Ridge(alpha = ridgeAlpha)
	ridge.fit(ridgeX, ridgeY[:, i])
	logging.info("Ridge %d end", i)
	return ridge

def train(data, feature, **config):
	return linearModel(data, feature, **config)