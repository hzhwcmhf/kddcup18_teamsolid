from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from LinearEnsemble import LinearEnsemble
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR
import multiprocessing as mp
import os
import pickle as pkl
import shutil
import logging
import numpy as np
import random
from utils import *

class FirstRegression:
	def __init__(self, data, feature, selection, selection2, args, cache):
		self.data = data
		self.feature = feature
		self.selection = selection
		self.selection2 = selection2
		self.args = args
		self.cache_prefix = cache

	def saveModule(self, modulename, ok):
		fname = self.cache_prefix + "_%s" % modulename
		dic = {"model":getattr(self, modulename), "ok":ok}
		with open(fname + ".tmp", 'wb') as f:
			pkl.dump(dic, f, protocol=4)
		shutil.copy(fname + ".tmp", fname)

	def loadModule(self, modulename):
		fname = self.cache_prefix + "_%s" % modulename
		if not os.path.exists(fname):
			return False
		with open(fname, 'rb') as f:
			dic = pkl.load(f)
		setattr(self, modulename, dic['model'])
		return dic["ok"]
	
	def divide(self, idx, blockLen=5000):
		last = 0
		for i,j in enumerate(idx):
			if idx[last] // blockLen < j // blockLen:
				yield idx[last:i]
				last = i
		yield idx[last:]

	def fit(self):
		dontcheck = "dontcheck" in self.args
		
		if not dontcheck:
			if "useorz" in self.args and self.loadModule("orz"):
				X1, Y1, X1_test, Y1_test, X2, Y2, X2_test, Y2_test, lastX2, lastX2_test = self.orz
			else:
				trainset1 = set(self.selection.train_Xindex)
				trainset2 = set(self.selection2.train_Xindex)
				XindexEnsemble = list(set(trainset1) | set(trainset2))
				random.shuffle(XindexEnsemble)
				XindexEnsemble = XindexEnsemble[:50000]

				trainset1 = trainset1 | set(XindexEnsemble)
				trainset2 = trainset2 | set(XindexEnsemble)
				trainlist1 = sorted(list(trainset1))
				trainlist2 = sorted(list(trainset2))
				traindict1 = dict([(j, i) for i, j in enumerate(trainlist1)])
				traindict2 = dict([(j, i) for i, j in enumerate(trainlist2)])

				train_index1 = sorted([traindict1[i] for i in (trainset1 - set(XindexEnsemble))])
				test_index1 = sorted([traindict1[i] for i in set(XindexEnsemble)])
				train_index2 = sorted([traindict2[i] for i in (trainset2 - set(XindexEnsemble))])
				test_index2 = sorted([traindict2[i] for i in set(XindexEnsemble)])

				_, _, X_all, Y_all = self.selection.passPCA(trainlist1)
				X1 = X_all[train_index1]
				Y1 = Y_all[train_index1]
				X1_test = X_all[test_index1]
				Y1_test = Y_all[test_index1]

				_, _, X_all, Y_all, lastX = self.selection2.passPCA(trainlist2)
				X2 = X_all[train_index2]
				Y2 = Y_all[train_index2]
				X2_test = X_all[test_index2]
				Y2_test = Y_all[test_index2]
				lastX2 = lastX[train_index2]
				lastX2_test = lastX[test_index2]
				self.orz = (X1, Y1, X1_test, Y1_test, X2, Y2, X2_test, Y2_test, lastX2, lastX2_test)
				self.saveModule("orz", True)
			logging.info("load data ok")

			logging.info("X1 %d X2 %d X1_test%d X2_test %d", X1.shape[0], X2.shape[0], X1_test.shape[0], X2_test.shape[0])
		self.models = []
		self.models2 = []

		if not self.loadModule("ridge"):
			self.fitRidge(X1, Y1)
			self.saveModule("ridge", True)
		self.models.append(self.passRidge)
		logging.info("ridge1 ok")

		if not self.loadModule("ridge2"):
			self.fitRidge(X2, Y2, "ridge2")
			self.saveModule("ridge2", True)
		self.models2.append(self.passRidge2)
		logging.info("ridge2 ok")

		if not self.loadModule("randomforest"):
			self.fitRF(X1, Y1)
			self.saveModule("randomforest", True)
		self.models.append(self.passRF)
		logging.info("randomforest ok")

		if not self.loadModule("randomforest2"):
			self.fitRF(X2, Y2, "randomforest2")
			self.saveModule("randomforest2", True)
		self.models2.append(self.passRF2)
		logging.info("randomforest2 ok")

		if not self.loadModule("SVR"):
			self.fitSVR(X1, Y1)
			self.saveModule("SVR", True)
		self.models.append(self.passSVR)
		logging.info("SVR ok")

		if not self.loadModule("SVR2"):
			self.fitSVR(X2, Y2, "SVR2", lastX = lastX2)
			self.saveModule("SVR2", True)
		self.models2.append(self.passSVR2)
		logging.info("SVR2 ok")
		
		self.ensemble = LinearEnsemble(self.cache_prefix + "_ensemble", len(self.models) + len(self.models2), 72*3 if self.args['location']=="bj" else 72*2)
		if not self.ensemble.restore():
			Y_p = []
			for m in self.models:
				Y_p.append(m(X1_test))
				logging.info("ensemble predict %s", m.__name__)
			for m in self.models2:
				Y_p.append(m(X2_test, lastX2_test))
				logging.info("ensemble predict %s", m.__name__)
			self.ensemble.fit(Y_p, Y1_test, 0.01 if self.args['location']=='bj' else 1e-4)
			self.ensemble.save()
		logging.info("ensemble ok")

	def predict(self, Xindex, checkValid=True):
		Xindex, Findex, X, Y = self.selection.passPCA(Xindex, checkValid=checkValid)
		Xindex, Findex, X2, Y2, lastX2 = self.selection2.passPCA(Xindex, checkValid=checkValid)

		Y_p = []
		for m in self.models:
			Y_p.append(m(X))
			logging.info("ensemble predict %s", m.__name__)
		for m in self.models2:
			Y_p.append(m(X, lastX2))
			logging.info("ensemble predict %s", m.__name__)
		Y_pred = self.ensemble.predict(Y_p)

		#Y_pred = self.passRF(X)

		return Xindex, Y_pred, Y

	def fitRidge(self, X, Y, name="ridge"):
		if not hasattr(self, name):
			ridge = []
			setattr(self, name, ridge)
		else:
			ridge = getattr(self, name)

		if "ridge_alpha" in self.args:
			alpha = self.args['ridge_alpha']
		else:
			ridgecv = RidgeCV(alphas=(10., 50., 100.))
			ridgecv.fit(X, Y[:, 24])
			logging.info("Ridge cv %f", ridgecv.alpha_)
			logging.info("Ridge score %f", ridgecv.score(X, Y[:, 24]))
			alpha = ridgecv.alpha_

		global ridgeX, ridgeY, ridgeAlpha
		ridgeX = X
		ridgeY = Y
		ridgeAlpha = alpha

		for idx in self.divide(list(range(len(ridge), Y.shape[1])), 18):
			with mp.Pool(6) as pool:
				ridge += pool.map(train_ridge, idx)
			logging.info("Ridge group %d", idx[0])
			self.saveModule(name, False)

		logging.info("Ridge ok")

	def passRidge(self, X):
		res = []
		for m in self.ridge:
			res.append(m.predict(X))
		return np.maximum(np.stack(res, 1), 0)

	def passRidge2(self, X, lastX):
		res = []
		for m in self.ridge2:
			res.append(m.predict(X))
		return np.maximum(self.selection2.addLast(lastX, np.stack(res, 1)), 0)

	def fitRF(self, X, Y, name="randomforest"):
		if not hasattr(self, name):
			randomforest = []
			setattr(self, name, randomforest)
		else:
			randomforest = getattr(self, name)

		Xselect = 5000
		Fselect = 50

		Xid = np.arange(Xselect)
		np.random.shuffle(Xid)
		Xid = Xid[:Xselect]

		#if "rf_leafnum" in self.args:
		#	leafnum = self.args['rf_leafnum']
		#else:
		#	sample_leaf_options = [0.01]
		#	bestscore = 0
		#	bestleaf = 25
		#	for leafnum in sample_leaf_options:
		#		model = RandomForestRegressor(n_estimators = 50, criterion='mae', oob_score = True, n_jobs = -1)
		#		model.fit(X, Y[:, 0])
		#		if model.oob_score_ > bestscore:
		#			bestscore = model.oob_score_ 
		#			bestleaf = leafnum
		#		logging.info("RandomForest try %d", leafnum)
		#	logging.info("RandomForest best leaf %d", bestleaf)

		global RFargs
		RFargs = (X[Xid][:, :Fselect], Y[Xid])
		logging.info("RF feature: %s", RFargs[0].shape)

		#RF = RandomForestRegressor(n_estimators = 50, criterion='mae', oob_score = True, n_jobs = 10)
		#RF.fit(X[Xid][:, :Fselect], Y[Xid])
		#self.randomforest = RF
		#self.saveModule("randomforest", False)

		for idx in self.divide(list(range(len(randomforest), Y.shape[1])), 18):
			with mp.Pool(3) as pool:
				randomforest += pool.map(train_RF, idx)
			logging.info("RandomForest group %d", idx[0])
			self.saveModule(name, False)

		logging.info("Randomforest ok")

	def passRF(self, X):
		res = []
		for m in self.randomforest:
			res.append(m.predict(X[:, :50]))
		return np.maximum(np.stack(res, 1), 0)

	def passRF2(self, X, lastX):
		res = []
		for m in self.randomforest2:
			res.append(m.predict(X[:, :50]))
		return np.maximum(self.selection2.addLast(lastX, np.stack(res, 1)), 0)

	def fitSVR(self, X, Y, name, lastX = None):
		if not hasattr(self, name):
			SVR = []
			setattr(self, name, SVR)
		else:
			SVR = getattr(self, name)

#		if "ridge_alpha" in self.args:
#			alpha = self.args['ridge_alpha']
#		else:
		epsilon_options = [0, 0.1, 10, 100]
		C_options = [0.1, 10, 100]

		Xselect = 30000

		kf = KFold(n_splits=5, shuffle=True)
		for i1, i2 in kf.split(X):
			train_index, test_index = i1[:Xselect], i2
			break

		bestscore = 3
		bestarg = None
		for epsilon in epsilon_options:
			for C in C_options:
				logging.info("SVR trying %f %f", epsilon, C)
				model = LinearSVR(epsilon=epsilon, C=C)
				model.fit(X[train_index], Y[train_index][:, 24])
				if lastX is None:
					predY = model.predict(X[test_index])
					score = calSMAPE1(Y[test_index][:, 24], predY)
				else:
					predY = lastX[test_index][:, 0] + model.predict(X[test_index])
					score = calSMAPE1(lastX[test_index][:, 0] + Y[test_index][:, 24], predY)
				if score < bestscore:
					bestscore = score
					bestarg = (epsilon, C)
				logging.info("SVR try %f %f, score %f", epsilon, C, score)
		epsilon, C = bestarg
		logging.info("SVR best %f %f, bestscore %f", epsilon, C, bestscore)

		global SVRargs
		SVRargs = (X[train_index], Y[train_index], epsilon, C)

		for idx in self.divide(list(range(len(SVR), Y.shape[1])), 18):
			with mp.Pool(6) as pool:
				SVR += pool.map(train_SVR, idx)
			logging.info("SVR group %d", idx[0])
			self.saveModule(name, False)

		logging.info("SVR ok")

	def passSVR(self, X):
		res = []
		for m in self.SVR:
			res.append(m.predict(X))
		return np.maximum(np.stack(res, 1), 0)

	def passSVR2(self, X, lastX):
		res = []
		for m in self.SVR2:
			res.append(m.predict(X))
		return np.maximum(self.selection2.addLast(lastX, np.stack(res, 1)), 0)

def train_ridge(i):
	global ridgeX, ridgeY, ridgeAlpha
	logging.info("Ridge %d start", i)
	ridge = Ridge(alpha = ridgeAlpha)
	ridge.fit(ridgeX, ridgeY[:, i])
	logging.info("Ridge %d end", i)
	return ridge

def train_RF(i):
	global RFargs
	X, Y = RFargs

	logging.info("RF %d start", i)
	RF = RandomForestRegressor(n_estimators = 50, criterion='mae', oob_score = True, n_jobs = 10)
	RF.fit(X, Y[:, i])
	logging.info("RF %d end", i)
	return RF

def train_SVR(i):
	global SVRargs
	X, Y, epsilon, C = SVRargs

	logging.info("SVR %d start", i)
	model = LinearSVR(epsilon=epsilon, C=C)
	model.fit(X, Y[:, i])
	logging.info("SVR %d end", i)
	return model