import numpy as np
import math
import logging
import os
from utils import *
from itertools import chain
import pickle as pkl
import shutil
from FeatureLoader import FeatureLoader

class FeatureGroup:
	feature_name = None
	feature_attr = None
	data = None

	def __init__(self):
		self.feature_name = []
		self.feature_attr = []
		self.data = []

	def items(self):
		return self.feature_name, self.feature_attr, self.data

	def assertSize(self):
		assert len(self.feature_attr) == len(self.feature_name)
		assert self.data.shape[1] == len(self.feature_name)

	@staticmethod 
	def mergeFeature(features):
		res = FeatureGroup()
		a, b, c = zip(*map(lambda fg: fg.items(), features))
		res.feature_name = list(chain(*a))
		res.feature_attr = list(chain(*b))
		res.data = np.concatenate(c, 1)
		return res

	@staticmethod
	def mergeData(features):
		res = FeatureGroup()
		res.feature_name = features[0].feature_name
		res.feature_attr = features[0].feature_attr
		c = list(map(lambda fg: fg.data, features))
		res.data = np.concatenate(c, 0)
		return res

	# data = [feature_1, feature_2, ...]
	def assertFeatureArray(self):
		assert len(self.feature_attr) == len(self.feature_name)
		assert len(self.data) == len(self.feature_name)

	@staticmethod
	def mergeFeatureArray(features):
		res = FeatureGroup()
		res.feature_name, res.feature_attr, res.data = map(lambda l: list(chain(*l)), zip(*map(lambda fg: fg.items(), features)))
		return res

class Feature:
	feature_name = None
	feature_attr = None
	sample_name = None
	sample_attr = None
	output_name = None
	output_attr = None

	def addFeature(self, new_fname, new_fattr):
		assert len(new_fname) == len(new_fattr)
		assert new_fname[0] not in self.feature_name

		self.feature_name += new_fname
		self.feature_attr += new_fattr
		self.saveInfo()
		#new_flen = len(new_fname)
		#self.X = np.concatenate([self.X, np.zeros((self.X.shape[0], new_flen))], 1)
		#self.XValid = np.concatenate([self.XValid, np.ones((self.XValid.shape[0], new_flen))], 1)

	def addOutput(self, new_oname, new_oattr):
		assert len(new_oname) == len(new_oattr)
		assert new_oname[0] not in self.output_name

		self.output_name += new_oname
		self.output_attr += new_oattr
		self.saveInfo()
		#new_olen = len(new_oname)
		#self.Y = np.concatenate([self.Y, np.zeros((self.Y.shape[0], new_olen))], 1)
		#self.YValid = np.concatenate([self.YValid, np.ones((self.YValid.shape[0], new_olen))], 1)
	def updateOutput(self, oname, oattr):
		a = self.output_name.index(oname[0])
		assert self.output_name[a + len(oname) - 1] == oname[-1]
		for i, attr in enumerate(oattr):
			self.output_attr[i+a] = attr
		self.saveInfo()

	def addSample(self, new_sname, new_sattr):
		assert len(new_sname) == len(new_sattr)
		assert new_sname[0] not in self.sample_name

		self.sample_name += new_sname
		self.sample_attr += new_sattr
		self.saveInfo()

		#a = len(self.sample_name)
		#slen = len(new_sname)
		#flen = len(self.feature_name)
		#olen = len(self.output_name)
		#self.fl.save(range(a, a+slen), range(flen), np.zeros((slen, flen)), np.ones((slen, flen), dtype=bool))
		#self.fl.save(range(a, a+slen), range(olen), np.zeros((slen, olen)), np.ones((slen, olen), dtype=bool), s='Y')
		#new_slen = len(new_sname)
		#self.X = np.concatenate([self.X, np.zeros((new_slen, self.X.shape[1]))], 0)
		#self.XValid = np.concatenate([self.XValid, np.ones((new_slen, self.XValid.shape[1]))], 0)
		#self.Y = np.concatenate([self.Y, np.zeros((new_slen, self.Y.shape[1]))], 0)
		#self.YValid = np.concatenate([self.YValid, np.ones((new_slen, self.YValid.shape[1]))], 0)
	def updateSample(self, sname, sattr):
		a = self.sample_name.index(sname[0])
		if len(sname) + a > len(self.sample_name):
			slen = len(self.sample_name) - a
			flen = len(self.feature_name)
			olen = len(self.output_name)
			self.fl.save(range(a, a+slen), range(flen), np.zeros((slen, flen)), np.ones((slen, flen), dtype=bool))
			self.fl.save(range(a, a+slen), range(olen), np.zeros((slen, olen)), np.ones((slen, olen), dtype=bool), s='Y')

			self.sample_name = self.sample_name[:a]
			self.sample_attr = self.sample_attr[:a]
			self.addSample(sname, sattr)
		else:
			assert self.sample_name[a + len(sname) - 1] == sname[-1]
			for i, attr in enumerate(sattr):
				self.sample_attr[i+a] = attr
			self.saveInfo()

	def updateX(self, sname, fname, newX):
		slen = newX.shape[0]
		flen = newX.shape[1]
		a = self.sample_name.index(sname)
		b = self.feature_name.index(fname)
		self.fl.save(list(range(a, a+slen)), list(range(b, b+flen)), newX, np.zeros((slen, flen), dtype=bool))
		#self.X[a:a+slen, b:b+flen] = newX
		#self.XValid[a:a+slen, b:b+flen] = 0

	def updateY(self, sname, oname, newY, newYValid = None):
		slen = newY.shape[0]
		olen = newY.shape[1]
		a = self.sample_name.index(sname)
		b = self.output_name.index(oname)
		if newYValid is None:
			newYValid = np.zeros((slen, olen), dtype=bool)
		self.fl.save(list(range(a, a+slen)), list(range(b, b+olen)), newY, newYValid, s='Y')
		#self.Y[a:a+slen, b:b+olen] = newY
		#self.YValid[a:a+slen, b:b+olen] = 0

	def checkX(self, sname, fname):
		if sname not in self.sample_name or fname not in self.feature_name:
			return False
		a = self.sample_name.index(sname)
		b = self.feature_name.index(fname)
		try:
			X, XValid = self.fl.load([a], [b])
		except (IndexError, FileNotFoundError):
			return False
		return XValid[0, 0] < 0.5

	def checkY(self, sname, oname):
		if sname[0] not in self.sample_name or oname[0] not in self.output_name:
			return False
		a = self.sample_name.index(sname[0])
		b = self.output_name.index(oname[0])
		try:
			Y, YValid = self.fl.load(list(range(a, a+len(sname))), [b], s='Y')
		except (IndexError, FileNotFoundError):
			return False
		for i in range(Y.shape[0]):
			if Y[i][0] == 0:
				YValid[i][0] = 1
		return np.sum(YValid) < 0.5

	def saveInfo(self):
		self.fl.saveInfo([self.feature_name, self.feature_attr, self.sample_name, self.sample_attr, self.output_name, self.output_attr, self.time_block])
	
	def restoreInfo(self):
		#self.feature_name, self.feature_attr, self.sample_name, self.sample_attr, self.output_name, self.output_attr = \
		#	self.fl.loadInfo()
		self.feature_name, self.feature_attr, self.sample_name, self.sample_attr, self.output_name, self.output_attr, self.time_block = \
			self.fl.loadInfo()

	def filter(self, fn, idx = None, s = "feature"):
		if s == "feature":
			name = self.feature_name
			attr = self.feature_attr
		elif s == 'sample':
			name = self.sample_name
			attr = self.sample_attr
		elif s == 'output':
			name = self.output_name
			attr = self.output_attr
		if idx is None:
			idx = range(len(name))
		return list(filter(lambda i: fn(name[i], attr[i]), idx))

	def filterIdx(self, fn, idx, s = 'feature'):
		if s == "feature":
			name = self.feature_name
			attr = self.feature_attr
		elif s == 'sample':
			name = self.sample_name
			attr = self.sample_attr
		elif s == 'output':
			name = self.output_name
			attr = self.output_attr
		if idx is None:
			idx = range(len(name))
		resi = []
		res = []
		for j, i in enumerate(idx):
			if fn(name[i], attr[i]):
				resi.append(j)
				res.append(i)
		return resi, res

	def select(self, Xindex = None, Findex = None, Oindex = None, checkValid=True):
		if Xindex is None:
			Xindex = list(range(len(self.sample_name)))
		if Findex is None:
			Findex = list(range(len(self.feature_name)))
		if len(Xindex) > 0:
			X, XValid = self.fl.load(Xindex, Findex)
			if checkValid:
				assert np.sum(XValid) < 1
		else:
			X = None
		if Oindex is None:
			return X
		else:
			if Oindex == True:
				Oindex = list(range(len(self.output_name)))
			Y, YValid = self.fl.load(Xindex, Oindex, s='Y')
			if checkValid:
				assert np.sum(YValid) < 1
			return X, Y

	# feature config
	blockPre = 72
	blockSuf = 72
	maxBlockLen = 1000
	timeScale = [1, 2, 4, 8, 16, 24, 48, 72]
	weather_input = ["temperature", "pressure", "humidity", "wind_direction", "wind_speed"]
	weather_seg = [
		(-15, 40, [1, 4]), 
		(980, 1040, [1, 4]), 
		(0, 100, [1, 4]), 
		(-45, 45, [1, 4]), 
		(0, 45, [1, 4])
	]
	weather_timeScale = [1, 4, 8, 24, 48, 72]
	timeScale_feature_config = [
		("mean", np.mean, []),
		("max", np.max, []),
		("min", np.min, []),
		("1per", lambda dd, axis=None: np.percentile(dd, 25, axis=axis), []),
		("2per", np.median, []),
		("3per", lambda dd, axis=None: np.percentile(dd, 75, axis=axis), [])
	]
	weather_timeScale_feature_config = [
		("mean", np.mean, []),
		("max", np.max, []),
		("min", np.min, [])
	]

	def __init__(self, data, cache, test = False, dontcheck = False, dontUpdateY=True):
		self.location = data.location
		self.cache = cache
		self.data = data
		self.test = test
		if self.location == "bj":
			self.AQ_input = ["PM25", "PM10", "NO2", "CO", "O3", "SO2"]
			self.AQ_output = ["PM25", "PM10", "O3"]
			self.AQ_output_idx = [0, 1, 4]
		elif self.location == 'ld':
			self.AQ_input = ["PM25", "PM10", "NO2"]
			self.AQ_output = ["PM25", "PM10"]
			self.AQ_output_idx = [0, 1]
		
		self.fl = FeatureLoader(cache+'b')
		try:
			self.restoreInfo()
		except:
			self.feature_name = []
			self.feature_attr = []
			self.sample_name = []
			self.sample_attr = []
			self.output_name = []
			self.output_attr = []
			self.time_block = {}
		
		### fix cache
		#self.time_block = {}
		#for station_idx, station_name in enumerate(data.station_name):
		#	self.time_block[station_name] = []
		#	for st, et in generateBlock(joinBlock([(86-self.blockPre, 10312)]), self.maxBlockLen, self.blockPre):
		#		self.time_block[station_name].append((st+self.blockPre, et))
		#self.saveInfo()
		#err()
		###

		self.dontcheck = dontcheck
		if dontcheck:
			logging.info("don't check feature")
			return
		self.dontUpdateY = dontUpdateY
		if dontUpdateY:
			logging.info("don't update Y")

		self.generateALLFeature()			

	query_last_para = {}
	query_last_res = {}
	def query_cache(self, fn, args):
		if fn.__name__ not in self.query_last_para or self.query_last_para[fn.__name__] != args:
			self.query_last_para[fn.__name__] = args
			self.query_last_res[fn.__name__] = fn(*args)
		return self.query_last_res[fn.__name__]

	def checkCache(self, flag, fname):
		if flag:
			logging.info("%s ok" % fname)
		else:
			logging.info("%s passed" % fname)
		
	'''fix_WATS_rec = {}
	def fix_WATS(self):
		i2a=[2, 4, 8]
		for i, (name, attr) in enumerate(zip(self.feature_name, self.feature_attr)):
			if "W_ATS" in name and name[-2] != 'a':
				if name not in self.fix_WATS_rec:
					self.fix_WATS_rec.append[name] = 0
				
				p = self.fix_WATS_rec.append[name] 
				self.fix_WATS_rec.append[name] += 1
				self.feature_name[i] += "_a%d" % i2a[p]
				logging.info("fix %s to %s", name, self.feature_name[i])
				self.saved += 1'''

	def getLastTime(self, station_name):
		all_sample = self.filter(lambda n, a: ("station_%s" % station_name) in a, None, 'sample')
		return max(map(lambda i: self.sample_attr[i][0], all_sample))

	def generateALLFeature(self):
		data = self.data
		if self.test:
			self.maxBlockLen = 200

		#self.fix_WATS()

		for station_idx, station_name in enumerate(data.station_name):
			if self.test and station_idx > 1:
				break

			logging.info("%d %s" % (station_idx, station_name))
			if station_name in self.time_block:
				block_reference = self.time_block[station_name]
			else:
				block_reference = []

			new_reference = []
			for st, et in generateBlock(joinBlock(data.time_block['query_station_AQI'], \
					data.time_block['query_station_weather']), self.maxBlockLen, self.blockPre, \
					mins=0, reference=block_reference):
				
				if st + self.blockPre >= et:
					continue
				
				if self.dontUpdateY:
					if (st+self.blockPre, et, 0) in block_reference:
						new_reference.append((st+self.blockPre, et, 0))
						continue
					elif (st+self.blockPre, et, 1) in block_reference:
						new_reference.append((st+self.blockPre, et, 1))
						continue

				if (st+self.blockPre, et, 0) in block_reference:
					OTS = True
				else:
					logging.info("Timeblock %d~%d" % (st + self.blockPre, et))
					OTS = self.generateOutput(station_name, st, et)
				
				if OTS:
					new_reference.append((st+self.blockPre, et, 0))
				else:
					new_reference.append((st+self.blockPre, et, 1))

				if (st+self.blockPre, et, 0) in block_reference or (st+self.blockPre, et, 1) in block_reference:
					continue  # not necessary to update X

				#AQI_m = data.query_station_AQI(station_name, st, et)
				#logging.info("\tAQI loaded")
				AQTS = self.generateAQTimeScale(station_name, st, et)
				self.checkCache(AQTS, "\tAQTS")
				AQH = self.generateAQHistorical(station_name, st, et)
				self.checkCache(AQH, "\tAQH")
				AQRL = self.generateAQRepeatLabel(station_name, st, et)
				self.checkCache(AQTS, "\tAQRL")
				
				#WEA_m = data.query_station_area_weather(station_name, 8, st, et)
				#logging.info("\tWEA loaded")
				WEATS = self.generateWeatherAreaTimeScale(station_name, st, et)
				self.checkCache(AQTS, "\tWEATS")

				type_num = int(np.max(np.array(list(data.station_info.values()))[:, 2])) + 1
				STT = self.generateStationInfo(station_name, data.station_info[station_name][2], type_num, st, et)
				self.checkCache(AQTS, "\tSTT")
			
			self.time_block[station_name] = new_reference
			self.saveInfo()
		self.fl.flush()

	def generateOutput(self, station_name, st, et):
		sample_name = []
		sample_attr = []
		output_name = []
		output_attr = []
		output_data = []

		follow_data = None
		try:
			follow_data = self.data.query_station_AQI(station_name, et, et+self.blockSuf)
		except:
			pass
		

		for j in range(self.blockPre + st, et):
			sample_name.append("%s_%d" % (station_name, j))
			now_attr = [j, "station_%s" % station_name, "station_type%d" % self.data.station_info[station_name][2]]
			t = ConvertNum2Time(j)
			if t.tm_year == 2018 and t.tm_mon >= 3:
				now_attr.append("recently")
			if t.tm_hour == 23:
				now_attr.append("exactly_point")
			if j >= et - self.blockSuf and follow_data is None:
				now_attr.append("unknown")
			sample_attr.append(now_attr)

		if sample_name[0] not in self.sample_name:
			self.addSample(sample_name, sample_attr)
		else:
			self.updateSample(sample_name, sample_attr)

		for p, p_name in zip(self.AQ_output_idx, self.AQ_output):
			for k in range(1, self.blockSuf+1):
				output_name.append("%s_%d" % (p_name, k))
				output_attr.append([p_name, '%d' % k])

		if output_name[0] not in self.output_name:
			self.addOutput(output_name, output_attr)
		else:
			self.updateOutput(output_name, output_attr)

		if self.checkY(sample_name, output_name):
			return True

		data = self.query_cache(self.data.query_station_AQI, (station_name, st, et))
		output_valid = []
		if follow_data is not None:
			data = np.concatenate([data, follow_data], 0)
			real_end = et
		else:
			real_end = max(et - self.blockSuf, self.blockPre + st)
		for j in range(self.blockPre, real_end - st):
			now_data = []
			for p, p_name in zip(self.AQ_output_idx, self.AQ_output):
				now_data.append(data[j+1:j+self.blockSuf+1, p])
			output_data.append(np.concatenate(now_data, 0))
			output_valid.append(np.zeros(output_data[0].size, dtype=bool))

		for j in range(real_end, et):
			output_data.append(np.zeros((len(output_name))))
			output_valid.append(np.ones(output_data[0].size, dtype=bool))
		self.updateY(sample_name[0], output_name[0], np.stack(output_data), np.stack(output_valid))
		return real_end == et
				
	def generateAQTimeScale(self, station_name, st, et):

		firstFeatureName = "AQ_TS_%s_%s_%d_%s" % ('o', self.AQ_input[0], self.timeScale[0], self.timeScale_feature_config[0][0])
		firstSampleName = "%s_%d" % (station_name, st + self.blockPre)
		assert firstSampleName in self.sample_name
		if self.checkX(firstSampleName, firstFeatureName):
			return False

		data = self.query_cache(self.data.query_station_AQI, (station_name, st, et))
		diffdata = data - np.concatenate((np.zeros((1, len(self.AQ_input))), data[:-1]))
		accedata = diffdata - np.concatenate((np.zeros((1, len(self.AQ_input))), diffdata[:-1]))
		
		res = FeatureGroup()
		for i in range(len(self.AQ_input)):

			for (dataname, datasrc, datasrc_idx, dataattr) in zip("oda", [data, diffdata, accedata], [0, 1, 2], [[], ['diff'], ['acce']]):
				for k in self.timeScale:
					if k <= datasrc_idx:
						continue
					for fname, fn, fattr in self.timeScale_feature_config:
						res.feature_name.append("AQ_TS_%s_%s_%d_%s" % (dataname, self.AQ_input[i], k, fname))
						res.feature_attr.append(fattr + dataattr)
					
					timedata = np.zeros((k, et-st-self.blockPre))
					for j in range(k):
						timedata[j, :] = datasrc[self.blockPre-j:et-st-j, i]
					partX = np.zeros((et-st-self.blockPre, len(self.timeScale_feature_config)))
					for fidx, (fname, fn, fattr) in enumerate(self.timeScale_feature_config):
						partX[:, fidx] = fn(timedata, axis=0)
					
					res.data.append(partX)

		res.data = np.concatenate(res.data, 1)
		res.assertSize()

		if firstFeatureName not in self.feature_name:
			self.addFeature(res.feature_name, res.feature_attr)
		self.updateX(firstSampleName, firstFeatureName, res.data)

		return True

	def generateAQHistorical(self, station_name, st, et):
		
		firstFeatureName = "AQ_H_%s_%d" % (self.AQ_input[0], 0)
		firstSampleName = "%s_%d" % (station_name, st + self.blockPre)
		assert firstSampleName in self.sample_name
		if self.checkX(firstSampleName, firstFeatureName):
			return False
		
		data = self.query_cache(self.data.query_station_AQI, (station_name, st, et))
		res = FeatureGroup()
		for k in range(self.blockPre):
			for i in range(len(self.AQ_input)):
				res.feature_name.append("AQ_H_%s_%d" % (self.AQ_input[i], k))
				res.feature_attr.append([])
		for j in range(self.blockPre, et-st):
			res.data.append(np.reshape(data[j-self.blockPre:j, :], (-1)))
		res.data = np.array(res.data)
		res.assertSize()

		if firstFeatureName not in self.feature_name:
			self.addFeature(res.feature_name, res.feature_attr)
		self.updateX(firstSampleName, firstFeatureName, res.data)

		return True
			
	def generateAQRepeatLabel(self, station_name, st, et):
		firstFeatureName = "AQ_RL_HOUR_%d" % (0)
		firstSampleName = "%s_%d" % (station_name, st + self.blockPre)
		assert firstSampleName in self.sample_name
		if self.checkX(firstSampleName, firstFeatureName):
			return False
		
		data = self.query_cache(self.data.query_station_AQI, (station_name, st, et))

		res = FeatureGroup()
		res.data = np.zeros((et-st-self.blockPre, 24+7+24*7))
		for i in range(24):
			res.feature_name.append("AQ_RL_HOUR_%d" % i)
			res.feature_attr.append(['label'])
		for i in range(7):
			res.feature_name.append("AQ_RL_WEEK_%d" % i)
			res.feature_attr.append(['label'])
		for i in range(7):
			for j in range(24):			
				res.feature_name.append("AQ_RL_WEEKxHOUR_%d_%d" % (i, j))
				res.feature_attr.append(['label', 'labelxlabel'])

		for j in range(self.blockPre, et-st):
			if self.location == 'bj':
				t = ConvertNum2Time(j)
			elif self.location == 'ld':
				t = ConvertNum2Time(j)
			res.data[j-self.blockPre, t.tm_hour] = 1
			res.data[j-self.blockPre, 24 + t.tm_wday] = 1
			res.data[j-self.blockPre, 24 + 7 + t.tm_wday * 24 + t.tm_hour] = 1

		res.assertSize()

		if firstFeatureName not in self.feature_name:
			self.addFeature(res.feature_name, res.feature_attr)
		self.updateX(firstSampleName, firstFeatureName, res.data)

		return True

	def generateWeatherAreaTimeScale(self, station_name, st, et):
		dataname = "%s_seg%din%s" % (self.weather_input[0], 0, self.weather_seg[0][2][0])
		firstFeatureName = "W_ATS_%s_%d_%s_a2" % (dataname, self.weather_timeScale[0], self.weather_timeScale_feature_config[0][0])
		firstSampleName = "%s_%d" % (station_name, st + self.blockPre)
		assert firstSampleName in self.sample_name
		if self.checkX(firstSampleName, firstFeatureName):
			return False
		
		data = self.query_cache(self.data.query_station_area_weather, (station_name, 8, st, et))

		res = FeatureGroup()

		assert len(self.weather_input) == 5
		assert self.weather_input[3] == "wind_direction"
		assert self.weather_input[4] == "wind_speed"

		firstFeature = []
		for i in [0, 1, 2, 4]:
			input = FeatureGroup()
			input.data.append(data[:, :, :, i])
			input.feature_name.append(self.weather_input[i])
			input.feature_attr.append([])
			segFeature = self.generateSegment(input, [self.weather_seg[i]])
			firstFeature.append(segFeature)

		for i in range(4):
			input = FeatureGroup()
			input.data.append(np.sin((data[:, :, :, 3]-i*45) / 180 * math.pi) * data[:, :, :, 4])
			input.feature_name.append("wind_dir%d_sin" % i)
			input.feature_attr.append([])
			input.data.append(np.cos((data[:, :, :, 3]-i*45) / 180 * math.pi) * data[:, :, :, 4])
			input.feature_name.append("wind_dir%d_cos" % i)
			input.feature_attr.append([])
			segFeature = self.generateSegment(input, [self.weather_seg[3], self.weather_seg[3]])
			firstFeature.append(segFeature)
		firstFeature = FeatureGroup.mergeFeatureArray(firstFeature)

		logging.info("\tfeaturenum: %d, shape: %s", len(firstFeature.feature_name), firstFeature.data[0].shape)
		diffFeature = self.generateDiff(firstFeature)

		logging.info("\tfeaturenum: %d, shape: %s", len(firstFeature.feature_name), firstFeature.data[0].shape)

		for feature, datasrc_idx in zip([firstFeature, diffFeature], [0, 1]):

			alldata = np.stack(feature.data, 3) # time * l * l * feature
			timedata = np.zeros((self.weather_timeScale[-1], et-st-self.blockPre, 8, 8, len(feature.feature_name)))
			# timescale * time * l * l * feature
			for j in range(self.weather_timeScale[-1]):
				timedata[j] = alldata[self.blockPre-j:et-st-j]
			
			max_h = len(self.weather_timeScale)
			max_h *= 3 * len(self.weather_timeScale_feature_config)
			block_data = np.zeros((et-st-self.blockPre, max_h, len(firstFeature.feature_name)))
			block_idx = 0
			logging.info("\tshape: %s", timedata.shape)

			for ts_idx, k in enumerate(self.weather_timeScale):
				if k <= datasrc_idx:
					continue
				for area_idx, area_l in enumerate([2, 4, 8]):
					#logging.info("\t\t%d_%d_%d" % (datasrc_idx, k, area_l))

					for fidx, (fname, fn, fattr) in enumerate(self.weather_timeScale_feature_config):
						for dataname, dataattr in zip(feature.feature_name, feature.feature_attr):
							res.feature_name.append("W_ATS_%s_%d_%s_a%d" % (dataname, k, fname, area_l))
							res.feature_attr.append(fattr + dataattr)

						block_data[:, block_idx, :] = \
							fn(timedata[:k, :, 4-area_l//2:4+area_l//2, 4-area_l//2:4+area_l//2], axis=(0, 2, 3))
						block_idx += 1
			
			res.data.append(np.reshape(block_data[:, :block_idx, :], [et-st-self.blockPre, -1]))

		res.data = np.concatenate(res.data, 1)
		res.assertSize()

		if firstFeatureName not in self.feature_name:
			self.addFeature(res.feature_name, res.feature_attr)
		self.updateX(firstSampleName, firstFeatureName, res.data)

		return True

	def generateStationInfo(self, station_name, type_i, type_num, st, et):
		firstFeatureName = "ST_TYPE%d" % (0)
		firstSampleName = "%s_%d" % (station_name, st + self.blockPre)
		assert firstSampleName in self.sample_name
		if self.checkX(firstSampleName, firstFeatureName):
			return False

		res = FeatureGroup()
		res.data = np.zeros((et-st-self.blockPre, type_num))

		for i in range(type_num):
			res.feature_name.append("ST_TYPE%d" % i)
			res.feature_attr.append(['label'])
		res.data[:, type_i] = 1

		res.assertSize()

		if firstFeatureName not in self.feature_name:
			self.addFeature(res.feature_name, res.feature_attr)
		self.updateX(firstSampleName, firstFeatureName, res.data)

		return res

	def generateSegment(self, input, seg_config):
		input.assertFeatureArray()
		res = FeatureGroup()
		for odata, oname, oattr, (s, t, n) in zip(input.data, input.feature_name, input.feature_attr, seg_config):
			for nn in n:
				for j in range(nn):
					nows = (t-s) * j // nn + s
					res.data.append(np.maximum(odata - nows, 0))
					res.feature_name.append("%s_seg%din%s" % (oname, j, nn))
					res.feature_attr.append(oattr + ["seg%d" % nn])
		res.assertFeatureArray()
		return res

	def generateDiff(self, input, suffix='d', attr=['diff']):
		input.assertFeatureArray()
		res = FeatureGroup()

		data = np.stack(input.data, 1)
		data = data - np.concatenate([np.zeros((1, ) + data.shape[1:]), data[:-1]])
		for oname, oattr in zip(input.feature_name, input.feature_attr):
			#res.data.append(odata - np.concatenate([np.zeros((1, ) + odata.shape[1:]), odata[:-1]]))
			res.feature_name.append("%s_%s" % (oname, suffix))
			res.feature_attr.append(oattr + attr)
		res.data = np.split(data, len(res.feature_name), axis=1)
		res.data = list(map(lambda x: x[:, 0], res.data))
		res.assertFeatureArray()
		return res


def generate(data, **config):
	return Feature(data, **config)