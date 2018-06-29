from utils import *
import logging
import load_data as Preprocessing
import numpy as np
import FeatureGenByHour as Feature
import LinearRegression as Model
import LinearRegressionDiff as ModelDiff
from FirstRegression import FirstRegression
import argparse
import sys
import csv
import utils

version = "1"
cachedir = "./cache/"
datadir = "./data/"

logging.basicConfig(
	filename = 0,
	level=logging.DEBUG,
	format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
	datefmt='%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
cargs = parser.parse_args(sys.argv[1:])

if cargs.debug:
    import ptvsd
    ptvsd.enable_attach('my_secret')
    # tcp://my_secret@gpu-kappa:5678
    logging.info("wait debug")
    ptvsd.wait_for_attach()

p_config = {
    "cache": cachedir + "p_cache%s" % version,
    "location": "bj",
    "aq_file": datadir + "beijing_aq_complete.csv",
    "historical_meo_grid_file": datadir + "Beijing_historical_meo_grid.csv",
    "station_file": datadir + "position_info_bj.csv"
}

data = Preprocessing.load(**p_config)

f_config = {
    "cache": cachedir + "/f_cache%s" % version,
    #"laststation": 0
    #"test": True, 
    #"dontcheck": True
}
feature = Feature.generate(data, **f_config)

m_config = {
    "cache": cachedir + "m_cache%s" % version,
	"args": {
		"corLowLimit": 0.1,
	}
}
model = Model.train(data, feature, **m_config)
model.fit()

m2_config = {
    "cache": cachedir + "m2_cache%s" % version,
	"args": {
		"corLowLimit": 0.1,
		"location":"bj"
	}
}
model2 = ModelDiff.train(data, feature, **m2_config)
model2.fit()

fr_config = {
	"cache": cachedir + "fr_cache%s" % version,
	"args": {
		"useorz":True, 
		"dontcheck":True,
		"location": "bj"
	}
}
firstRegression = FirstRegression(data, feature, model, model2, **fr_config)
firstRegression.fit()

def testmodel():
	Xindex = feature.filter(lambda n, a: "recently" in a and "unknown" not in a and "exactly_point" in a, None, 'sample')
	#Xindex = Xindex[:10]
	#Xindex, Y_pre, Y = model.predictModel(Xindex, model.Findex2, True)
	Xindex, Y_pre, Y = firstRegression.predict(Xindex, True)
	
	with open("./cache/output_sample", 'w') as f:
		for i in range(10):
			for j in range(Y.shape[1]):
				f.write("%f " % Y[i,j])
			f.write("\n")
			for j in range(Y.shape[1]):
				f.write("%f " % Y_pre[i,j])
			f.write("\n")
		f.write("\n")
	logging.info("24 SMAPE: %f", calSMAPE(Y, Y_pre, 24))
	logging.info("48 SMAPE: %f", calSMAPE(Y, Y_pre, 48))
	logging.info("72 SMAPE: %f", calSMAPE(Y, Y_pre, 72))


def predictFuture():
	Xindex = []
	t = 0
	temp_last_time = 0
	for station_name in data.station_name:
		feature.filter(lambda n, a: ("station_%s" % station_name) in a and "recently" in a, None, 'sample')
		t = feature.getLastTime(station_name)
		temp_last_time = t
		logging.info("%s last time %d", station_name, t)
		Xindex.append(feature.sample_name.index("%s_%d" % (station_name, t)))

	sorte = all(Xindex[i] <= Xindex[i+1] for i in range(len(Xindex)-1))
	assert sorte

	# find all last time
	#Xindex, Y_pre, Y = model.predictModel(Xindex, model.Findex2, False)
	Xindex, Y_pre, Y = firstRegression.predict(Xindex, False)
	logging.info("%s", Y_pre.shape)

	sorte = all(Xindex[i] <= Xindex[i+1] for i in range(len(Xindex)-1))
	assert sorte

	with open("./cache/output_sample", 'w') as f:
		for i in range(Y.shape[0]):
			for j in range(Y.shape[1]):
				f.write("%f " % Y_pre[i,j])
			f.write("\n")

	output_mapping = [6, 23, 10, 26, 0, 18, 25, 2, 34, 9, 33, 11, 8, 3, 29, 24, 22, 19, 14, 20, 12, 15, 28, 4, 1, 16, 5, 31, 32, 13, 21, 30, 27, 17, 7]

	f_sample = csv.reader(open('sample_submission_bj.csv', 'r'))
	f_our = csv.writer(open('our_submission_bj.csv', 'w'))

	cnt_line = 0
	current_station = ""
	start_pre_time = 0
	for line in f_sample:
		if cnt_line > 0:
			no_station = (cnt_line - 1) // 48
			if no_station == len(data.station_name):
				break
			if (cnt_line -1) % 48 == 0:
				current_station = data.station_name[output_mapping[no_station]]
				start_pre_time = feature.getLastTime(current_station) + 1
			delta_hour = int(line[0].split('#')[1])
			#temp_output = []
			if start_pre_time % 24 == 0:
				next_day_time = start_pre_time
				#temp_output.append([Y_pre[output_mapping[no_station], delta_hour], Y_pre[output_mapping[no_station], 72 + delta_hour], Y_pre[output_mapping[no_station], 144 + delta_hour]])
			else:
				next_day_time = ((start_pre_time // 24) + 1) * 24
			temp_output = [Y_pre[output_mapping[no_station], next_day_time - start_pre_time + delta_hour], Y_pre[output_mapping[no_station], next_day_time - start_pre_time + delta_hour + 72], Y_pre[output_mapping[no_station], next_day_time - start_pre_time + delta_hour + 144]]
			f_our.writerow([line[0], temp_output[0], temp_output[1], temp_output[2]])
		else:
			f_our.writerow([line[0], line[1], line[2], line[3]])
		cnt_line += 1
		#if (cnt_line % 100 == 0):
			#logging.info(cnt_line)
	print(utils.ConvertNum2Time(start_pre_time))
	print(utils.ConvertNum2Time(next_day_time))
	return temp_last_time

feature_last_time = predictFuture()

#print(feature_last_time)

file_timestamp = open("feature_last_time_bj.txt","w")
file_timestamp.write(str(feature_last_time))
file_timestamp.close()

logging.info("all finish")
