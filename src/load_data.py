# -*- coding: UTF-8 -*-
import csv
import numpy as np
import pickle as pkl
import os
import utils
import requests
import time
import math

def dist(c1, c2, xy):
	return math.sqrt((xy[c1][0]-xy[c2][0])*(xy[c1][0]-xy[c2][0])+(xy[c1][1]-xy[c2][1])*(xy[c1][1]-xy[c2][1]))

def load_data(aq_data, weather_data, station_data, location, aq_base, weather_base):
	f_aq = csv.reader(open(aq_data, 'r'))
	f_weather = csv.reader(open(weather_data, 'r'))
	f_pos = csv.reader(open(station_data, 'r'))
	cnt = 0
	time_order = []
	city_order = []
	city_num = 0
	main_data = []
	pre_line = ""
	cnt_aq = 0
	if location == 'bj':
		cnt_aq = 6
	else:
		cnt_aq = 3
	for line in f_aq:
		if (cnt > 0):
			time_order.append(utils.ConvertStr2Num(line[1]))
			if (time_order[len(time_order)-1] - time_order[len(time_order)-2] > 1) and (line[0] == pre_line[0]):
				big_step = time_order[len(time_order)-1] - time_order[len(time_order)-2]
				for i in range(big_step-1):
					temp = []
					temp_num = 0.0
					for j in range(2, 2+cnt_aq):
						temp_num = (float(line[j])-float(pre_line[j])/big_step)*(i+1)+float(line[j])
						temp.append(temp_num)
					main_data[len(main_data)-1].append(temp)
			if (line[0] != pre_line[0]):
				city_num += 1
				city_order.append(line[0])
				main_data.append([])
			if (line[0] != pre_line[0]) or (line[1] != pre_line[1]):
				main_data[len(main_data)-1].append([float(line[i]) for i in range(2, 2+cnt_aq)])
		cnt += 1
		pre_line = line

	grid_weather = {}
	weather_cnt = 0
	if (location == 'bj'):
		len_x, len_y, len_z = 31, 21, 5
	else:
		len_x, len_y, len_z = 41, 21, 5
	cur_x, cur_y = 0, 0
	pre_num_time = -1
	for line in f_weather:
		if (weather_cnt>0):
			if line[3] != pre_line[3]:
				num_time = utils.ConvertStr2Num(line[3])
			else:
				num_time = pre_num_time
			if num_time not in grid_weather:
				grid_weather[num_time] = np.zeros((len_x, len_y, len_z))
				cur_x, cur_y = 0, 0
			grid_weather[num_time][cur_x][cur_y][0],grid_weather[num_time][cur_x][cur_y][1] = float(line[4]), float(line[5])
			grid_weather[num_time][cur_x][cur_y][2] = float(line[6])
			grid_weather[num_time][cur_x][cur_y][3], grid_weather[num_time][cur_x][cur_y][4] = float(line[7]), float(line[8])
			pre_num_time = num_time
			cur_y += 1
			if cur_y == len_y:
				cur_x += 1
				cur_y = 0
		weather_cnt += 1
		pre_line = line

	city_pos = {}
	type_collection = []
	for line in f_pos:
		if line[3] not in type_collection:
			type_collection.append(line[3])
		city_pos[line[0]] = [float(line[1]), float(line[2]), type_collection.index(line[3])]

	time_block = {}
	time_block['query_station_AQI'] = [(utils.ConvertStr2Num(aq_base), utils.ConvertStr2Num(aq_base)+len(main_data[0]))]
	#self.time_block = [(utils.ConvertStr2Num(self.aq_time_base), utils.ConvertStr2Num(self.aq_time_base)+len(self.aq_data[0]))]
	#self.time_block['query_station_weather'].append((utils.ConvertStr2Num(self.weather_time_base), utils.ConvertStr2Num(self.weather_time_base)+len(self.weather_data)))
	time_block['query_station_weather'] = [(utils.ConvertStr2Num(weather_base), utils.ConvertStr2Num(weather_base)+len(grid_weather))]
	#print(time_block)
	return main_data, city_order, city_pos, grid_weather, time_block




class MyError(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)



class QWdata:
	aq_data = []
	station_name = [] # num_station*1, string
	station_info = [] # num_station*3, float, (经度,纬度,监测站类型)
	weather_data = []
	time_block = {}
	location = ''
	aq_time_base = ''
	weather_time_base = ''
	grid_st_x = 0.0
	grid_st_y = 0.0
	grid_len_x = 0
	grid_len_y = 0
	grid_step = 0.1
	cnt_aq = 0
	def __init__(self, cache, location, aq_file, weather_file, station_file):
		self.location = location
		self.weather_time_base = '2017-01-01 00:00:00'
		if location == 'bj':
			self.aq_time_base = '2017-01-01 14:00:00'
			self.grid_st_x = 115.0 # 115.0~118.0
			self.grid_st_y = 39.0 # 39.0~41.0
			self.grid_len_x = 31
			self.grid_len_y = 21
			self.cnt_aq = 6
		else:
			self.aq_time_base = '2017-01-01 00:00:00'
			self.grid_st_x = -2.0 # -2.0~2.0
			self.grid_st_y = 50.5 # 50.5~52.5
			self.grid_len_x = 41
			self.grid_len_y = 21
			self.cnt_aq = 3
		if (os.path.exists(cache)):
			data_file = open(cache, 'rb')
			self.aq_data, self.station_name, self.station_info, self.weather_data, self.time_block = pkl.load(data_file)
		else:
			self.aq_data, self.station_name, self.station_info, self.weather_data, self.time_block = load_data(aq_file, weather_file, station_file, location, self.aq_time_base, self.weather_time_base)

		current_gm_time = time.gmtime()
		current_time = utils.ConvertTime2Num(current_gm_time)
		data_last_time = self.time_block['query_station_AQI'][len(self.time_block['query_station_AQI'])-1][1]
		data_gm_time = utils.ConvertNum2Time(data_last_time)

		if data_last_time < current_time:

			print('https://biendata.com/competition/airquality/'+ location + ('/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (data_gm_time.tm_year, data_gm_time.tm_mon, data_gm_time.tm_mday, data_gm_time.tm_hour, current_gm_time.tm_year, current_gm_time.tm_mon, current_gm_time.tm_mday, current_gm_time.tm_hour)))

			# get new AQI data
			url_aq = 'https://biendata.com/competition/airquality/'+ location + ('/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (data_gm_time.tm_year, data_gm_time.tm_mon, data_gm_time.tm_mday, data_gm_time.tm_hour, current_gm_time.tm_year, current_gm_time.tm_mon, current_gm_time.tm_mday, current_gm_time.tm_hour))
			responses = requests.get(url_aq)
			with open(location + '_new_aq.csv','w') as f:
				f.write(responses.text)
			f_new_aq = csv.reader(open(location + '_new_aq.csv','r'))
			is_new_empty = 0
			for line in f_new_aq:
				if line[0] == 'None':
					is_new_empty = 1
					break
				break
			if (is_new_empty):
				data_file = open(cache, 'wb')
				pkl.dump([self.aq_data, self.station_name, self.station_info, self.weather_data, self.time_block], data_file)
				return			
			new_aq_data = []
			new_time_order = []
			for i in range(len(self.station_name)):
				new_aq_data.append([])
				new_time_order.append([])
			cnt_line = 0
			new_aq_st, new_aq_en = 1000000, 0
			for line in f_new_aq:
				if cnt_line>-1:
					num_time_aq = utils.ConvertStr2Num(line[2])
					if num_time_aq < new_aq_st:
						new_aq_st = num_time_aq
					if num_time_aq > new_aq_en:
						new_aq_en = num_time_aq
					temp = []
					for i in range(3, 3+self.cnt_aq):
						if line[i]!='':
							temp.append(float(line[i]))
						else:
							temp.append(0.0)

					if line[1] in self.station_name:
						temp_time = utils.ConvertStr2Num(line[2])
						city_id = self.station_name.index(line[1])
						# 用插值处理内部的missing value
						if (len(new_time_order[city_id])>0) and (temp_time - new_time_order[city_id][len(new_time_order[city_id])-1] > 1):
							big_step = temp_time - new_time_order[city_id][len(new_time_order[city_id])-1]
							inter_st = new_aq_data[city_id][len(new_aq_data[city_id])-1]
							time_st = new_time_order[city_id][len(new_time_order[city_id])-1]
							for i in range(big_step-1):
								inter_ele = []
								for j in range(self.cnt_aq):
									inter_ele.append((i+1)*(temp[j]-inter_st[j])/big_step+inter_st[j])
								new_aq_data[city_id].append(inter_ele)
								new_time_order[city_id].append(time_st+i+1)
						if (len(new_time_order[city_id]) == 0) or (temp_time > new_time_order[city_id][len(new_time_order[city_id])-1]):
							new_aq_data[city_id].append(temp)
							new_time_order[city_id].append(temp_time)
						
				cnt_line += 1
			# AQ起始点特殊处理1: 全部丢失
			if (new_aq_st != data_last_time):
				last_value_time = data_last_time - 1
				for i in range(len(self.aq_data)):
					#inter_start = self.aq_data[i][last_value_time]
					#inter_end = new_aq_data[i][new_aq_st]
					#big_step = new_aq_st - last_value_time
					temp = new_aq_data[i][0]
					for j in range(new_aq_st - last_value_time - 1):
						#temp = []
						#for k in range(self.cnt_aq):
							#temp.append((j+1)*(inter_start[k]-inter_end[k])/big_step+inter_end[k])
						
						new_aq_data[i].insert(0, temp)
				new_aq_st = data_last_time
			#if (location == 'bj') and (data_last_time !=) 
			
			# AQ起始点特殊处理2: 部分站点丢失
			for i in range(len(new_aq_data)):
				if len(new_aq_data[i]) != new_aq_en - new_aq_st + 1:
					if (new_time_order[i][0] != new_aq_st):
						cnt_fill = new_aq_en - new_aq_st + 1 - len(new_aq_data[i])
						for j in range(cnt_fill):
							new_aq_data[i].insert(0, [0 for j in range(self.cnt_aq)])
					else:
						cnt_fill = new_aq_en - new_aq_st + 1 - len(new_aq_data[i])
						for j in range(cnt_fill):
							new_aq_data[i].append([0 for j in range(self.cnt_aq)])						

			for i in range(len(new_aq_data)):
				for j in range(len(new_aq_data[i])):
					for k in range(len(new_aq_data[i][j])):
						if new_aq_data[i][j][k] == 0:
							dist_arr, value_arr  = [], []
							for ii in range(len(new_aq_data)):
								if (ii != i) and (new_aq_data[ii][j][k]>0):
									dist_arr.append(dist(self.station_name[i], self.station_name[ii], self.station_info))
									value_arr.append(new_aq_data[ii][j][k])
							norm_dist, weight_sum = sum(dist_arr), 0.0
							for ii in range(len(dist_arr)):
								weight_sum += dist_arr[ii]*value_arr[ii]/norm_dist
							new_aq_data[i][j][k] = weight_sum
			if new_aq_st == data_last_time:
				temp_st = self.time_block['query_station_AQI'][len(self.time_block['query_station_AQI'])-1][0]
				del self.time_block['query_station_AQI'][len(self.time_block['query_station_AQI'])-1]
				self.time_block['query_station_AQI'].append((temp_st, new_aq_en+1))
			else:
				self.time_block['query_station_AQI'].append((new_aq_st, new_aq_en+1))

			for i in range(len(self.aq_data)):
				self.aq_data[i][len(self.aq_data[i]):len(self.aq_data[i])] = new_aq_data[i]

		# get new grid weather data
		data_last_time = self.time_block['query_station_weather'][len(self.time_block['query_station_weather'])-1][1]
		data_gm_time = utils.ConvertNum2Time(data_last_time)
		if current_time > data_last_time:
			url_grid = 'https://biendata.com/competition/meteorology/'+ location + ('_grid/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (data_gm_time.tm_year, data_gm_time.tm_mon, data_gm_time.tm_mday, data_gm_time.tm_hour, current_gm_time.tm_year, current_gm_time.tm_mon, current_gm_time.tm_mday, current_gm_time.tm_hour))
			print(url_grid)
			responses = requests.get(url_grid)
			with open(location + '_new_grid_weather.csv','w') as f:
				f.write(responses.text)
			f_new_grid = csv.reader(open(location + '_new_grid_weather.csv','r'))
			is_new_empty = 0
			for line in f_new_grid:
				if line[0] == 'None':
					is_new_empty = 1
					break
				pre_line = line
				break
			if (is_new_empty):
				data_file = open(cache, 'wb')
				pkl.dump([self.aq_data, self.station_name, self.station_info, self.weather_data, self.time_block], data_file)
				return
			cnt_line = 0
			new_weather_st, new_weather_en = 1000000, 0
			#pre_line = line
			for line in f_new_grid:
				if cnt_line > -1:
					num_time_grid = utils.ConvertStr2Num(line[2])
					if num_time_grid < new_weather_st:
						new_weather_st = num_time_grid
					if num_time_grid > new_weather_en:
						new_weather_en = num_time_grid
					pos_num = int(line[1].split('_')[2])
					if line[2] != pre_line[2]:
						#pre_num_time_grid = utils.ConvertStr2Num(pre_line[2])
						#if num_time_grid - pre_num_time_grid == 1:
						self.weather_data[num_time_grid] = np.zeros((self.grid_len_x, self.grid_len_y, 5))
						#else:
						#for j in range()
					cur_x, cur_y = pos_num//(self.grid_len_y), pos_num % (self.grid_len_y)
					self.weather_data[num_time_grid][cur_x][cur_y][0], self.weather_data[num_time_grid][cur_x][cur_y][1] = float(line[4]), float(line[5])
					self.weather_data[num_time_grid][cur_x][cur_y][2], self.weather_data[num_time_grid][cur_x][cur_y][3] = float(line[6]), float(line[7])
					self.weather_data[num_time_grid][cur_x][cur_y][4] = float(line[8])
				pre_line = line
				cnt_line += 1

			# Weather起始点特殊处理: 全部丢失
			if (new_weather_st != data_last_time) and (new_weather_st != 10903):
				last_value_time = data_last_time - 1
				#for i in range(len(self.aq_data)):
					# inter_start = self.aq_data[i][last_value_time]
					# inter_end = new_aq_data[i][new_aq_st]
					# big_step = new_aq_st - last_value_time
				temp = self.weather_data[new_weather_st]
				for j in range(new_weather_st-1, last_value_time, -1):
						# temp = []
						# for k in range(self.cnt_aq):
						# temp.append((j+1)*(inter_start[k]-inter_end[k])/big_step+inter_end[k])
					self.weather_data[j] = temp
				new_weather_st = data_last_time
			#if (new_weather_st == self.time_block['query_station_weather'][len(self.time_block['query_station_weather'])-1][1]):
			if (new_weather_st == data_last_time):
				#self.time_block['query_station_weather'][len(self.time_block['query_station_weather'])-1][1] = new_weather_en + 1
				temp_st = self.time_block['query_station_weather'][len(self.time_block['query_station_weather'])-1][0]
				del self.time_block['query_station_weather'][len(self.time_block['query_station_weather'])-1]
				self.time_block['query_station_weather'].append((temp_st, new_weather_en+1))
			else:
				#if data_last_time ==
				self.time_block['query_station_weather'].append((new_weather_st, new_weather_en+1))
			
			i_st = self.time_block['query_station_weather'][len(self.time_block['query_station_weather'])-1][0]
			i_en = self.time_block['query_station_weather'][len(self.time_block['query_station_weather'])-1][1]
			ptr_st = i_st
			for i in range(i_st, i_en):
				if i not in self.weather_data:
					#print(i)
					#del self.time_block['query_station_weather'][len(self.time_block['query_station_weather'])-1]
					#if ptr != i:
						#self.time_block['query_station_weather'].append((ptr, i))
					#if i+1 != i_en:
						#self.time_block['query_station_weather'].append((i+1, i_en))
						#ptr = i+1
					ptr_st = i-1
					while (ptr_st>=i_st) and (ptr_st not in self.weather_data):
						ptr_st = ptr_st - 1
					ptr_en = i+1
					while (ptr_en<i_en) and (ptr_en not in self.weather_data):
						ptr_en = ptr_en + 1
					if (ptr_st != i_st-1):
						self.weather_data[i] = self.weather_data[ptr_st]
					else:
						self.weather_data[i] = self.weather_data[ptr_en]
							


			# get new weather data
			#if location == 'bj':
				#url_meo = 'https://biendata.com/competition/meteorology/'+ location + '/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (data_gm_time.tm_year, data_gm_time.tm_month, data_gm_time.tm_day. data_gm_time.tm_hour, current_gm_time.tm_year, current_gm_time.tm_month, current_gm_time.tm_day. current_gm_time.tm_hour)
				#responses = requests.get(url_meo)
				#with open(location + '_new_meo_weather.csv','w') as f:
					#f.write(responses.text)





		data_file = open(cache, 'wb')
		pkl.dump([self.aq_data, self.station_name, self.station_info, self.weather_data, self.time_block], data_file)
		#self.time_block['query_station_AQI'].append((utils.ConvertStr2Num(self.aq_time_base), utils.ConvertStr2Num(self.aq_time_base)+len(self.aq_data[0])))
		#self.time_block = {}

	
	# 以2017-01-01为时间戳基准，由于北京的AQ数据缺了14小时，所以query AQ时需要进行一定的处理
	# 区间均处理为左闭右开
	
	def query_station_AQI(self, _station_name, start_time, end_time):
		temp_aq = []
		out_flag = 0
		start_pos = 0
		for i in range(len(self.time_block['query_station_AQI'])):
			if (start_time>=self.time_block['query_station_AQI'][i][0]) and (end_time<=self.time_block['query_station_AQI'][i][1]):
				start_pos += start_time - self.time_block['query_station_AQI'][i][0]
				out_flag = 1
				break
			start_pos += self.time_block['query_station_AQI'][i][1] - self.time_block['query_station_AQI'][i][0]
		if (out_flag == 0):
			raise MyError('Out of history: Air Quality Query')	
		#for i in range(start_time - utils.ConvertStr2Num(self.aq_time_base), end_time - utils.ConvertStr2Num(self.aq_time_base)):
		for i in range(start_pos, start_pos + end_time - start_time):
			temp_aq.append(self.aq_data[self.station_name.index(_station_name)][i])
		return np.array(temp_aq)
	
	def query_station_area_weather(self, _station_name, area_len, start_time, end_time):
		temp_weather = []
		out_flag = 0
		for i in range(len(self.time_block['query_station_weather'])):
			if (start_time>=self.time_block['query_station_weather'][i][0]) and (end_time<=self.time_block['query_station_weather'][i][1]):
				out_flag = 1
				break
		if (out_flag == 0):
			raise MyError('Out of history: Weather Query')
		if (area_len>self.grid_len_y) or (area_len>self.grid_len_x):
			raise MyError('Out of range: Weather area is too large')
		pos_x, pos_y = int(round((self.station_info[_station_name][0] - self.grid_st_x)/self.grid_step)), int(round((self.station_info[_station_name][1] - self.grid_st_y)/self.grid_step))
		#print(pos_x, pos_y)
		if pos_x-(area_len-1)//2<0:
			st_j, en_j = 0, area_len
		else:
			if pos_x+(area_len-1)//2>=self.grid_len_x:
				st_j, en_j = self.grid_len_x-area_len, self.grid_len_x
			else:
				st_j, en_j = pos_x-(area_len-1)//2, pos_x-(area_len-1)//2+area_len
		if pos_y-(area_len-1)//2<0:
			st_k, en_k = 0, area_len
		else:
			if pos_y+(area_len-1)//2>=self.grid_len_y:
				st_k, en_k = self.grid_len_y-area_len, self.grid_len_y
			else:
				st_k, en_k = pos_y-(area_len-1)//2, pos_y-(area_len-1)//2+area_len
		#print(st_j, en_j, st_k, en_k)
		temp_weather = np.zeros((end_time-start_time, area_len, area_len, 5))
		for i in range(start_time - utils.ConvertStr2Num(self.weather_time_base), end_time - utils.ConvertStr2Num(self.weather_time_base)):
			for j in range(st_j, en_j):
				for k in range(st_k, en_k):
					new_i = i - start_time + utils.ConvertStr2Num(self.weather_time_base)
					new_j = j - st_j
					new_k = k - st_k
					#print(i, j, k)
					#print(new_i, new_j, new_k)
					for v in range(5):
						temp_weather[new_i][new_j][new_k][v] = self.weather_data[i][j][k][v]
		return temp_weather


'''
p_config = {
    "cache": cachedir + "p_cache%d" % version,
    "location": "bj",
    "aq_file": datadir + "beijing_aq_complete.csv",
    "historical_meo_grid_file": datadir + "Beijing_historical_meo_grid.csv",
    "station_file": datadit + "position_info_bj.csv"
	......
}
传入参数
cache参数为预处理后的pickle文件。如果存在，直接读取，节省时间；不存在，处理后保存在该文件内。
'''
def load(**kargs):
	#data = QWdata(**kargs)
	data = QWdata(kargs['cache'], kargs['location'], kargs['aq_file'], kargs['historical_meo_grid_file'], kargs['station_file'])
	return data

'''
成员变量
data.station_name <- city_data_bj
data.station_info <- city_pos_bj  另外加上北京观测站的类型（比如城区、郊区、交通、对照）

时间不要使用字符串表示，全部转化成标准时间戳/小时（整数，比如x为1点，x+1就是2点，x-1为0点）
填一下utils里的两个转换函数。

给出每个API存在数据的时间分段（注意排序）
data.time_block = {
	"query_station_AQI" : [(t1_start, t1_end), (t2_start, t2_end)], 
	"query_station_weather" : [(t3_start, t3_end), (t4_start, t4_end)], 
}

成员函数
data.query_station_AQI(station_name, start_time, end_time) 返回一个矩阵，最好用numpy的矩阵 time_length * AQI_feature
data.query_station_weather(station_name, start_time, end_time)   北京有相关数据，伦敦需要做一个最近邻估计一下 返回一个矩阵， time_length * weather_feature
data.query_station_area_weather(station_name, area_len, start_time, end_time) 返回站点周边area_len*area_len个格点的天气情况 time_length * area_len * area_len * weather_feature
以上API若时间非法或数据缺失及时报错。
'''

'''
datadir = "/home/kp/文档/kddcup/"

p_config = {
    #"cache": cachedir + "p_cache%d" % version,
    "cache": "bj_cache",
    "location": "bj",
    "aq_file": datadir + "beijing_aq_complete.csv",
    "historical_meo_grid_file": datadir + "Beijing_historical_meo_grid.csv",
    "station_file": datadir + "position_info_bj.csv",
}

bj_data = load(**p_config)

temp_AQI = bj_data.query_station_AQI('aotizhongxin_aq', 10913, 10914)
temp_Weather = bj_data.query_station_area_weather('aotizhongxin_aq', 3, 10913, 10914)

print(temp_AQI)
print(temp_Weather)


q_config = {
    #"cache": cachedir + "p_cache%d" % version,
    "cache": "ld_cache",
    "location": "ld",
    "aq_file": datadir + "London_aq_complete.csv",
    "historical_meo_grid_file": datadir + "London_historical_meo_grid.csv",
    "station_file": datadir + "London_AirQuality_Stations.csv",
}


ld_data = load(**q_config)

ld_data.query_station_AQI('CD1', 0, 10897)
ld_data.query_station_AQI('CD1', 10903, 11560)
ld_data.query_station_area_weather('BL0', 3, 0, 10806)
ld_data.query_station_area_weather('BL0', 3, 10903, 11561)


bj_data.query_station_AQI('aotizhongxin_aq', 14, 10912)
bj_data.query_station_AQI('aotizhongxin_aq', 10913, 11560)
bj_data.query_station_area_weather('badaling_aq', 3, 0, 10806)
bj_data.query_station_area_weather('badaling_aq', 3, 10903, 11561)

#print(temp_AQI)
#print(temp_Weather)

print(bj_data.time_block)
print(ld_data.time_block)
#print(len(bj_data.station_name))
#print(len(ld_data.station_name))

bj_data.query_station_area_weather('aotizhongxin_aq', 8, 10985, 11553)
for block in bj_data.time_block['query_station_weather']:
	for j in range(block[0], block[1]):
		if j not in ld_data.weather_data:
			print(j)
'''
