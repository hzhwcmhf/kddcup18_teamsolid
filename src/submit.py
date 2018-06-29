# coding: utf-8
import logging
import requests
import time
import utils
import sys
import csv
import os
#import FeatureGenByHour as Feature


mode = ""
if (len(sys.argv) == 1) or ((len(sys.argv) == 2) and (sys.argv[1] == 'manual')):
	mode = "manual"
else:
	if ((len(sys.argv) == 2) and (sys.argv[1] == 'auto')):
		mode = 'auto'
	else:
		logging.info("Please choose manual or auto.")
		sys.exit(0)

def submit(mode, now_time, result_dir):
	while True:
		submit_success = 0
		error_in_main = 1
		try:
			os.system("rm -rf our_submission_bj.csv")
			os.system("rm -rf our_submission_ld.csv")
			os.system("python3 main.py")
			os.system("python3 main_london.py")
		finally:
			if os.path.exists("our_submission_bj.csv") and os.path.exists("our_submission_ld.csv"):
				error_in_main = 0
		error_after_time = time.gmtime()
		if (error_after_time.tm_mday != now_time.tm_mday):
			break
		if error_in_main == 1:
			continue
		file_last_time_bj = open("feature_last_time_bj.txt","r").readlines()
		file_last_time_ld = open("feature_last_time_ld.txt","r").readlines()
		feature_bj = int(file_last_time_bj[0].strip())
		feature_ld = int(file_last_time_ld[0].strip())
		#file_last_time_bj.close()
		#file_last_time_ld.close()

		feature_bj = utils.ConvertNum2Time(feature_bj)
		feature_ld = utils.ConvertNum2Time(feature_ld)
		file_bj = csv.reader(open('our_submission_bj.csv','r'))
		file_ld = csv.reader(open('our_submission_ld.csv','r'))
		final_file = "our_submission_%d_%d_%d_%d_%d_%d.csv" % (feature_bj.tm_mon, feature_bj.tm_mday, feature_bj.tm_hour, feature_ld.tm_mon, feature_ld.tm_mday, feature_ld.tm_hour)
		origin_sum = open(result_dir + final_file, 'w')
		file_sum = csv.writer(origin_sum)

		for line in file_bj:
			file_sum.writerow([line[0], line[1], line[2], line[3]])

		cnt_line = 0
		for new_line in file_ld:
			if cnt_line > 0:
				file_sum.writerow([new_line[0], new_line[1], new_line[2]])
			cnt_line += 1

		origin_sum.close()

		logging.info(final_file)
		run_after_time = time.gmtime()
		if run_after_time.tm_mday != now_time.tm_mday:
			break

		'''
		while (submit_success == 0):
			files={'files': open(result_dir + final_file,'rb')}
			data = {
    		"user_id": "kepei1106",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
    		"team_token": "6942c63e663b0e0bd2bf422d6b31bd5d57cdc52a474bf17c13e4355928124245", #your team_token.
    		"description": 'Naive Model',  #no more than 40 chars.
    		"filename": final_file, #your filename
			}

			url = 'https://biendata.com/competition/kdd_2018_submit/'
			response = requests.post(url, files=files, data=data)
			print(response.text)
			print(time.gmtime())
			if "true" in response.text:
				submit_success = 1
				break
			now_submit_time = time.gmtime()
			if (now_submit_time.tm_mday != now_time.tm_mday):
				submit_success = 1
				break
			if (mode == "manual"):
				submit_success = 1
				break
		'''

		submit_success = 1
		if (submit_success == 1):
			break

pre_time = time.gmtime()

result_dir = './result/'

if mode == "manual":
	submit(mode, time.gmtime(), result_dir)
else:
	while True:
		while True:
			now_time = time.gmtime()
			if ((now_time.tm_hour >= 20) and (pre_time.tm_hour < 20)) or ((now_time.tm_hour >= 23) and (pre_time.tm_hour < 23)):
			#if ((now_time.tm_min >= 57) and (pre_time.tm_min < 57)) or ((now_time.tm_min >= 30) and (pre_time.tm_min < 30)):
				break
			#time.sleep(10)
			pre_time = now_time
		submit(mode, now_time, result_dir)
		pre_time = now_time