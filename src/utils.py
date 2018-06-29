# -*- coding: UTF-8 -*-

import time, datetime, calendar
import numpy as np

# 将数字转化成time对象
def ConvertNum2Time(t, time_base = '2017-01-01 00:00:00', timezone_offset = 0):
	array_base = time.strptime(time_base, '%Y-%m-%d %H:%M:%S')
	stamp_base = int(calendar.timegm(array_base))
	stamp_t = stamp_base + t*3600
	array_utc = time.gmtime(stamp_t + timezone_offset*3600)
	return array_utc

# 将字符串转化为数字
def ConvertStr2Num(s, time_base = '2017-01-01 00:00:00', timezone_offset = 0):
	if s.find('/') == -1:
		array_s = time.strptime(s, '%Y-%m-%d %H:%M:%S')
	else:
		array_s = time.strptime(s, '%Y/%m/%d %H:%M')
	return ConvertTime2Num(array_s, time_base, timezone_offset)

# 将time对象转化成数字
def ConvertTime2Num(array_s, time_base = '2017-01-01 00:00:00', timezone_offset = 0):
	#time_base = '2017-01-01 00:00:00'
	array_base = time.strptime(time_base, '%Y-%m-%d %H:%M:%S')
	stamp_base = int(calendar.timegm(array_base))
	stamp_s = int(calendar.timegm(array_s))
	return int((stamp_s - stamp_base)//3600) - timezone_offset

# 找time block的交集
def joinBlock(*blocks):
	if len(blocks) == 1:
		for s, e in blocks[0]:
			yield s, e
	else:
		try:
			B = iter(joinBlock(*blocks[1:]))
			Bs, Be = next(B)

			for As, Ae in blocks[0]:
				while True:
					if Bs >= Ae:
						break
					if Be <= As:
						Bs, Be = next(B)
						continue
					yield max(As, Bs), min(Ae, Be)
					if Be >= Ae:
						break
					else:
						Bs, Be = next(B)
			raise StopIteration
		except StopIteration:
			raise

def generateBlock(block, blocklen, overlap, mins=0, reference=None):
	rid = 0
	for s, t in block:
		now = max(s, mins)
		if now >= t:
			continue
		while True:
			if now + overlap >= t:
				break

			end = t
			if reference and rid < len(reference):
				assert now + overlap <= reference[rid][0]
				if now + overlap == reference[rid][0]:
					end = reference[rid][1]
					rid += 1
				else:
					end = reference[rid][0]

			if now + blocklen < end:
				end = now + blocklen

			yield now, end
			now = end-overlap

def calSMAPE(Y, pred_Y, time=72):
	pred_Y = np.maximum(pred_Y, 0)
	ans = []
	for i in range(0, Y.shape[1], 72):
		subY = Y[:, i:i+time]
		subY_pred = pred_Y[:, i:i+time]
		ans.append(np.mean(2 * np.abs(subY - subY_pred) / (subY + subY_pred + 1e-10)))
	return np.mean(ans)

def calSMAPE1(Y, pred_Y):
	pred_Y = np.maximum(pred_Y, 0)
	return np.mean(2 * np.abs(Y - pred_Y) / (Y + pred_Y + 1e-10))