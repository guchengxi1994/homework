#python3
# -*- coding: utf-8 -*-
"""
Created on 2020/9/8 1:23 PM
@author  : XFBY
@Software: PyCharm
"""
import cv2
import numpy as np
import imageio
import time
from skimage import measure
from skimage import transform
import os
from itertools import combinations
import pandas as pd
import random
from tqdm import tqdm
import pickle
from multiprocessing import Pool
from sklearn.externals import joblib

_RESIDUAL_THRESHOLD = 50#最佳适配值，不轻易改
def count_best_point(matches,kps_leftc,kps_rightc):
	goodMatch = []
	locations_1_to_use = []
	locations_2_to_use = []
	disdif_avg = 0
	# 统计平均距离差
	for m, n in matches:
		disdif_avg += n.distance - m.distance
	disdif_avg = disdif_avg / len(matches)
	for m, n in matches:
		# 自适应阈值
		if n.distance > m.distance + disdif_avg:
			goodMatch.append(m)
			p2 = cv2.KeyPoint(kps_rightc[m.trainIdx][0], kps_rightc[m.trainIdx][1], 1)
			p1 = cv2.KeyPoint(kps_leftc[m.queryIdx][0], kps_leftc[m.queryIdx][1], 1)
			locations_1_to_use.append([p1.pt[0], p1.pt[1]])
			locations_2_to_use.append([p2.pt[0], p2.pt[1]])
	# goodMatch = sorted(goodMatch, key=lambda x: x.distance)
	# print('match num is %d' % len(goodMatch))
	locations_1_to_use = np.array(locations_1_to_use)
	locations_2_to_use = np.array(locations_2_to_use)
	#RANSAC
	_, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
								transform.AffineTransform,
								min_samples=3,
								residual_threshold=_RESIDUAL_THRESHOLD,
								max_trials=101)
	try :
		return locations_1_to_use,locations_2_to_use,sum(inliers),np.nonzero(inliers)[0]
	except:
		return 0,0,0,0

def match_rules(matchess,locations_1_to_use,locations_2_to_use):
	up_left = []
	down_right = []
	up_ud = []
	down_ud = []
	for i in range(matchess.shape[0]):
		idx1 = matchess[i, 0]
		idx2 = matchess[i, 1]
		up_left.append(int(locations_1_to_use[idx1, 0]))
		down_right.append(int(locations_2_to_use[idx2, 0]))
		up_ud.append(int(locations_1_to_use[idx1, 1]))
		down_ud.append(int(locations_2_to_use[idx2, 1]))

	up_left, down_right = (list(t) for t in zip(*sorted(zip(up_left, down_right))))  # 排序
	# rep_up = [abs(x - y) for x, y in zip(up_left[0:-1], up_left[1:])]
	diff_up_left = [max(1, abs(x - y)) for x, y in zip(up_left[0:-1], up_left[1:])]
	diff_down_right = [max(1, abs(x - y)) for x, y in zip(down_right[0:-1], down_right[1:])]
	diff_up_down = [abs((u / d) - 1) for u, d in zip(diff_up_left, diff_down_right)]
	dt = (len(diff_up_down) - (len(down_right) - len(set(down_right))))
	if dt / len(diff_up_down) < 0.5:
		return 0, 9999, 0, 0, 0, 0, 0, 0, 0
	down100_num = len(list(filter(lambda x: x < 100, down_right)))
	if down100_num / len(down_right) >= 0.4:
		return 0, 9999, 0, 0, 0, 0, 0, 0, 0
	up1_num = len(list(filter(lambda x: x > 1, diff_up_down)))
	lendud = len(diff_up_down)
	pp_num02 = len(list(filter(lambda x: x < 0.2, diff_up_down)))
	if pp_num02 < 4:
		return 0, 9999, 0, 0, 0, 0, 0, 0, 0
	low_ju = pp_num02 / (len(diff_up_down))  # 上下点平行关系-(len(up_left)-len(set(up_left))))
	if low_ju < 0.4:
		return 0, 9999, 0, 0, 0, 0, 0, 0, 0
	diff_up = len(list(filter(lambda x: x < 200, diff_up_left))) / len(diff_up_left)
	diff_down = len(list(filter(lambda x: x < 200, diff_down_right))) / len(diff_down_right)
	if diff_down > 0.86:
		return 0, 9999, 0, 0, 0, 0, 0, 0, 0

	return 1, low_ju, up1_num, lendud, pp_num02, diff_up, diff_down, min(diff_up_down), max(diff_up_down)


def pool_count(item0,item1,kps_left,des_left,kps_right,des_right,rfc):
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=40)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des_left, des_right, k=2)
	locations_1,locations_2,best_point,np_inline = count_best_point(matches,kps_left,kps_right)
	if best_point<10:
		return [],0
	matchess = np.column_stack((np_inline, np_inline))
	mark, low_ju, up1_num, lendud, pp_num02, diff_up, diff_down, min_diff_up_down, max_diff_up_down = match_rules(
		matchess, locations_1, locations_2)
	if mark == 0:
		return [],best_point
	temp_df = pd.DataFrame(
		[[best_point, low_ju, up1_num, lendud, pp_num02, diff_up, diff_down, min_diff_up_down, max_diff_up_down]],
		columns=['match_count', 'low_ju', 'up1_num', 'lendud', 'pp_num02', 'diff_up', 'diff_down', 'min_diff_up_down',
				 'max_diff_up_down'])
	label = rfc.predict(temp_df)

	if label==1:
		return [item0,item1],best_point
	else:
		return [],best_point

def split_list(alist, parts=1):
	length = len(alist)
	return [alist[i * length // parts: (i + 1) * length // parts] for i in range(parts)]

def split_bd_pic(path,part_num):
	all_pic = os.listdir(path)
	# txt_path = 'result.txt'
	# result = []
	# for line in open(txt_path, encoding='utf8'):
	# 	ss = str(line).split('>>>>>>>>')
	# 	pic1 = ss[0]
	# 	pic2 = ss[1][:-1]
	# 	result.append(pic1)
	# 	result.append(pic2)
	# print('已有相似：{} 组 ！'.format(len(result)))
	# print(len(all_pic))
	# all_picl = []
	# for pic in all_pic:
	# 	if not pic in result:  # not pic in result and
	# 		all_picl.append(pic)
	# 	if pic in result:
	# 		print('pass:',pic)
	# print('all to match :', len(all_pic))
	# print('perare to match :',len(all_picl))
	# all_picl = [pic not in result for pic in all_pic]
	# random.shuffle(all_pic)
	print('共筛选出{}个待组合图片!'.format(len(all_pic)))
	split_part = split_list(all_pic, part_num)
	return split_part

if __name__ == '__main__':
	fs_excel = [["sim_1000_21bd_prefict1.txt","sim_1000_21bd_prefict1.xls"],
				["sim_1000_21bd_prefict2.txt","sim_1000_21bd_prefict2.xls"]]
	weld_path = '/home/aijr/weld/D/21/'
	with open('./save_feature_last/db4fc4158b43b05f1eb520a0995986b7.dump', 'rb') as fr:
		pic_feature = pickle.load(fr)

	pic_list_all = split_bd_pic('/home/aijr/weld/D/21/',2)
	print('原21标段共拆分为 {} 块 ！'.format(len(pic_list_all)))
	# print(pic_list_all)
	for i in range(0,2):
		print('===============>> 计算第{}数据块 !'.format(i+1))
		group = list(combinations(pic_list_all[i], 2))
		# random.shuffle(group)
		random.shuffle(group)
		num = 0
		result = []
		error_pic = []
		no=0
		# group_sp = split_list(group,10)
		tt = time.time()#
		pool = Pool(18);pool_list = []
		rfc = joblib.load('rc01x3d195.pkl')
		for item in tqdm(group):
			# print(item)
			# try:
			if item[0].split('-')[1].split('+')[0] == item[1].split('-')[1].split('+')[0]:
				continue
			if not item[0][-6:-4] == item[1][-6:-4]:
				continue
			try:
				left_result = pic_feature[item[0]]
				right_result = pic_feature[item[1]]
			except:
				error_pic.append([item[0],item[1]])
			if left_result[0] != right_result[0]:
				no+=1
				continue
			kps_left, sco_left, des_left = left_result[1]  # pic_feature['XQⅡ-GH005+M205-02.jpg']
			kps_right, sco_right, des_right = right_result[1]

			# print(item[0])
			# print(item[1])
			# print('Feature_extract left: %6.3f,right %6.3f' % (len(des_left), len(des_right)))
			resultspool = pool.apply_async(pool_count, (item[0], item[1], kps_left, des_left, kps_right, des_right,rfc))
			pool_list.append(resultspool)
			# except:
			# 	print('fuck!')
		print('成功分配任务数：{} !'.format(len(pool_list)))

		with open(fs_excel[i][0], "a") as f:
			for pr in tqdm(pool_list):
				re_list,best_point = pr.get()
				if re_list:
					num+=1
					result.append([re_list[0],re_list[1],best_point])
					f.write(re_list[0] + '=======' + re_list[1] + '======='+ str(best_point) + "\n")
					print('=======> 已发现 {} 组疑似假片 !！<========\n'.format(num))
			print('找到 {} 组疑似假片!'.format(len(result)))
			print('总耗时：{} !'.format(time.time() - tt))
			data = pd.DataFrame(result, columns=['pic1', 'pic2','best_point'])
			data.to_excel(fs_excel[i][1])
			pool.close()
			pool.join()
