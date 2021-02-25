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
								max_trials=100)

	return locations_1_to_use,locations_2_to_use,sum(inliers),np.nonzero(inliers)[0]

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
	diff_up_left = [abs(x - y) for x, y in zip(up_left[0:-1], up_left[1:])]
	diff_down_right = [abs(x - y) for x, y in zip(down_right[0:-1], down_right[1:])]
	diff_up_down = [abs(u - d) for u, d in zip(diff_up_left, diff_down_right)]
	low_ju = len(list(filter(lambda x: x < 50, diff_up_down))) / len(diff_up_down)  # 上下点平行关系
	up_ju = len(list(filter(lambda x: x > 400, up_ud))) / len(up_ud)  # 匹配到数字点的情况---一般在最下方,resize到600阈值给450
	down_ju = len(list(filter(lambda x: x > 400, down_ud))) / len(down_ud)  # 另一张图片
	diff_up = len(list(filter(lambda x: x < 100, diff_up_left))) / len(diff_up_left)
	diff_down = len(list(filter(lambda x: x < 100, diff_down_right))) / len(diff_down_right)
	if low_ju>=0.75 and diff_down<0.8 and down_ju<0.7 and diff_up<0.86:
		return 1
	else:
		return 0



def pool_count(item0,item1,kps_left,des_left,kps_right,des_right):
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=40)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des_left, des_right, k=2)
	locations_1,locations_2,best_point,np_inline = count_best_point(matches,kps_left,kps_right)
	if best_point<35:
		return []
	matchess = np.column_stack((np_inline, np_inline))
	mark = match_rules(matchess,locations_1,locations_2)
	if mark==1:
		return [item0,item1]
	else:
		return []

def split_list(alist, parts=1):
	length = len(alist)
	return [alist[i * length // parts: (i + 1) * length // parts] for i in range(parts)]

if __name__ == '__main__':
	f = "sim_result_test.txt"
	# weld_path = '/home/aijr/cnnf/all_long/'
	with open('./save_feature_21bd/db4fc4158b43b05f1eb520a0995986b7.dump', 'rb') as fr:
		pic_feature = pickle.load(fr)
	txt_path = 'result.txt'
	group = []
	for line in open(txt_path,encoding='utf8'):
		ss = str(line).split('>>>>>>>>')
		group.append([ss[0],ss[1][:-1]])
	# random.shuffle(group)
	num = 0;repi = 0
	result = []
	error_pic = []
	t_start = time.time()
	# group_sp = split_list(group,10)
	
	pool = Pool(8);pool_list = []
	for item in group:
		# print(item)
		try:
			kps_left, sco_left, des_left = pic_feature[item[0]]  # pic_feature['XQⅡ-GH006+M088-02.jpg']#
			kps_right, sco_right, des_right = pic_feature[item[1]]
			# print('Feature_extract left: %6.3f,right %6.3f' % (len(kps_left), len(kps_right)))
			resultspool = pool.apply_async(pool_count, (item[0], item[1], kps_left, des_left, kps_right, des_right))
			pool_list.append(resultspool)
		except:
			print('fuck!')
	print('成功创建进程数：{} !'.format(len(pool_list)))
	tt = time.time()
	with open(f, "a") as f:
		for pr in tqdm(pool_list):
			re_list = pr.get()
			if re_list:
				num+=1
				print('找到第 {} 组假片{}!'.format(num,re_list))
				result.append(re_list)
				f.write(re_list[0] + '=======' + re_list[1] + "\n")
		data = pd.DataFrame(result, columns=['pic1', 'pic2'])
		data.to_excel("sim_result_test.xls")
		pool.close()
		pool.join()
		print('找到 {} 组假片!'.format(len(result)))
		print('总耗时：{} !'.format(time.time() - tt))
		# print(result)
