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
import traceback
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
								max_trials=200)

	return locations_1_to_use,locations_2_to_use,sum(inliers),np.nonzero(inliers)[0]

def match_rules(matchess,locations_1_to_use,locations_2_to_use,imgNames:tuple=None):
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
	# if low_ju>=0.75 and diff_down<0.8 and down_ju<0.7 and diff_up<0.86:
	if low_ju>=0.75 and diff_down<0.9 and down_ju<0.7 and diff_up<0.86:
		return 1
	else:
		with open('./abs930.txt', 'a',encoding='utf-8') as f:
			f.write(imgNames[0]+'>>>>>>>>'+imgNames[1]+'>>>>>>>>'+'({},{},{},{})'.format(low_ju,diff_down,down_ju,diff_up)+'\n')
		return 0



def pool_count(item0,item1,kps_left,des_left,kps_right,des_right):
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=40)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des_left, des_right, k=2)
	locations_1,locations_2,best_point,np_inline = count_best_point(matches,kps_left,kps_right)
	if best_point<40:
		return []
	matchess = np.column_stack((np_inline, np_inline))
	mark = match_rules(matchess,locations_1,locations_2,imgNames=(item0,item1))
	# # if mark==1:
	# return [item0,item1,len(matchess),mark]
	# # else:
	# # 	return []
	if mark==1:
		# return [item0,item1]
		return [item0,item1,len(matchess),mark]
	else:
		return []

def split_list(alist, parts=1):
	length = len(alist)
	return [alist[i * length // parts: (i + 1) * length // parts] for i in range(parts)]

if __name__ == '__main__':
	fs = "sim_result_19_5000.txt"
	weld_path = '/home/aijr/weld/D/21/'
	with open('./save_feature_test200600/3c2ad152b7bbd7688ab73545fdb039d8.dump', 'rb') as fr:
		pic_feature = pickle.load(fr)

	txt_path = 'allFakeImg.txt'
	txt_pathx = 'allFakeImg.txt'

	num = 0;getx = 0;getxa=0
	group = []
	for line in open(txt_path):
		ss = str(line).split('>>>>>>>>')
		pic1 = ss[0]
		pic2 = ss[1]
		group.append([pic1,pic2])
		for linea in open(txt_pathx, encoding='gbk'):
			ss = str(linea).split('>>>>>>>>')
			gga = [ss[0], ss[1][:-1]]
			if pic1 in gga and pic2 in gga:
				getxa += 1


	result = []
	error_pic = []
	t_start = time.time()
	# group_sp = split_list(group,10)
	tt = time.time()
	pool = Pool(16);pool_list = []
	for item in tqdm(group):
		# print(item)
		try:
			tmp0 = item[0].replace('\n','')
			tmp1 = item[1].replace('\n','')
			kps_left, sco_left, des_left = pic_feature[tmp0]  # pic_feature['XQⅡ-GH006+M088-02.jpg']#
			kps_right, sco_right, des_right = pic_feature[tmp1]
			# print('Feature_extract left: %6.3f,right %6.3f' % (len(kps_left), len(kps_right)))
			resultspool = pool.apply_async(pool_count, (tmp0, tmp1, kps_left, des_left, kps_right, des_right))
			pool_list.append(resultspool)
		except Exception :
			# print('fuck!')
			traceback.print_exc()
			error_pic.append([item[0], item[1]])
	print('成功创建进程数：{} !'.format(len(pool_list)))

	with open(fs, "a") as f:
		for pr in tqdm(pool_list):
			# f.write(pr[0] + '=======' + pr[1] + '=======' + pr[2] + '=======' + pr[3] + "\n")
			re_list = pr.get()
			print('=======> 已发现 {} 组疑似假片 !！<========\n'.format(num))
			if re_list:
				num+=1
				result.append(re_list)
				f.write(str(re_list[0]) + '=======' + str(re_list[1]) + '=======' + str(re_list[2]) + '=======' + str(re_list[3]) + "\n")
				for liner in open(txt_pathx, encoding='utf8'):
					ss = str(liner).split('>>>>>>>>')
					gg = [ss[0], ss[1][:-1]]
					if re_list[0] in gg and re_list[1] in gg:
						getx+=1
		print('找到 {} 组疑似假片!'.format(len(result)))
		print('总耗时：{} !'.format(time.time() - tt))
		data = pd.DataFrame(result, columns=['pic1', 'pic2','matches','mark'])
		data.to_excel("sim_result_21bd_checks_203.xls")
		pool.close()
		pool.join()
		print(error_pic)
		print(len(error_pic))
		print('all in {}'.format(getxa))
		print(getx)

