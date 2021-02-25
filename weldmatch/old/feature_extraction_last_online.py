#python3
# -*- coding: utf-8 -*-
"""
Created on 2020/9/8 1:51 AM
@author  : XFBY
@Software: PyCharm
"""
import cv2
from PIL import Image
import imageio
from tools.feature import feature_extract
import pickle,hashlib
import glob,os
from tqdm import tqdm
import numpy as np

cache_root_dir = 'save_feature_last'
if not os.path.exists(cache_root_dir):
	os.makedirs(cache_root_dir)
def md5(s):
	m = hashlib.md5()
	m.update(s.encode("utf8"))
	return m.hexdigest()
def cache_key(f, *args, **kwargs):
	s = '%s-%s-%s' % (f.__name__, str(args), str(kwargs))
	return os.path.join(cache_root_dir, '%s.dump' % md5(s))
def cache(f):
	def wrap(*args, **kwargs):
		fn = cache_key(f, *args, **kwargs)
		if os.path.exists(fn):
			print('loading cache')
			with open(fn, 'rb') as fr:
				return pickle.load(fr)

		obj = f(*args, **kwargs)
		with open(fn, 'wb') as fw:
			pickle.dump(obj, fw,protocol=4)
		return obj
	return wrap

def cut_edges(img,size):
    height, width = img.shape[:2]
    lr = []
    for left in range(0, width):
        if img[:, left].mean() < 200:
            lr.append(left)
            break
    for right in range(width - 1, 0, -1):
        if img[:, right].mean() < 200:
            lr.append(right)
            break
    imgx = img[200:-200,lr[0]+size:lr[1]-size]  # 200:-200
    return imgx

#ubuntu下改为‘/’
error = []
class_15k = [];class_7k = [];class_2d8k=[]
weld_path = '/home/aijr/weld/D/21/'#'D:/JunRui-Another/test/0000111.jpg'#'D:/JunRui-Another/imgcompare-master/images/stilllife_getty.jpg'#'D:/JunRui-Another/test/00001.jpg'
@cache
def feature_ex(weld_path):
    pic_feature = {}
    pic_list = glob.glob(weld_path + '*.jpg')
    print('A total of {} welding images were found to be extracted !'.format(len(pic_list)))
    for pic_path in tqdm(pic_list):
        pic_name = pic_path.split('/')[-1]  # ubuntu下改为‘/’
        print(pic_name)
        try:
            image = imageio.imread(pic_path)
        except:
            error.append(pic_path)
            continue

        if image.shape[1]>15000:
            imagec = cut_edges(image,1000)
            imagem = cv2.resize(imagec, (13000, 300))
            class_15k.append(pic_name)
            ss = 13000
        elif image.shape[1]<15000 and image.shape[1]>6500:
            imagec = cut_edges(image,100)
            imagem = cv2.resize(imagec, (6800, 300))
            class_7k.append(pic_name)
            ss = 6800
        else:
            imagec = cut_edges(image, 0)
            imagem = cv2.resize(imagec, (2800, 300))
            class_2d8k.append(pic_name)
            ss = 2800
        # cv2.imwrite('imgx.png', imagem)
        # break
        kps, sco, des = feature_extract(imagem,  nfeatures = -1)
        pic_feature[pic_name] = [ss,(kps, sco, des)]
        # print(len(pic_feature))
        # print(len(error))
    return pic_feature

feature_ex(weld_path)
print(len(error))
print(error)
print('15k:',len(class_15k))
print('7k:',len(class_7k))
print('2d8k:',len(class_2d8k))