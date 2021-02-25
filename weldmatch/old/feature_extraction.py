#python3
# -*- coding: utf-8 -*-
"""
Created on 2020/9/8 1:51 AM
@author  : XFBY
@Software: PyCharm
"""
import cv2
import imageio
from tools.feature import feature_extract
import pickle,hashlib
import glob,os
from tqdm import tqdm


cache_root_dir = 'save_feature_21_1000'
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

def cut_edges(img):
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
    imgx = img[200:-200,lr[0]+1000:lr[1]-1000]  # 200:-200
    return imgx

#ubuntu下改为‘/’
weld_path = '/home/aijr/weld/D/21/'
@cache
def feature_ex(weld_path):
    pic_feature = {}
    pic_list = glob.glob(weld_path+'*.jpg')
    print('A total of {} welding images were found to be extracted !'.format(len(pic_list)))
    for pic_path in tqdm(pic_list):
        pic_name = pic_path.split('/')[-1]#ubuntu下改为‘/’
        print(pic_name)
        image = imageio.imread(pic_path)
        imagec = cut_edges(image)
        imagem = cv2.resize(imagec, (13000, 250))
        kps, sco, des = feature_extract(imagem,  nfeatures = -1)
        pic_feature[pic_name] = (kps, sco, des)
    return pic_feature

feature_ex(weld_path)