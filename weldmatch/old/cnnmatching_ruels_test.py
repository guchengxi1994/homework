import argparse
import cv2
import numpy as np
import imageio
import plotmatch
# from lib.cnn_feature import cnn_feature_extract
import matplotlib.pyplot as plt
import time
from skimage import measure
from skimage import transform
from PIL import Image
import math
import pickle
import glob

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
    imgx = img[200:-200,lr[0]+800:lr[1]-800]  # 200:-200
    return imgx
#time count
start = time.perf_counter()

_RESIDUAL_THRESHOLD = 50
#Test1nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
# read = 'D:\\JunRui-Another\\all_long\\'#'D:/JunRui-Another/test/0000111.jpg'#'D:/JunRui-Another/imgcompare-master/images/stilllife_getty.jpg'#'D:/JunRui-Another/test/00001.jpg'
imgfile1= 'E:\\zhongshiyou\\21\\XQⅡ-EL015+4+M016-02.jpg'#'D:/JunRui-Another/imgcompare-master/images/stilllife_img1.pn
imgfile2 = 'E:\\zhongshiyou\\21\\XQⅡ-EM034+M039-02.jpg'
# pic_list = glob.glob(read+'*.jpg')
# print(pic_list)
# imgfile1 = pic_list[0]
# imgfile2 = pic_list[1]
#
start = time.perf_counter()

# read left image
# img1 = Image.open(imgfile1)
# img2 = Image.open(imgfile2)
# image1 = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2BGR)
# image2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)

# image1 = imageio.imread(imgfile1)
# image2 = imageio.imread(imgfile2)
# image1 = cut_edges(image1)
# image2 = cut_edges(image2)
# image1 = cv2.resize(image1, (13000, 300))
# image2 = cv2.resize(image2, (13000, 300))
# cv2.imwrite('img1.png',image1)
# cv2.imwrite('img2.png',image2)


print('read image time is %6.3f' % (time.perf_counter() - start))

start0 = time.perf_counter()
with open('./save_feature_21_800/db4fc4158b43b05f1eb520a0995986b7.dump', 'rb') as fr:
    pic_feature = pickle.load(fr)
kps_left, sco_left, des_left = pic_feature['XQⅡ-EM049+M071-W-01.jpg']  # pic_feature['XQⅡ-GH006+M088-02.jpg']#
kps_right, sco_right, des_right = pic_feature['XQⅡ-EK001+M002-W-01.jpg']

print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(kps_left), len(kps_right)))
start = time.perf_counter()

#Flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=40)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_left, des_right, k=2)

goodMatch = []
locations_1_to_use = []
locations_2_to_use = []

# 匹配对筛选
min_dist = 1000
max_dist = 0
disdif_avg = 0
# 统计平均距离差
for m, n in matches:
    disdif_avg += n.distance - m.distance
disdif_avg = disdif_avg / len(matches)

for m, n in matches:
    #自适应阈值
    if n.distance > m.distance + disdif_avg:
        goodMatch.append(m)
        p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
        p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
        locations_1_to_use.append([p1.pt[0], p1.pt[1]])
        locations_2_to_use.append([p2.pt[0], p2.pt[1]])
#goodMatch = sorted(goodMatch, key=lambda x: x.distance)
print('match num is %d' % len(goodMatch))
locations_1_to_use = np.array(locations_1_to_use)
locations_2_to_use = np.array(locations_2_to_use)

# Perform geometric verification using RANSAC.
_, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                          transform.AffineTransform,
                          min_samples=3,
                          residual_threshold=_RESIDUAL_THRESHOLD,
                          max_trials=100)

print('Found %d inliers' % sum(inliers))

inlier_idxs = np.nonzero(inliers)[0]
#最终匹配结果
matchess = np.column_stack((inlier_idxs, inlier_idxs))


'''
def match_rules(matchess,locations_1_to_use,locations_2_to_use):
    up_left = []
    down_right = []
    for i in range(matchess.shape[0]):
        idx1 = matchess[i, 0]
        idx2 = matchess[i, 1]
        up_left.append(locations_1_to_use[idx1, 0])
        down_right.append(locations_2_to_use[idx2, 0])
    diff_up_left = [abs(x - y) for x, y in zip(up_left[0:-1], up_left[1:])]
    diff_down_right = [abs(x - y) for x, y in zip(down_right[0:-1], down_right[1:])]
    diff_up = len(list(filter(lambda x:x<100, diff_up_left)))/len(diff_up_left)
    diff_down = len(list(filter(lambda x:x<100, diff_down_right)))/len(diff_down_right)
    print(diff_up_left)
    print(diff_down_right)
    print(diff_up)
    print(diff_down)
    if diff_up >= 0.5 or diff_down >=0.5:
        return 'nosim'
    else:
        return 'sim'
'''
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

    up_left,down_right = (list(t) for t in zip(*sorted(zip(up_left,down_right))))#排序
    diff_up_left = [abs(x - y) for x, y in zip(up_left[0:-1], up_left[1:])]
    diff_down_right = [abs(x - y) for x, y in zip(down_right[0:-1], down_right[1:])]
    diff_up_down = [abs(u-d) for u,d in zip(diff_up_left,diff_down_right)]
    low_ju = len(list(filter(lambda x: x < 45, diff_up_down))) / len(diff_up_down)#上下点平行关系
    up_ju = len(list(filter(lambda x: x > 400, up_ud))) / len(up_ud)  #匹配到数字点的情况---一般在最下方,resize到600阈值给450
    down_ju = len(list(filter(lambda x: x > 400, down_ud))) / len(down_ud)  #另一张图片
    diff_up = len(list(filter(lambda x: x < 105, diff_up_left))) / len(diff_up_left)
    diff_down = len(list(filter(lambda x: x < 105, diff_down_right))) / len(diff_down_right)
    jj  = sorted(down_right)
    ab_dis = (jj[-5]-jj[5])/15000
    print(diff_up_left)
    print(up_left)
    print(down_right)
    print(diff_down_right)
    print(diff_up_down)
    print('平行度  {}'.format(low_ju))

    print(up_ud)
    print(down_ud)
    print('点靠下程度up  {}'.format(up_ju))
    print('点靠下程度down  {}'.format(down_ju))

    print('点聚集程度up  {}'.format(diff_up))
    print('点聚集程度down  {}'.format(diff_down))
    print(jj)
    print(ab_dis)
    if low_ju>=0.75 and diff_down<0.69 and diff_up<0.69:
        print('sim')
    else:
        # if up_ju>=0.45 or down_ju>=0.45:
        #     print('nosim')
        # else:
        print('nosim')


t = time.time()
print(match_rules(matchess,locations_1_to_use,locations_2_to_use))
print(time.time()-t)
