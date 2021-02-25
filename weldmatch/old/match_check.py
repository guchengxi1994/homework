import argparse
import cv2
import numpy as np
import imageio
import plotmatch
from tools.feature import feature_extract
import matplotlib.pyplot as plt
import time
from skimage import measure
from skimage import transform
from PIL import Image
import glob,os

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
    imgx = img[200:-200,lr[0]:lr[1]]  # 200:-200
    return imgx
#time count


_RESIDUAL_THRESHOLD = 50

if __name__ == '__main__':
    txt_path = 'sim_result_21bd_x.txt'
    for line in open(txt_path):
        ss = str(line).split('=======')
        pic1 = ss[0]
        pic2 = ss[1][:-1]
        imgfile1= '/home/aijr/weld/D/21/'+pic1 #'D:/JunRui-Another/imgcompare-master/images/stilllife_img1.pn
        imgfile2 = '/home/aijr/weld/D/21/'+pic2

        img1 = Image.open(imgfile1)
        img2 = Image.open(imgfile2)
        image1 = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)
        # image1 = imageio.imread(imgfile1)
        # image2 = imageio.imread(imgfile2)
        image1 = cut_edges(image1)
        image2 = cut_edges(image2)
        image1 = cv2.resize(image1, (15000, 600))
        image2 = cv2.resize(image2, (15000, 600))
        start = time.perf_counter()
        kps_left, sco_left, des_left = feature_extract(image1,  nfeatures = -1)
        kps_right, sco_right, des_right = feature_extract(image2,  nfeatures = -1)

        print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(kps_left), len(kps_right)))


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
                                  max_trials=1000)

        print('Found %d inliers' % sum(inliers))

        inlier_idxs = np.nonzero(inliers)[0]
        #最终匹配结果
        matches = np.column_stack((inlier_idxs, inlier_idxs))
        plt.rcParams['savefig.dpi'] = 100 #图片像素
        plt.rcParams['figure.dpi'] = 100 #分辨率
        plt.rcParams['figure.figsize'] = (80.0, 10.0) # 设置figure_size尺寸
        _, ax = plt.subplots()
        plotmatch.plot_matches(
            ax,
            image1,
            image2,
            locations_1_to_use,
            locations_2_to_use,
            np.column_stack((inlier_idxs, inlier_idxs)),#column
            plot_matche_points = 1,
            matchline = True,
            matchlinewidth = 0.3)
        ax.axis('off')
        ax.set_title('')
        pp = os.path.join('test_save',pic1[:-4]+'_'+pic2[:-4]+'.png')
        plt.savefig(pp)
        plt.close()

