'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2021-01-15 08:39:52
LastEditors: xiaoshuyui
LastEditTime: 2021-02-25 09:33:14
'''
import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt
from skimage import io

imageA = io.imread('D:\\projectWelds\\AWelds\\XQⅡ-GD000-M001-W-01.jpg')

imageA = cv.cvtColor(imageA, cv.COLOR_RGB2BGR)

imageB = io.imread('D:\\projectWelds\\AWelds\\XQⅡ-GD000-M001-W-01.jpg')

# imageB = cv.imread('D:\\projectWelds\\AWelds\\XQⅡ-GD000-M001-W-02.jpg')
imageB = cv.cvtColor(imageB, cv.COLOR_RGB2BGR)

print(imageA.shape)

grayA = cv.cvtColor(imageA, cv.COLOR_BGR2GRAY)

grayA = np.rot90(grayA)

# cv.imshow("grayA", grayA)

grayB = cv.cvtColor(imageB, cv.COLOR_BGR2GRAY)

grayB = np.rot90(grayB)

grayA = cv.GaussianBlur(grayA,(5,5),0)

grayB = cv.GaussianBlur(grayB,(5,5),0)

# cv.imshow("grayB", grayB)

min_hessian = 1000

sift = cv.xfeatures2d.SIFT_create(min_hessian)

keypointsA, featuresA = sift.detectAndCompute(grayA,None)

keypointsB, featuresB = sift.detectAndCompute(grayB,None)

kpImgA=cv.drawKeypoints(grayA,keypointsA,imageA)

kpImgB=cv.drawKeypoints(grayB,keypointsB,imageB)

# cv.imshow("kpImgA", kpImgA)

# cv.imshow("kpImgB", kpImgB)

FLANN_INDEX_KDTREE = 0

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

search_params = dict(checks=50)   

# 使用FlannBasedMatcher 寻找最近邻近似匹配

flann = cv.FlannBasedMatcher(index_params,search_params)

# 使用knnMatch匹配处理，并返回匹配matches

matches = flann.knnMatch(featuresA, featuresB, k=2)

matchesMask = [[0,0] for i in range(len(matches))]

coff = 0.2 # 0.1 0.7  0.8

for i,(m,n) in enumerate(matches):

    if m.distance < coff * n.distance:

        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),

                   singlePointColor = (255,0,0),

                   matchesMask = matchesMask,

                   flags = 2)

resultImg = cv.drawMatchesKnn(grayA, keypointsA, grayB,keypointsB, matches,None,**draw_params)

resultImg1 = cv.drawMatchesKnn(imageA, keypointsA, imageB,keypointsB, matches,None,**draw_params)

plt.imshow(resultImg,),plt.show()

# cv.imshow("resultImg", resultImg)

# cv.imshow("resultImg1", resultImg1)

# cv.waitKey(0)

# cv.destroyAllWindows()