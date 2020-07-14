
import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label
import cv2
# from skimage.data import astronaut
from skimage.io import imsave,imread
import copy

# from similarity_index_of_label_graph_package import similarity_index_of_label_graph_class
from networkx.generators.directed import gnr_graph
from networkx.generators import spectral_graph_forge

from utils import *
from utils2 import *

import pylab as pl
import scipy.signal as signal




def processUnet(img_or_path):
    if isinstance(img_or_path,str):
        img = imread(img_or_path)
    else:
        img = img_or_path
    
    # imgShape = img.shape
    # if imgShape[0]>imgShape[1]:
    #     pass
    # else:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fea2 = np.sum(img,axis=1).astype(np.float32)
    # print(fea2.shape)

    return fea2



def imFill(img):
    img = img*255
    img = np.array(img,dtype=np.uint8)
    im_floodfill = img.copy()
    h, w = img.shape[:2]
  
    mask = np.zeros((h+2, w+2), np.uint8)  
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);   
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    cv2.imwrite("out22222.png",im_floodfill_inv)
    # Combine the two images to get the foreground.
    im_out = img + im_floodfill_inv

    cv2.imwrite("out33333.png",im_out)


#不好用，运算时间太长
def largestConnectComponent(bw_img):
    '''
    compute largest Connect component of an labeled image

    Parameters:
    ---

    bw_img:
        binary image

    Example:
    ---
        >>> lcc = largestConnectComponent(bw_img)

    '''

    labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)    
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    for i in range(1, num): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    # x = np.where

    return lcc





def getMaxRegion(img):
    # kernel = np.ones((5,5),np.uint8)
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    x,y = stats.shape

    # for i in (1,x+1):
    # stats1 = stats[1:,:]
    maxArea = 0
    lab = 0
    for i in range(1,x):
        if stats[i][4]>maxArea:
            maxArea = stats[i][4]
            lab = i 
    
    labels[labels!=lab] = 0

    labels[labels!=0] = 1

    return labels

    
        









def xdog(im, gamma=0.98, phi=200, eps=-0.1, k=1.6, sigma=0.8, binarize=False):
    # Source : https://github.com/CemalUnal/XDoG-Filter
    # Reference : XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization
    # Link : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.151&rep=rep1&type=pdf
    try:
        if im.shape[2] == 3:
            im = rgb2gray(im)
    except Exception:
        pass
    imf1 = gaussian_filter(im, sigma)
    imf2 = gaussian_filter(im, sigma * k)
    imdiff = imf1 - gamma * imf2
    imdiff = (imdiff < eps) * 1.0  + (imdiff >= eps) * (1.0 + np.tanh(phi * imdiff))
    imdiff -= imdiff.min()
    imdiff /= imdiff.max()
    if binarize:
        th = threshold_otsu(imdiff)
        imdiff = imdiff >= th
    imdiff = imdiff.astype('float32')
    
    imgShape = imdiff.shape
    if imgShape[0]>imgShape[1]:
        imdiff[:,int(0.5*imgShape[1])] = 1
    else:
        imdiff[int(0.5*imgShape[0]),:] = 1
    # imFill(imdiff)
    return imdiff

def getShape(i2):
    s_i2 = np.sum(i2)
    s_h_i2 = np.sum(i2,axis=0)
    s = 0
    times = 0
    for i in s_h_i2:
        if s>0.5*s_i2:
            break
        else:
            s = s+i
            times = times + 1 

    # print(times)
    # # i3 = copy.deepcopy(i2)
    # print(np.max(i2))

    i2 = np.array(i2,dtype=np.float32)

    # baseLine = i2[:,times] 
    p1 = i2[:,0:times]
    p2 = i2[:,times+1:] 

    l1 = np.sum(p1,1)
    l2 = np.sum(p2,1)
    res = l1 - l2
    return res


def local_threshold(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,10)
    binary = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 0)
    return binary



def his(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    if len(image.shape) == 3:
        b, g, r = cv2.split(image)
        # 创建局部直方图均衡化
        
        # 对每一个通道进行局部直方图均衡化
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        # 合并处理后的三通道 成为处理后的图
        image = cv2.merge([b, g, r])

        return image
    else:
        return cv2.merge([clahe.apply(image),clahe.apply(image),clahe.apply(image)])

def his2(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    if len(image.shape) == 3:
        b, g, r = cv2.split(image)
        # 创建局部直方图均衡化
        
        # 对每一个通道进行局部直方图均衡化
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        # 合并处理后的三通道 成为处理后的图
        image = cv2.merge([b, g, r])
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # print(image.shape)
        # return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image
        # return rgb2gray(image)*255
        # return image
    else:
        return clahe.apply(image)
        # return cv2.merge([clahe.apply(image),clahe.apply(image),clahe.apply(image)])


def process3(im):
    # im = cv2.imread(imgPath)
    im = his(im)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgIn = im / 255.0
    imgIn = xdog(imgIn, binarize=True,k=20)
 
        # imgIn = np.array(imgIn,dtype=np.uint8)*255
        # imgIn = self.getMaxRegion(self,imgIn)
    fe = imgIn * im 
    fea2 = np.sum(fe,axis=1,dtype=np.float32)

    return fea2


def process(imgPath):
    if isinstance(imgPath,str):
        im = imread(imgPath)
    else:
        im = imgPath
    # im = 
    im = his(im)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    imgShape = im.shape 
    if imgShape[0]>imgShape[1]:
        pass 
    else:
        trans_img = cv2.transpose(im)
        im = cv2.flip(trans_img, 1)

    
    imgIn = im / 255.0
    imgIn = xdog(imgIn, binarize=True,k=20)
    # im = local_threshold(im)

    # cv2.imwrite("out33333.png", imgIn*im)
    fe = imgIn * im 
    fea2 = np.sum(fe,axis=1,dtype=np.float32)


    img = np.array(imgIn,dtype=np.uint8)
    # print(np.max(img))
    # img[:,int(0.5*img.shape[1])] = 1
    lcc = getMaxRegion(img)
    # lcc = np.array(lcc,dtype=np.uint8)

    wL = np.sum(lcc,axis=1,dtype=np.float32)
    # wwl = (wL != 0)
    wL[wL == 0] = 1
    # w = np.nanmean(wL)
    print(wL.shape)
    # img[:,int(0.5*img.shape[1])] = 1
    # lcc = largestConnectComponent(img)
    # lcc = np.array(lcc,dtype=np.uint8)

    return lcc,1,fea2
    # return lcc,w,nomarlLize(fea2)


# def _compare_newMethod(l1,l2):
#     """
#     Successfully installed networkx-2.4 scipy-1.5.0 similarity-index-of-label-graph-2.0.1
#     """
#     # G1 = gnr_graph(100, 0.3, seed = 65535)
#     sim = similarity_index_of_label_graph_class()

#     return sim(l1,l2)


def getFea(im):
    img = copy.deepcopy(im)
    imgIn = im / 255.0
    imgIn = xdog(imgIn, binarize=True,k=20)
    fe = imgIn * img 
    fea2 = np.sum(fe,axis=1,dtype=np.float32)
    return fea2



def process4(im):
    imgShape = im.shape 
    
    if imgShape[0]>imgShape[1]:
        pass 
    else:
        trans_img = cv2.transpose(im)
        im = cv2.flip(trans_img, 1)
    # im = cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_BGR2GRAY)

    img1 = his2(im[:int(0.25*imgShape[0]),:])
    # print(np.max(img1))
    # print(img1.shape)
    img2 = his2(im[int(0.25*imgShape[0]):int(0.5*imgShape[0]),:])
    img3 = his2(im[int(0.5*imgShape[0]):int(0.75*imgShape[0]),:])
    img4 = his2(im[int(0.75*imgShape[0]):,:])


    fea1 = getFea(img1)
    fea2 = getFea(img2)
    fea3 = getFea(img3)
    fea4 = getFea(img4)

    fea = np.concatenate((fea1,fea2,fea3,fea4),axis=0)

    return fea




    


def process2(im):
    imgShape = im.shape 
    im = his(im)
    if imgShape[0]>imgShape[1]:
        pass 
    else:
        trans_img = cv2.transpose(im)
        im = cv2.flip(trans_img, 1)

    # im = white_balance_5(im)
    im = cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_RGB2GRAY)
    # im = cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_RGB2BGR)

    # im = homomorphic_filter(im)
    # cv2.imwrite("D:\\testALg\\homework\\house\\227\out777777777.png", im)
    # im = unevenLightCompensate(im,32)
    # im = deHaze(im)

    imgIn = im / 255.0
    imgIn = xdog(imgIn, binarize=True,k=20)

    # imgIn = me(im,mask=imgIn)
    # imgIn = getMaxRegion(imgIn.astype(np.uint8))
    # cv2.imwrite("D:\\testALg\\homework\\house\\227\\out6666.png", 255*imgIn)

    fe = imgIn * im 
    fea2 = np.sum(fe,axis=1,dtype=np.float32)
    # print(len(fea2))
    return fea2


def smooth(a,WSZ = 3):
  # a:原始数据，NumPy 1-D array containing the data to be smoothed
  # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
  # WSZ: smoothing window size needs, which must be odd number,
  # as in the original MATLAB implementation
  out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
  r = np.arange(1,WSZ-1,2)
  start = np.cumsum(a[:WSZ-1])[::2]/r
  stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
  return np.concatenate(( start , out0, stop ))

def nomarlLize(r):
    # print(np.min(r))
    # print(np.max(r))
    # print(w)
    # r = [x if abs(x)>5 and abs(x)<w*0.5 else 0 for x in r]
    r = np.array(r,dtype=np.float32)
    minV = np.min(r)
    maxV = np.max(r)
    return (r-minV)/(maxV - minV)

    

    # return(r)

def me(image,mask):
    shape = image.shape
    if shape[0]>shape[1]:
        pass
    else:
        trans_img = cv2.transpose(image)
        image = cv2.flip(trans_img, 1)
        shape = image.shape
    
    thres = np.mean(image,axis=1).reshape((shape[0],1))

    # kernel = np.ones((5,1),dtype=np.float32)/5
    # thres = cv2.filter2D(thres,-1,kernel)


    thres_mat = np.tile(thres,shape[1])

    image_thres = image - thres_mat

    image_thres[image_thres>0] = 1
    image_thres[image_thres!=1] = 0
    
    return image_thres * mask


def tr(img):
    # im = his(im)
    shape = img.shape
    if shape[0]>shape[1]:
        pass
    else:
        trans_img = cv2.transpose(img)
        img = cv2.flip(trans_img, 1)
        shape = img.shape
    
    return img


def unevenLightCompensate(img, blockSize=16):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    # dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst



def _compare(l1,l2):
    r11 = l1[int(0.2*len(l1)):int(0.4*len(l1))]
    r12 = l1[int(0.4*len(l1)):int(0.6*len(l1))]
    r13 = l1[int(0.6*len(l1)):int(0.8*len(l1))]


    r21 = l2[int(0.2*len(l2)):int(0.4*len(l2))]
    r22 = l2[int(0.4*len(l2)):int(0.6*len(l2))]
    r23 = l2[int(0.6*len(l2)):int(0.8*len(l2))]


    res1 = cv2.matchTemplate(l2[:int(0.6*len(l2))], r11, cv2.TM_CCOEFF_NORMED)
    _, max_val1, _, max_indx1 = cv2.minMaxLoc(res1)

    res2 = cv2.matchTemplate(l2[int(0.2*len(l2)):int(0.8*len(l2))], r12, cv2.TM_CCOEFF_NORMED)
    _, max_val2, _, max_indx2 = cv2.minMaxLoc(res2)

    res3 = cv2.matchTemplate(l2[int(0.4*len(l2)):], r13, cv2.TM_CCOEFF_NORMED)
    _, max_val3, _, max_indx3 = cv2.minMaxLoc(res3)

    print(max_val1)
    print(max_val2)
    print(max_val3)

    print('=================================')

    m1 = max(max_val1,max_val2,max_val3)

    res1 = cv2.matchTemplate(l1[:int(0.6*len(l1))], r21, cv2.TM_CCOEFF_NORMED)
    _, max_val1, _, max_indx4 = cv2.minMaxLoc(res1)

    res2 = cv2.matchTemplate(l1[int(0.2*len(l1)):int(0.8*len(l1))], r22, cv2.TM_CCOEFF_NORMED)
    _, max_val2, _, max_indx5 = cv2.minMaxLoc(res2)

    res3 = cv2.matchTemplate(l1[int(0.4*len(l1)):], r23, cv2.TM_CCOEFF_NORMED)
    _, max_val3, _, max_indx6 = cv2.minMaxLoc(res3)

    print(max_val1)
    print(max_val2)
    print(max_val3)

    # print(max_indx1[1])
    # print(int(0.2*len(l2))+max_indx2[1])
    # print(int(0.4*len(l2))+max_indx3[1])
    # print(max_indx4[1])
    # print(int(0.2*len(l1))+max_indx5[1])
    # print(int(0.4*len(l1))+max_indx6[1])

    m2 = max(max_val1,max_val2,max_val3)

    return 0.5*(m1+m2)


def _compare2(l1,l2):
    r11 = l1[int(0.1*len(l1)):int(0.26*len(l1))]
    r12 = l1[int(0.26*len(l1)):int(0.42*len(l1))]
    r13 = l1[int(0.42*len(l1)):int(0.58*len(l1))]
    r14 = l1[int(0.58*len(l1)):int(0.74*len(l1))]
    r15 = l1[int(0.74*len(l1)):int(0.9*len(l1))]


    r21 = l2[int(0.1*len(l2)):int(0.26*len(l2))]
    r22 = l2[int(0.26*len(l2)):int(0.42*len(l2))]
    r23 = l2[int(0.42*len(l2)):int(0.58*len(l2))]
    r24 = l2[int(0.58*len(l2)):int(0.74*len(l2))]
    r25 = l2[int(0.74*len(l2)):int(0.9*len(l2))]


    res1 = cv2.matchTemplate(l2[int(0.05*len(l2)):int(0.35*len(l2))], r11, cv2.TM_CCOEFF_NORMED)
    _, max_val1, _, max_indx1 = cv2.minMaxLoc(res1)

    res2 = cv2.matchTemplate(l2[int(0.2*len(l2)):int(0.5*len(l2))], r12, cv2.TM_CCOEFF_NORMED)
    _, max_val2, _, max_indx2 = cv2.minMaxLoc(res2)

    res3 = cv2.matchTemplate(l2[int(0.35*len(l2)):int(0.65*len(l2))], r13, cv2.TM_CCOEFF_NORMED)
    _, max_val3, _, max_indx3 = cv2.minMaxLoc(res3)


    res4 = cv2.matchTemplate(l2[int(0.5*len(l2)):int(0.8*len(l2))], r14, cv2.TM_CCOEFF_NORMED)
    _, max_val4, _, max_indx4 = cv2.minMaxLoc(res4)

    res5 = cv2.matchTemplate(l2[int(0.65*len(l2)):int(0.95*len(l2))], r15, cv2.TM_CCOEFF_NORMED)
    _, max_val5, _, max_indx5 = cv2.minMaxLoc(res5)

    print(max_val1)
    print(max_val2)
    print(max_val3)
    print(max_val4)
    print(max_val5)

    print('=================================')

    m1 = max(max_val1,max_val2,max_val3,max_val4,max_val5)

    res1 = cv2.matchTemplate(l1[int(0.05*len(l1)):int(0.35*len(l1))], r21, cv2.TM_CCOEFF_NORMED)
    _, max_val6, _, max_indx6 = cv2.minMaxLoc(res1)

    res2 = cv2.matchTemplate(l1[int(0.2*len(l1)):int(0.5*len(l1))], r22, cv2.TM_CCOEFF_NORMED)
    _, max_val7, _, max_indx7 = cv2.minMaxLoc(res2)

    res3 = cv2.matchTemplate(l1[int(0.35*len(l1)):int(0.65*len(l1))], r23, cv2.TM_CCOEFF_NORMED)
    _, max_val8, _, max_indx8 = cv2.minMaxLoc(res3)

    res4 = cv2.matchTemplate(l1[int(0.5*len(l1)):int(0.8*len(l1))], r24, cv2.TM_CCOEFF_NORMED)
    _, max_val9, _, max_indx9 = cv2.minMaxLoc(res4)


    res5 = cv2.matchTemplate(l1[int(0.65*len(l1)):int(0.95*len(l1))], r25, cv2.TM_CCOEFF_NORMED)
    _, max_val10, _, max_indx10 = cv2.minMaxLoc(res5)

    print(max_val6)
    print(max_val7)
    print(max_val8)
    print(max_val9)
    print(max_val10)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    a1 = int(0.05*len(l2))+max_indx1[1]
    a2 = int(0.2*len(l2))+max_indx2[1]
    a3 = int(0.35*len(l2))+max_indx3[1]
    a4 = int(0.5*len(l2))+max_indx4[1]
    a5 = int(0.65*len(l1))+max_indx5[1]
    # print(int(0.05*len(l2))+max_indx1[1])
    # print(int(0.2*len(l2))+max_indx2[1])
    # print(int(0.35*len(l2))+max_indx3[1])
    # print(int(0.5*len(l2))+max_indx4[1])
    # print(int(0.65*len(l1))+max_indx5[1])
    print(a2-a1)
    print(a3-a2)
    print(a4-a3)
    print(a5-a4)
    # print(a5-a3)

    print('===============================================')
    b1 = int(0.05*len(l1))+max_indx6[1]
    b2 = int(0.2*len(l1))+max_indx7[1]
    b3 = int(0.35*len(l1))+max_indx8[1]
    b4 = int(0.5*len(l1))+max_indx9[1]
    b5 = int(0.65*len(l1))+max_indx10[1]
    # print(int(0.05*len(l1))+max_indx6[1])
    # print(int(0.2*len(l1))+max_indx7[1])
    # print(int(0.35*len(l1))+max_indx8[1])
    # print(int(0.5*len(l1))+max_indx9[1])
    # print(int(0.65*len(l1))+max_indx10[1])
    print(b2-b1)
    print(b3-b2)
    print(b4-b3)
    print(b5-b4)
    # print(b5-b3)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    m1 = 0.5*(max_val1+max_val6)
    m2 = 0.5*(max_val2+max_val7)
    m3 = 0.5*(max_val3+max_val8)
    m4 = 0.5*(max_val4+max_val9)
    m5 = 0.5*(max_val5+max_val10)

    return max(m1,m2,m3,m4,m5)



def _compare3(l1,l2):
    r11 = l1[int(0.1*len(l1)):int(0.26*len(l1))]
    r12 = l1[int(0.26*len(l1)):int(0.42*len(l1))]
    r13 = l1[int(0.42*len(l1)):int(0.58*len(l1))]
    r14 = l1[int(0.58*len(l1)):int(0.74*len(l1))]
    r15 = l1[int(0.74*len(l1)):int(0.9*len(l1))]

    res1 = cv2.matchTemplate(l2[int(0.05*len(l2)):int(0.35*len(l2))], r11, cv2.TM_CCOEFF_NORMED)
    _, max_val1, _, max_indx1 = cv2.minMaxLoc(res1)

    res2 = cv2.matchTemplate(l2[int(0.2*len(l2)):int(0.5*len(l2))], r12, cv2.TM_CCOEFF_NORMED)
    _, max_val2, _, max_indx2 = cv2.minMaxLoc(res2)

    res3 = cv2.matchTemplate(l2[int(0.35*len(l2)):int(0.65*len(l2))], r13, cv2.TM_CCOEFF_NORMED)
    _, max_val3, _, max_indx3 = cv2.minMaxLoc(res3)


    res4 = cv2.matchTemplate(l2[int(0.5*len(l2)):int(0.8*len(l2))], r14, cv2.TM_CCOEFF_NORMED)
    _, max_val4, _, max_indx4 = cv2.minMaxLoc(res4)

    res5 = cv2.matchTemplate(l2[int(0.65*len(l2)):int(0.95*len(l2))], r15, cv2.TM_CCOEFF_NORMED)
    _, max_val5, _, max_indx5 = cv2.minMaxLoc(res5)

    print(max_val1)
    print(max_val2)
    print(max_val3)
    print(max_val4)
    print(max_val5)

    # m1 = max(max_val1,max_val2,max_val3,max_val4,max_val5)
    print('=================================')


    r21 = l2[int(0.05*len(l2))+max_indx1[1]:int(0.05*len(l2))+len(r11)+max_indx1[1]]
    r22 = l2[int(0.2*len(l2))+max_indx2[1]:int(0.2*len(l2))+len(r12)+max_indx2[1]]
    r23 = l2[int(0.35*len(l2))+max_indx3[1]:int(0.35*len(l2))+len(r13)+max_indx3[1]]
    r24 = l2[int(0.5*len(l2))+max_indx4[1]:int(0.5*len(l2))+len(r14)+max_indx4[1]]
    r25 = l2[int(0.65*len(l2))+max_indx5[1]:int(0.65*len(l2))+len(r15)+max_indx5[1]]

    

    res1 = cv2.matchTemplate(l1[int(0.05*len(l1)):int(0.35*len(l1))], r21, cv2.TM_CCOEFF_NORMED)
    _, max_val6, _, max_indx6 = cv2.minMaxLoc(res1)

    res2 = cv2.matchTemplate(l1[int(0.2*len(l1)):int(0.5*len(l1))], r22, cv2.TM_CCOEFF_NORMED)
    _, max_val7, _, max_indx7 = cv2.minMaxLoc(res2)

    res3 = cv2.matchTemplate(l1[int(0.35*len(l1)):int(0.65*len(l1))], r23, cv2.TM_CCOEFF_NORMED)
    _, max_val8, _, max_indx8 = cv2.minMaxLoc(res3)

    res4 = cv2.matchTemplate(l1[int(0.5*len(l1)):int(0.8*len(l1))], r24, cv2.TM_CCOEFF_NORMED)
    _, max_val9, _, max_indx9 = cv2.minMaxLoc(res4)


    res5 = cv2.matchTemplate(l1[int(0.65*len(l1)):int(0.95*len(l1))], r25, cv2.TM_CCOEFF_NORMED)
    _, max_val10, _, max_indx10 = cv2.minMaxLoc(res5)

    print(max_val6)
    print(max_val7)
    print(max_val8)
    print(max_val9)
    print(max_val10)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    a1 = int(0.05*len(l2))+max_indx1[1]
    a2 = int(0.2*len(l2))+max_indx2[1]
    a3 = int(0.35*len(l2))+max_indx3[1]
    a4 = int(0.5*len(l2))+max_indx4[1]
    a5 = int(0.65*len(l1))+max_indx5[1]
    # print(max_indx1[1]-max_indx6[1])
    # print(max_indx2[1]-max_indx7[1])
    # print(max_indx3[1]-max_indx8[1])
    # print(max_indx4[1]-max_indx9[1])
    # print(max_indx5[1]-max_indx10[1])
    print(a2-a1)
    print(a3-a2)
    print(a4-a3)
    print(a5-a4)
    # print(a5-a3)

    print('===============================================')
    b1 = int(0.05*len(l1))+max_indx6[1]
    b2 = int(0.2*len(l1))+max_indx7[1]
    b3 = int(0.35*len(l1))+max_indx8[1]
    b4 = int(0.5*len(l1))+max_indx9[1]
    b5 = int(0.65*len(l1))+max_indx10[1]
    # print(max_indx6[1]-max_indx1[1])
    # print(max_indx7[1]-max_indx2[1])
    # print(max_indx8[1]-max_indx3[1])
    # print(max_indx9[1]-max_indx4[1])
    # print(max_indx10[1]-max_indx5[1])
    print(b2-b1)
    print(b3-b2)
    print(b4-b3)
    print(b5-b4)
    # print(b5-b3)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # m2 = max(max_val1,max_val2,max_val3,max_val4,max_val5)
    m1 = 0.5*(max_val1+max_val6)
    m2 = 0.5*(max_val2+max_val7)
    m3 = 0.5*(max_val3+max_val8)
    m4 = 0.5*(max_val4+max_val9)
    m5 = 0.5*(max_val5+max_val10)

    return max(m1,m2,m3,m4,m5)



def np_move_avg(a,n,mode="same"):
    return (np.convolve(a, np.ones((n,))/n, mode=mode))



if __name__ == '__main__':
    # p1 = 'D:\\testALg\\homework\\house\\227\\weld\\XQⅡ-BK041+M127-FP_14.jpg'
    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\XQⅢ-BJ018+M004-FP_06.jpg'

    # p2 = 'D:\\getWeld\\CJSTS\\XQⅢ-AD030+M080\\XQⅢ-AD030+M080_01.jpg'
    # p1 = 'D:\\getWeld\\CJSTS\\XQⅢ-AD030+M080-FP\\XQⅢ-AD030+M080-FP_02.jpg'

    # p1 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_498-F-125-12-0000.jpg'
    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_500-F-126-12-0000.jpg'

    # p1 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_1149-B-55-0-0000.jpg'
    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_1150-B-56-14-0000.jpg'

    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\o1.jpg'
    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\o2.jpg'


    # p1 = 'D:\\testALg\\homework\\house\\227\\weld\\XQⅡ-BK043+1+M005-FP_06.jpg'
    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\XQⅢ-BJ018+M004-FP_06.jpg'

    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_669-H-112-22-0000.jpg'
    # p1 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_672-H-113-22-0000.jpg'

    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_1149-B-55-0-0000.jpg'
    # p1 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_1150-B-56-14-0000.jpg'

    # p1 = 'D:\\testALg\\homework\\house\\227\\weld\\w1.jpg'
    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\w2.jpg'

    # p1 = 'D:\\getWeld\\results\\uploadTime-15735481088070000.jpg'
    # p2 = 'D:\\getWeld\\results\\uploadTime-15735477899070003.jpg'

    # p1 = 'D:\\getWeld\\results\\pipelineCode-150-LD22002-B2A-N131-weldingCode-G33_0003.jpg'
    # p2 = 'D:\\getWeld\\results\\pipelineCode-150-LD22002-B2A-N131-weldingCode-G33A_0004.jpg'

    # p1 = "D:\\getPianwei\\origin\\pipelineCode-100-P21511-B1D-P131-weldingCode-5_0003.jpg"
    # p2 = "D:\\getPianwei\\origin\\pipelineCode-150-LD22002-B2A-N131-weldingCode-G7_0006.jpg"

    # p1 = "D:\\getWeld\\results2\\pipelineCode-100-P21511-B1D-P131-weldingCode-5_0003.jpg"
    # p2 = "D:\\getWeld\\results2\\pipelineCode-150-LD22002-B2A-N131-weldingCode-G7_0006.jpg"

    # p1 = 'D:\\getWeld\\CJSSSSSSSSS\\XQⅢ-AF001+M002-W-FP\\XQⅢ-AF001+M002-W-FP_01.jpg'
    # p2 = 'D:\\getWeld\\CJSSSSSSSSS\\XQⅢ-AF001+M002-W-FP\\XQⅢ-AF001+M002-W-FP_05.jpg'

    # p2 = 'D:\getWeld\CJSSSSSSSSS\YH-03-QB010+001-Y\\YH-03-QB010+001-Y_01.jpg'
    # p1 = 'D:\getWeld\CJSSSSSSSSS\YH-03-QB010+001-Y\\YH-03-QB010+001-Y01R.jpg'

    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\XQⅡ-BK043+1+M005-FP_06.jpg'
    # p1 = 'D:\\testALg\\homework\\house\\227\\weld\\XQⅢ-BJ018+M004-FP_06.jpg'

    # p2 = 'D:\\getWeld\\results\\1146-B-53-16-0000.jpg'
    # p1 = 'D:\\getWeld\\results\\1068-B-4-0-0000.jpg'
    # p1 = 'D:\\getWeld\\results\\1068-B-4-16-0000.jpg'

    # p1 = 'D:\\testALg\homework\house\\227\\weld2\\7.jpg'
    # p2 = 'D:\\testALg\homework\house\\227\\weld2\\8.jpg'

    # p1 = 'D:\getWeld\\test\\1129-B-42-16-0000.jpg'
    # p2 = 'D:\getWeld\\test\\1122-B-38-22-0000.jpg'

    # p2 = 'D:\\getWeldPics\\test5\\XQⅡ-BK007-3+M006_01.jpg'


    # p2 = 'C:\\Users\\admin\\Desktop\\result\\焊缝提取\\7\\XQⅡ-GH000+M001+2-01.jpg'
    # p1 = 'C:\\Users\\admin\\Desktop\\result\\焊缝提取\\7\\XQⅡ-GH000+M002-01.jpg'

    # p2 = 'C:\\Users\\admin\\Desktop\\result\\焊缝提取\\8\\XQⅡ-GH005+M066-RW-XF-01.jpg'
    # p1 = 'C:\\Users\\admin\\Desktop\\result\\焊缝提取\\8\\XQⅡ-GH005-M041-01.jpg'

    # p2 = 'C:\\Users\\admin\\Desktop\\result\\原片\\8\\XQⅡ-GH008+M049-W-01.jpg'
    # p1 = 'C:\\Users\\admin\\Desktop\\result\\原片\\8\\XQⅡ-GH008+M049-W-02.jpg'

    # p2 = 'C:\\Users\\admin\\Desktop\\result\\原片\\9\\XQⅡ-GH004-M146-T001-02.jpg'
    # p1 = 'C:\\Users\\admin\\Desktop\\result\\原片\\9\\XQⅡ-GH004-M146-T001-06.jpg'

    # p2 = 'C:\\Users\\admin\\Desktop\\result\\焊缝提取\\1\\XQⅡ-GH000+M001+2-01.jpg'
    # p1 = 'C:\\Users\\admin\\Desktop\\result\\焊缝提取\\1\\XQⅡ-GH000+M002-01.jpg'




    # p2 = 'C:\\Users\\admin\\Desktop\\result\\焊缝提取\\4\\XQⅡ-GH005+M066-RW-XF-02.jpg'
    # p1 = 'C:\\Users\\admin\\Desktop\\result\\焊缝提取\\4\\XQⅡ-GH005-M041-02.jpg'

    # p2 = 'D:\\ten\\projectWelds\\XQⅡ-CF082-M005-W-01.jpg'
    # p1 = 'D:\\ten\\projectWelds\\XQⅡ-CF082-M006-W-01.jpg'

    # p2 = 'D:\\projectimg\\XQⅡ-CF082-M005-W-01.jpg'
    # p1 = 'D:\\projectimg\\XQⅡ-CF082-M006-W-01.jpg'

    p2 = 'D:\\ten\\projectWelds\\XQⅡ-GH000+M006-02.jpg'
    # p1 = 'D:\\ten\\mask\\getWeld\\weld\\XQⅡ-GI009+M026-02.jpg'


    # p2 = 'D:\\ten\\mask\\getWeld\\weld\\XQⅡ-GI009+M001-W-02.jpg'
    p1 = 'D:\\ten\\projectWelds\\XQⅡ-GH008+M217-02.jpg'


    # p2 = 'C:\\Users\\admin\\Desktop\\result\\焊缝提取\\5\\XQⅡ-GH008+M198-01.jpg'
    # p1 = 'C:\\Users\\admin\\Desktop\\result\\焊缝提取\\5\\XQⅡ-GH009+M045-01.jpg'



    # p2 = "D:\\getWeld\\pipeweld\\pipelineCode-150-LD22002-B2A-N131-weldingCode-G11_0003.jpg"
    # p1 = "D:\\getWeld\\pipeweld\\pipelineCode-150-LD22002-B2A-N131-weldingCode-G12_0005.jpg"


    import cv2,glob
    import matplotlib.pyplot as plt

 




    i1 = imread(p1)

    # i1 = his(i1)
    
    # print(i1.shape)
    i2 = imread(p2)

    # i2 = his(i2)



    # # i1 = cv2.resize(i1,(i1.shape[1]*2,i1.shape[0]*2))
    
    # i1 = cv2.cvtColor(i1,cv2.COLOR_BGR2GRAY)

    # # i1 = unevenLightCompensate(i1)

    
    # # i1 = np.power(i1 / 255.0, 2.2)
    # # i1 =  cv2.equalizeHist(i1)
    # i2 = cv2.cvtColor(i2,cv2.COLOR_BGR2GRAY)
    # i2 = np.power(i2 / 255.0, 2.2)
    # i2 =  cv2.equalizeHist(i2)

    # i2 = unevenLightCompensate(i2)
    # cv2.imwrite('D:\\testALg\\homework\\house\\227\\623a.jpg',i1)
    # cv2.imwrite('D:\\testALg\\homework\\house\\227\\623b.jpg',i2)

    



    i1 = tr(i1)
    i2 = tr(i2)



    # fea1 = np.sum(i1,0).astype(np.float32)
    # fea2 = np.sum(i2,0).astype(np.float32)


    


    # fea1 = process2(i1) 
    # fea2 = process2(i2) 

    fea1 = process4(i1) 
    fea2 = process4(i2) 



    fea1 = np_move_avg(fea1,5).astype(np.float32)
    fea2 = np_move_avg(fea2,5).astype(np.float32)


    ss = _compare2(fea1,fea2)
    # ss = _compare(fea1,fea2)
    print('============================')
    print(ss)

    #######################


    # print("ssssssss======>>>>{}".format(_compare_newMethod(fea1,fea2)))









    #############################

    

    
    # im = astronaut()
    # im = imread('D:\\testALg\\homework\\house\\227\\1-2weld.jpg')
    # im = imread('D:\\testALg\\homework\\house\\227\\1122.jpg')

    # i2 = copy.deepcopy(lcc)
    # i1,w1,fea1 = process(p1)
    # # print(w1)
    # i2,w2,fea2 = process(p2)

    
    # # lcc = lcc*255

    # # r1 = getShape(i1)
    # # r2 = getShape(i2)

    # r1 = nomarlLize(getShape(i1))
    # r2 = nomarlLize(getShape(i2))

    # # r1 = getShape(i1)
    # # r2 = getShape(i2)


    # # import fastDtw


    # r11 = r1[int(0.25*len(r1)):int(0.75*len(r1))]
    # r22 = r2[int(0.25*len(r2)):int(0.75*len(r2))]



    # res = cv2.matchTemplate(r1, r22, cv2.TM_CCOEFF_NORMED)
    # min_val1, max_val, _, _ = cv2.minMaxLoc(res)

    # res2 = cv2.matchTemplate(r2, r11, cv2.TM_CCOEFF_NORMED)
    # min_val2, max_val2, _, _ = cv2.minMaxLoc(res2)
    # print(0.5*(max_val+max_val2))



    
    # r11 = fea1[int(0.1*len(fea1)):int(0.5*len(fea1))]
    # r22 = fea2[int(0.1*len(fea2)):int(0.5*len(fea2))]

    # # print(np.mean(r11))
    # # print(np.mean(r22))
    # # print(0.5*len(r11))

    # # r11 = np.flipud(r11)   # test  2020.5.18

    # res = cv2.matchTemplate(fea1, r22, cv2.TM_CCOEFF_NORMED)
    # min_val1, max_val, _, _ = cv2.minMaxLoc(res)

    # res2 = cv2.matchTemplate(fea2, r11, cv2.TM_CCOEFF_NORMED)
    # min_val2, max_val2, _, _ = cv2.minMaxLoc(res2)


    # # ma = lambda r1,r2:np.abs(r1-r2)
    # # d, _,_,_ = accelerated_dtw(r11, r22, dist='euclidean')

    # # print(a2-a1)
    # print(max (max_val,max_val2))

    # print("焊缝A的形状为{}，焊缝B的形状为{}，"+"\n"+"初筛相似度{}，二次筛选相似度{}".format(str(i1.shape),str(i2.shape), \
    #     str(round(0.5*(max_val+max_val2),4)),str(0.9791)))

    # print(min(min_val1,min_val2))
    # print(min_val)

    # print(d)

    # imsave('out1.png', i1*255)
    # imsave('out2.png', i2*255)
    
    # f = open('D:\\testALg\\homework\\house\\227\\result.txt','a')





    y1 = fea1
    y2 = fea2

    print(len(y1))
    x1 = np.linspace(1, len(y1), len(y1))
    x2 = np.linspace(1, len(y2), len(y2))
    plt.plot(x1, y1, ls="-", lw=2, label="Weld A")
    plt.plot(x2, y2, ls="-", lw=2, label="Weld B")

    plt.legend()

    plt.show()

  

    # imgs = glob.glob("D:\\ten\\mask\\getWeld\\weld\\*.jpg")
    # f = open('D:\\testALg\\homework\\house\\227\\result.txt','a')
    # p1 = 'D:\\ten\\mask\\getWeld\\weld\\XQⅡ-GI009+M001-W-02.jpg'

    # for i in imgs:
    #     p2 = i
    #     i1 = imread(p1)

    #     i2 = imread(p2)

    #     i1 = tr(i1)
    #     i2 = tr(i2)
    #     fea1 = processUnet(i1) 
    #     fea2 = processUnet(i2) 
    #     fea1 = np_move_avg(fea1,5).astype(np.float32)
    #     fea2 = np_move_avg(fea2,5).astype(np.float32)

    #     ss = _compare(fea1,fea2)

    #     f.write(i+">>>>>>>>"+str(ss)+'\n')
    
    # f.close()






    










 




