# import cv2
# import numpy as np

# img = cv2.imread("D:\\testALg\\homework\\house\\227\\111.jpg")
# lsc = cv2.ximgproc.createSuperpixelLSC(img)
# lsc.iterate(5)
# mask_lsc = lsc.getLabelContourMask()
# label_lsc = lsc.getLabels()

# # print(label_lsc)
# number_lsc = lsc.getNumberOfSuperpixels()
# print(mask_lsc)
# mask_inv_lsc = cv2.bitwise_not(mask_lsc)
# img_lsc = cv2.bitwise_and(img,img,mask = mask_inv_lsc)
# cv2.imshow("img_lsc",img_lsc)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# from skimage.segmentation import slic,mark_boundaries
# from skimage import io
# import matplotlib.pyplot as plt
# # import numpy as np
# #
# # np.set_printoptions(threshold=np.inf)

# img = io.imread("D:\\testALg\\homework\\house\\227\\111.jpg")


# # segments = slic(img, n_segments=60, compactness=10)
# # out=mark_boundaries(img,segments)
# # # print(segments)
# # plt.subplot(121)
# # plt.title("n_segments=60")
# # plt.imshow(out)

# segments2 = slic(img, n_segments=12, compactness=100)
# out2=mark_boundaries(img,segments2)
# # plt.subplot(122)
# plt.title("n_segments=300")
# plt.imshow(out2)

# plt.show()

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt


# img = cv2.imread('D:\\testALg\\homework\\house\\227\\111_wps.jpg')
# imgshape = img.shape
# mask = np.zeros(img.shape[:2],np.uint8)
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# rect = (50,50,imgshape[0]-50,imgshape[1]-50)#划定区域
# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)#函数返回值为mask,bgdModel,fgdModel
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')#0和2做背景

# img = img*mask2[:,:,np.newaxis]#使用蒙板来获取前景区域



# cv2.imshow('p',img)
# cv2.waitKey(0)


import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label
import cv2
# from skimage.data import astronaut
from skimage.io import imsave,imread
import copy

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


def process(imgPath):
    im = imread(imgPath)

    im = im / 255.0
    im = xdog(im, binarize=True, k=20)

    img = np.array(im,dtype=np.uint8)
    # img[:,int(0.5*img.shape[1])] = 1
    lcc = largestConnectComponent(img)
    lcc = np.array(lcc,dtype=np.uint8)

    return lcc




if __name__ == '__main__':
    p1 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_1149-B-55-0-0000.jpg'
    p2 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_1150-B-56-14-0000.jpg'


    import cv2
    import matplotlib.pyplot as plt

    

    
    # im = astronaut()
    # im = imread('D:\\testALg\\homework\\house\\227\\1-2weld.jpg')
    # im = imread('D:\\testALg\\homework\\house\\227\\1122.jpg')

    # i2 = copy.deepcopy(lcc)
    i1,i2 = process(p1),process(p2)

    
    # lcc = lcc*255

    r1 = getShape(i1)
    r2 = getShape(i2)

    from dtw import dtw,accelerated_dtw
    # import fastDtw
    import time
    a1 = time.time()

    r11 = r1[int(0.25*len(r1)):int(0.75*len(r1))]
    r22 = r2[int(0.25*len(r2)):int(0.75*len(r2))]



    res = cv2.matchTemplate(r1, r22, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, _, _ = cv2.minMaxLoc(res)


    # ma = lambda r1,r2:np.abs(r1-r2)
    # d, _,_,_ = accelerated_dtw(r11, r22, dist='euclidean')
    a2 = time.time()
    # print(a2-a1)
    print(max_val)
    # print(min_val)

    # print(d)
    
    # y = r1
    # x = np.linspace(1, len(y), len(y))
    # plt.plot(x, y, ls="-", lw=2, label="plot figure")

    # plt.legend()

    # plt.show()





 


    imsave('out1.png', i1*255)
    imsave('out2.png', i2*255)

