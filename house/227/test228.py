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
    b, g, r = cv2.split(image)
    # 创建局部直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    # 对每一个通道进行局部直方图均衡化
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    # 合并处理后的三通道 成为处理后的图
    image = cv2.merge([b, g, r])

    return image


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
    im = imread(imgPath)
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


def process2(im):
    imgShape = im.shape 

    # im = cv2.imread(imgPath)
    im = his(im)
    if imgShape[0]>imgShape[1]:
        pass 
    else:
        trans_img = cv2.transpose(im)
        im = cv2.flip(trans_img, 1)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    imgIn = im / 255.0
    imgIn = xdog(imgIn, binarize=True,k=20)


    imgIn = me(im,mask=imgIn)

    cv2.imwrite("out6666.png", 255*imgIn)
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






if __name__ == '__main__':
    # p1 = 'D:\\testALg\\homework\\house\\227\\weld\\XQⅡ-BK041+M127-FP_14.jpg'
    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\XQⅢ-BJ018+M004-FP_06.jpg'

    # p2 = 'D:\\getWeld\\CJSTS\\XQⅢ-AD030+M080\\XQⅢ-AD030+M080_01.jpg'
    # p1 = 'D:\\getWeld\\CJSTS\\XQⅢ-AD030+M080-FP\\XQⅢ-AD030+M080-FP_02.jpg'

    p1 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_498-F-125-12-0000.jpg'
    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\extract_500-F-126-12-0000.jpg'

    p2 = 'D:\\testALg\\homework\\house\\227\\weld\\o1.jpg'
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

    # p1 = "D:\\getWeld\\results\\457-G-160-8-0000.jpg"
    # p2 = "D:\\getWeld\\results\\516-F-133-10-0000.jpg"

    # p2 = 'D:\\testALg\\homework\\house\\227\\weld\\XQⅡ-BK043+1+M005-FP_06.jpg'
    # p1 = 'D:\\testALg\\homework\\house\\227\\weld\\XQⅢ-BJ018+M004-FP_06.jpg'

    # p2 = 'D:\\getWeld\\results\\1146-B-53-16-0000.jpg'
    # p1 = 'D:\\getWeld\\results\\1068-B-4-0-0000.jpg'
    # p1 = 'D:\\getWeld\\results\\1068-B-4-16-0000.jpg'

    # p2 = "D:\\getWeld\\pipeweld\\pipelineCode-150-LD22002-B2A-N131-weldingCode-G11_0003.jpg"
    # p1 = "D:\\getWeld\\pipeweld\\pipelineCode-150-LD22002-B2A-N131-weldingCode-G12_0005.jpg"


    import cv2
    import matplotlib.pyplot as plt

    i1 = imread(p1)
    # i1 = cv2.resize(i1,(i1.shape[1]*2,i1.shape[0]*2))
    # print(i1.shape)
    i2 = imread(p2)
    # print(i2.shape)

    # i1 = i1 + 30
    # i2 = i2 - 30

    fea1 = process2(i1) 
    fea2 = process2(i2) 

    #######################
    # params = np.load("D:\\getWeld\\features_LONG.npy",allow_pickle=True)
    # s = list(filter(lambda x:x.pipeline == "XQⅡ-BK043+1+M005-FP" ,params))

    # f = s[0].features









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



    
    r11 = fea1[int(0.25*len(fea1)):int(0.75*len(fea1))]
    r22 = fea2[int(0.25*len(fea2)):int(0.75*len(fea2))]

    print(np.mean(r11))
    print(0.5*len(r11))

    # r11 = np.flipud(r11)   # test  2020.5.18

    res = cv2.matchTemplate(fea1, r22, cv2.TM_CCOEFF_NORMED)
    min_val1, max_val, _, _ = cv2.minMaxLoc(res)

    res2 = cv2.matchTemplate(fea2, r11, cv2.TM_CCOEFF_NORMED)
    min_val2, max_val2, _, _ = cv2.minMaxLoc(res2)


    # ma = lambda r1,r2:np.abs(r1-r2)
    # d, _,_,_ = accelerated_dtw(r11, r22, dist='euclidean')

    # print(a2-a1)
    print(0.5*(max_val+max_val2))

    # print("焊缝A的形状为{}，焊缝B的形状为{}，"+"\n"+"初筛相似度{}，二次筛选相似度{}".format(str(i1.shape),str(i2.shape), \
    #     str(round(0.5*(max_val+max_val2),4)),str(0.9791)))

    # print(min(min_val1,min_val2))
    # print(min_val)

    # print(d)

    # imsave('out1.png', i1*255)
    # imsave('out2.png', i2*255)
    
    y1 = fea1
    y2 = fea2

    print(len(y1))
    x1 = np.linspace(1, len(y1), len(y1))
    x2 = np.linspace(1, len(y2), len(y2))
    plt.plot(x1, y1, ls="-", lw=2, label="Weld A")
    plt.plot(x2, y2, ls="-", lw=2, label="Weld B")

    plt.legend()

    plt.show()

    










 




