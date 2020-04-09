import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label
import cv2
# from skimage.data import astronaut
from skimage.io import imsave,imread
import copy
import math
import removeHighlight


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

    # cv2.imwrite("out22222.png",im_floodfill_inv)
    # Combine the two images to get the foreground.
    im_out = img + im_floodfill_inv

    # cv2.imwrite("out33333.png",im_out)


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
    # print(stats.shape)

    # for i in (1,x+1):
    #     stats1 = stats[1:,:]
    #     print(stats1)
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
    
    # imgShape = imdiff.shape
    # if imgShape[0]>imgShape[1]:
    #     imdiff[int(0.25*imgShape[0]):int(0.75*imgShape[0]),int(0.25*imgShape[1])] = 1
    #     imdiff[int(0.25*imgShape[0]):int(0.75*imgShape[0]),int(0.75*imgShape[1])] = 1
    # else:
    #     imdiff[int(0.25*imgShape[0]),int(0.25*imgShape[1]):int(0.75*imgShape[1])] = 1
    #     imdiff[int(0.75*imgShape[0]),int(0.25*imgShape[1]):int(0.75*imgShape[1])] = 1
    
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
    # print(wL.shape)
    # img[:,int(0.5*img.shape[1])] = 1
    # lcc = largestConnectComponent(img)
    # lcc = np.array(lcc,dtype=np.uint8)

    return lcc,1,fea2
    # return lcc,w,nomarlLize(fea2)


def process2(im):
    imgShape = im.shape 
    lcc2 = removeHighlight.remove(im)
      
    if imgShape[0]<imgShape[1]:
        pass 
    else:
        trans_img ,trans_img2= cv2.transpose(im),cv2.transpose(lcc2)
        im,lcc2 = cv2.flip(trans_img, 1),cv2.flip(trans_img2, 1)
    
    rect = cv2.boundingRect(lcc2)


    
    im = his(im)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgIn = im / 255.0
    imgIn = xdog(imgIn, binarize=True,k=20)
    
    
    fe = imgIn * im 
    cv2.rectangle(im,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),2)
    cv2.imwrite("out33333.png", im)
    # fea2 = np.sum(fe,axis=1,dtype=np.float32)
    # print(len(fea2))
    return fe,rect[0],rect[2]


def process3(im):
    imgShape = im.shape 
    lcc2 = removeHighlight.remove(im)
      
    if imgShape[0]<imgShape[1]:
        pass 
    else:
        trans_img ,trans_img2= cv2.transpose(im),cv2.transpose(lcc2)
        im,lcc2 = cv2.flip(trans_img, 1),cv2.flip(trans_img2, 1)
    
    rect = cv2.boundingRect(lcc2)

    im = im[:,rect[0]:rect[0]+rect[2]]
    
    im = his(im)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgIn = im / 255.0
    imgIn = xdog(imgIn, binarize=True,k=20)
    
    
    fe = imgIn * im 
    # cv2.rectangle(im,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),2)
    cv2.imwrite("out33333.png", im*imgIn)
    # fea2 = np.sum(fe,axis=1,dtype=np.float32)
    # print(len(fea2))
    return fe, -1, -1


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


def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr



def t2c(*args):
    # print(len(args[0]))
    img = args[0][0]
    start = args[0][1]
    width = args[0][2]
    imgShape = img.shape
    if imgShape[0]>imgShape[1]:
        trans_img = cv2.transpose(img)
        img = cv2.flip(trans_img, 1)
        imgShape = img.shape

    if start!=-1:  
        img = img[:,start:start+width]
 
    imgShape = img.shape
    
    h = imgShape[0]*0.5
    # print(hh)
    r = 0.5*imgShape[1]
    # rr = 0.5*r
    # # print(r)
    # h =  r

    t = math.sqrt(r**2/(0.75*h**2+0.25*r**2))
    # print(t)
    
    upper = img[0:int(0.5*imgShape[0]),:]
    lower = img[int(0.5*imgShape[0]):,:]

    upper_sum = np.sum(upper,axis=0)
    lower_sum = np.sum(flip180(lower),axis=0)

    u1 = upper[:,0:int(0.25*(imgShape[1]))]
    # print(u1)
    u1 = np.array(u1,dtype=np.uint8)
    # print(u1.shape)
    u2 = upper_sum[int(0.25*(imgShape[1])):int(0.75*(imgShape[1]))]
    u3 = upper[:,int(0.75*(imgShape[1])):]
    u3 = np.array(u3,dtype=np.uint8)

    l1 = flip180(lower[:,0:int(0.25*(imgShape[1]))])
    # print(l1.shape)
    l1 = np.array(l1,dtype=np.uint8)
    l2 = lower_sum[int(0.25*(imgShape[1])):int(0.75*(imgShape[1]))]
    l3 = flip180(lower[:,int(0.75*(imgShape[1])):])
    l3 = np.array(l3,dtype=np.uint8)

    u1_lengthen = np.sum(cv2.resize(u1,(u1.shape[0],int(u1.shape[1]*t))),axis=0)
    u3_lengthen = np.sum(cv2.resize(u3,(u3.shape[0],int(u3.shape[0]*t))),axis=0)

    l1_lenthen = np.sum(cv2.resize(l1,(l1.shape[0],int(l1.shape[0]*t))),axis=0)
    l3_lenthen = np.sum(cv2.resize(l3,(l3.shape[0],int(l3.shape[0]*t))),axis=0)

    return np.concatenate((u1_lengthen,u2,u3_lengthen,l3_lenthen,l2,l1_lenthen))
    







if __name__ == '__main__':

    # p2 = "D:\\getWeld\\pipeweld\\pipelineCode-51X51-3-weldingCode-1_0006.jpg"
    # p1 = "D:\\getWeld\\pipeweld\\pipelineCode-51X51-3-weldingCode-1_0005.jpg"

    # p2 = "D:\\getWeld\\pipeweld\\pipelineCode-576-weldingCode-1_0001.jpg"
    # p1 = "D:\\getWeld\\pipeweld\\pipelineCode-576-weldingCode-1_0002.jpg"

    p1 = "D:\\getWeld\\pipeweld\\pipelineCode-51X51-3-weldingCode-1_0001.jpg"
    p2 = "D:\\getWeld\\pipeweld\\pipelineCode-51X51-3-weldingCode-1_0004.jpg"


    # import cv2
    import matplotlib.pyplot as plt

    i1 = imread(p1)
    i2 = imread(p2)

    mode = 2

    if mode == 2:

        fea1 = t2c(process2(i1))
        fea2 = t2c(process2(i2))

        # fea1 = t2c(process3(i1))
        # fea2 = t2c(process3(i2))

        fea1 = np.array(fea1,dtype=np.float32)
        # print(type(fea1))
        fea2 = np.array(fea2,dtype=np.float32)
        # print()

        # print(fea1.shape)
        # print(fea2.shape)



        r11 = fea1[int(0.25*len(fea1)):int(0.75*len(fea1))]
        # print(type(r11))
        r22 = fea2[int(0.25*len(fea2)):int(0.75*len(fea2))]
        # print(r22.shape)
        # print(r11.shape)
        # print(r22.shape)

        res = cv2.matchTemplate(fea1, r22, cv2.TM_CCOEFF_NORMED)
        # res = cv2.matchTemplate(fea1[int(0.15*len(fea1)):int(0.85*len(fea1))], r22, cv2.TM_CCOEFF_NORMED)
        min_val1, max_val, _, _ = cv2.minMaxLoc(res)

        # res2 = cv2.matchTemplate(fea2[int(0.15*len(fea2)):int(0.85*len(fea2))], r11, cv2.TM_CCOEFF_NORMED)
        res2 = cv2.matchTemplate(fea2, r11, cv2.TM_CCOEFF_NORMED)
        min_val2, max_val2, _, _ = cv2.minMaxLoc(res2)


        print(0.5*(max_val+max_val2))

    
        
        y1 = fea1
        y2 = fea2

   
        x1 = np.linspace(1, len(y1), len(y1))
        x2 = np.linspace(1, len(y2), len(y2))
        plt.plot(x1, y1, ls="-", lw=2, label="plot figure")
        plt.plot(x2, y2, ls="-", lw=2, label="plot figure")

        plt.legend()

        plt.show()