import cv2
import numpy as np
import mid
from skimage.measure import label


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


def getBounding(img):
    # img[img>=1] = 255
    img = np.array(img,dtype= np.uint8)
    # print(img.shape)
    # print(type(img))
    # _, img_bin = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    # img_bin = img[:,:,0]
    contours, _,_ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    st_x, st_y, width, height = cv2.boundingRect(contours[0])
    return st_x,st_y,width,height


img = cv2.imread("D:\\homework\\homework\\house\\test111.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th = np.ones(gray.shape,dtype= np.float32)

for i in range(251,601,30):
    kernel = np.ones((i,3),dtype = np.float32)
    kernel = kernel/np.sum(kernel)

    dst = cv2.filter2D(gray,-1,kernel)
    _,th1 = cv2.threshold(dst,0,1,cv2.THRESH_OTSU)

    th = th * th1
    # th [th>=1] = 1 


###########test

st_x,st_y,width,height = getBounding(th)

tem = np.zeros(gray.shape,dtype= np.float32)

tem[st_x:st_x+height,st_y:st_y+width] = 1



###########



m = mid.getMid(gray)

th = th * m
# th = tem * m

# th = largestConnectComponent(th)

th [th!=0] = 1



# kernel1 = np.ones((77,3),dtype = np.float32)
# kernel2 = np.ones((99,3),dtype = np.float32)
# kernel3 = np.ones((111,3),dtype = np.float32)

# kernel1 = kernel1/np.sum(kernel1)
# kernel2 = kernel2/np.sum(kernel2)
# kernel3 = kernel3/np.sum(kernel3)

# dst1 = cv2.filter2D(gray,-1,kernel1)
# dst2 = cv2.filter2D(gray,-1,kernel2)
# dst3 = cv2.filter2D(gray,-1,kernel3)

# _,th1 = cv2.threshold(dst1,0,1,cv2.THRESH_OTSU)
# _,th2 = cv2.threshold(dst2,0,1,cv2.THRESH_OTSU)
# _,th3 = cv2.threshold(dst3,0,1,cv2.THRESH_OTSU)

# th = th1*th2*th3 
# th = th1

f = th*gray

cv2.imwrite('D:\\homework\\homework\\house\\221_22_ori.jpg', f)

th = np.array(th,dtype = np.uint8)

th[th >= 1] = 255



cv2.imwrite('D:\\homework\\homework\\house\\221_22_weldthres1.jpg', th)

m[m>0] = 255
cv2.imwrite('D:\\homework\\homework\\house\\221_tem.jpg', m)