import cv2
import numpy as np

img = cv2.imread("D:\\homework\\homework\\house\\weld333.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th = np.ones(gray.shape,dtype= np.float32)

for i in range(251,601,30):
    kernel = np.ones((i,3),dtype = np.float32)
    kernel = kernel/np.sum(kernel)

    dst = cv2.filter2D(gray,-1,kernel)
    _,th1 = cv2.threshold(dst,0,1,cv2.THRESH_OTSU)

    th = th*th1 





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

cv2.imwrite('D:\\homework\\homework\\house\\ori.jpg', f)

th[th == 1] = 255

cv2.imwrite('D:\\homework\\homework\\house\\weldthres1.jpg', th)