import cv2 
import numpy as np 

x = np.zeros((512,512,3),dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

imgzi = cv2.putText(x, 'U R a pig', (50, 300), font, 1.2, (255, 255, 255), 2)


cv2.imwrite("D:\\testALg\\homework\\imgInpaint\\tem.jpg",imgzi)

# print(np.max(imgzi))

ori = cv2.imread('D:\\testALg\\homework\\imgInpaint\\lenna.png')

ori = np.array(ori,dtype=int)

ori = ori + imgzi

ori[ori>255] = 255

cv2.imwrite("D:\\testALg\\homework\\imgInpaint\\ori.jpg",ori)
