import numpy as np
import cv2
import copy

img = cv2.imread("D:\\homework\\homework\\house\\test111.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# dst = img[img>200]
dst = copy.deepcopy(img)

dst[dst<150] = 0 
dst[dst!=0] = 255

cv2.imwrite("D:\\homework\\homework\\house\\212121.jpg",dst)

dd = cv2.inpaint(img,dst,3,cv2.INPAINT_TELEA)

cv2.imwrite("D:\\homework\\homework\\house\\inpaint.jpg",dd)

