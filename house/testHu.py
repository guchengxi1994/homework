import cv2
import numpy as np
from math import copysign,log10

def getHus(img):
    moment = cv2.moments(img)
    huMoments = cv2.HuMoments(moment)
    # print(huMoments)
    huS = []
    for i in huMoments:
        mo = -1*copysign(1.0,i)*log10(abs(i))
        huS.append(mo)
    
    return huS


img = cv2.imread("D:\\homework\\homework\\house\\219_1_weldthres1.jpg")

img2 = cv2.imread("D:\\homework\\homework\\house\\219_2_weldthres1.jpg")
# print(img2.shape)

huS1 = getHus(img[:,:,0])
huS2 = getHus(img2[:,:,0])

print (huS1)
print (huS2)

m1 = cv2.matchShapes(img[:,:,0],img2[:,:,0],cv2.CONTOURS_MATCH_I2,0)
print(m1)


from fuDe import findDescriptor

x1 = findDescriptor(img[:,:,0])
x2 = findDescriptor(img2[:,:,0])

print(x1)
print(x2)


