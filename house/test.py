import cv2
import numpy as np
import copy

img = cv2.imread("D:\\house\\extract_test.jpg")
grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

minV = np.min(grayImg)
maxV = np.max(grayImg)

x,y = grayImg.shape

print((x,y))

grayImg = np.array(grayImg,dtype=float)

grayImg = np.log2((grayImg-minV)/maxV + 1)

print(grayImg)

c1 = copy.deepcopy(grayImg)
c1[c1<=0.5] = 1                        #大于0.5
c2 = copy.deepcopy(grayImg)
c2[c2>0.5] = 0

n1 = 1-2*(1-c1)**2
n2 = 2*c2**2

# print("&&&&&&&&&&&&&&&&&&&&&&")
# print(n1)
# print(n2)
# print("######################")

n1[n1 == 1] = 0

newMatrix = n1 + n2

# newMatrix = [2*grayImg[i][j]**2 if grayImg[i][j]<0.5 else 1-2*(1-grayImg[i][j])**2 for j in range(0,y) for i in range(0,x)]

# newMatrix = 2*grayImg**2 if grayImg.any()<0.5 else 1-2*(1-grayImg)**2 

print(newMatrix.shape)

newMatrix = np.floor(newMatrix*255)

print(newMatrix)

cv2.imwrite("D:\\house\\test.jpg",newMatrix)


img2 = np.power(grayImg/255.0,2.2)
newMatrix2 = np.floor(img2*255)

cv2.imwrite("D:\\house\\gamma.jpg",newMatrix)

