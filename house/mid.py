import cv2
import numpy as np

def getMid(img):
    imgshape = img.shape 
    if len(imgshape) == 3:
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    x,y = gray.shape
    line = np.mean(gray,1)
    line = np.reshape(line,(x,1))
    conImg = np.tile(line,(1,y))
    res = gray - conImg

    res[res<0] = 0
    res[res!=0] = 1 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(13,13))
    opened1 = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel,iterations=1)

    # ori = opened1*gray
    # data = img.ravel()[np.flatnonzero(ori)]

    # thres = np.percentile(data, [ 90])

    # ori[ori>thres] = 0
    # ori[ori!=0] = 1



    return opened1



# i = cv2.imread("D:\\homework\\homework\\house\\219_2.jpg")
# # i = i[:,:,0]
# gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
# print(gray.shape)

# x,y = gray.shape

# line = np.mean(gray,1)
# line = np.reshape(line,(x,1))
# # print(x)
# # print(y)

# conImg = np.tile(line,(1,y))


# ##########test
# line2 = np.mean(gray,0)
# conImg2 = np.tile(line2,(x,1))




# #############test



# print(conImg.shape)
# print(conImg2.shape)

# res = gray - conImg

# res2 = gray - conImg2

# res[res<0] = 0
# res[res!=0] = 1 

# res2[res2<0] = 0
# res2[res2!=0] = 1 
# # res2 = np.array(res2,dtype= np.uint8)

# # print(np.max(res2))


# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(13,13))
# opened1 = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel,iterations=1)


# resG = opened1*gray 

# resG2 = res2*gray 

# cv2.imwrite("D:\\homework\\homework\\house\\111111.jpg",resG)

# cv2.imwrite("D:\\homework\\homework\\house\\222222.jpg",resG2)