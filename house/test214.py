import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import copy

img = cv2.imread("D:\\homework\\homework\\house\\219_22_ori.jpg")

img1 = copy.deepcopy(img)

data = img.ravel()[np.flatnonzero(img)]
# data = hist
data = data.reshape(-1,1)

print(data.shape)

kmeans = KMeans(3)
kmeans.fit(data)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# print(labels)
print(labels.shape)
labels = np.reshape(labels,data.shape)

print(centroids)

clus = np.append(data,labels,axis=1)

print(clus.shape)

clus1 = copy.deepcopy(clus)

x = clus[clus[:,1] == 0]
y = clus[clus[:,1] == 1]
z = clus[clus[:,1] == 2]
print(x.shape)

m1 = max(x[:,0])
m2 = max(y[:,0])
m3 = max(z[:,0])

print(m1)
print(m2)
print(m3)


print(min(m1,m2,m3))

thres = min(m1,m2,m3)

# clus[clus]

# m1 = np.max()

# hist, bins = np.histogram(img.ravel(), 256, [0, 256])

# img[img<36] = 0
# # img[img>=63 && img<=220] = 127
# img[img>114] = 255

img[img>thres] = 0
img[img!=0] = 1

img2 = img1*img

cv2.imwrite('D:\\homework\\homework\\house\\ori214.jpg', img2)





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

    return lcc


aa = largestConnectComponent(img)

# 




# aa = np.array(aa).astype(np.uint8)
aa = np.array(aa,dtype= np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(71, 71))
iClose = cv2.morphologyEx(aa, cv2.MORPH_CLOSE, kernel)
iClose = cv2.morphologyEx(iClose, cv2.MORPH_CLOSE, kernel)

img3 = img1*iClose
cv2.imwrite('D:\\homework\\homework\\house\\close.jpg', img3)





# thresh, binary = cv2.threshold(aa, 1, 255, cv2.THRESH_BINARY_INV)

# img_contour, contours, hierarchy = cv2.findContours(aa, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# rect = cv2.minAreaRect(contours[0])
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(img1, [box], 0, (0, 0, 255), 2)




aa[aa>0] = 255
aa = np.uint8(aa)
# print(aa)
cv2.imwrite('D:\\homework\\homework\\house\\aa219_22.jpg', aa)

cv2.imwrite('D:\\homework\\homework\\house\\img1.jpg',img1)






# plt.hist(img1.ravel(), 256, [1, 240])
# plt.show()