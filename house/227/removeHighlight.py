'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-09 13:59:55
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-09 15:27:06
'''
import cv2
import numpy as np
# from testTuoyuan import getMaxRegion


def getMaxRegion_pipe(img,thres):
    # kernel = np.ones((5,5),np.uint8)
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    x1,y1 = stats.shape
    x,y = img.shape
    lab = []
    if x>y:
        # maxArea = 0     
        for i in range(1,x1):
            # print(stats[i][4])
            if stats[i][4]>thres \
                and (centroids[i][1]<0.2*y or centroids[i][1]>0.8*y):
               
                lab.append(i) 

        labels[labels!=lab] = 0
        labels[labels!=0] = 1
        return labels

    else:
        # maxArea = 0
        # lab = 0
        for i in range(1,x1):
            # print(stats[i][4])
            if stats[i][4]>thres \
                and (centroids[i][0]<0.2*x or centroids[i][0]>0.8*x):
              
                lab.append(i)   
    # print(lab)
    # labels[labels  not in  lab] = 0
    for i in lab:
        labels[labels==i] = 255
    labels[labels!=255] = 0

    labels = np.array(labels,dtype=np.float32) /255
    
    return 1 - labels



def remove(img_or_path):
    if isinstance(img_or_path,str):
        img = cv2.imread(img_or_path)
    else:
        img = img_or_path

    

    imgShape = img.shape 
    if len(imgShape) >2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    thres = 0.01*imgShape[0]*imgShape[1]

    # print(thres)
    
    maxVal = np.max(img)

    img[img>0.8*maxVal] = 255
    img[img!=255] = 0

    # cv2.imwrite("jjjj.jpg",img)

    ll = getMaxRegion_pipe(img,thres)

    ll = np.array(ll,dtype=np.uint8)
    # print(np.max(ll))
    rect = cv2.boundingRect(ll)
    print(rect)
    # cv2.imwrite("ffff.jpg",ll*255)
    # return rect[0],rect[2]
    return ll
    

    




if __name__ == "__main__":
    
    i = "D:\\getWeld\\pipeweld\\pipelineCode-51X51-3-weldingCode-1_0002.jpg"
    remove(i)