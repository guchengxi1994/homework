'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-02-27 17:01:01
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-24 17:03:28
'''
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler


BASE_DIR = os.path.abspath(os.curdir)
model_ROOT = os.path.join(BASE_DIR, 'static')
image_ROOT = os.path.join(BASE_DIR,"images")
# print(image_ROOT)


def FT(src):
    lab = cv2.cvtColor(src,cv2.COLOR_BGR2LAB)
    gaussian_blur=cv2.GaussianBlur(src,(5,5),0)

    mean_lab = np.mean(lab,axis=(0,1))
    print(mean_lab.shape)

    salient_map = (gaussian_blur - mean_lab)*(gaussian_blur - mean_lab)
    salient_map = (salient_map-np.amin(salient_map))/(np.amax(salient_map)-np.amin(salient_map))
    # salient_map = np.array(salient_map,dtype=np.uint8)
    # print(np.min(salient_map))
    # salient_map = (1-salient_map) * 255
    salient_map = salient_map*255
    salient_map = np.array(salient_map,dtype=np.uint8)
    salient_map = cv2.cvtColor(salient_map,cv2.COLOR_BGR2GRAY)
    salient_map = np.array(salient_map,dtype=np.float32)

    salient_map = salient_map/np.max(salient_map)*255
    salient_map = np.array(salient_map,dtype=np.uint8)

    salient_map = 255 - salient_map


    
    # xs = np.reshape(salient_map,(-1,1))
    # xs = np.reshape(xs,(1,len(xs)))

    # plt.hist(salient_map.ravel(), 256, [0, 256])
    # plt.show()



    # salient_map[salient_map ==0] = 255
    # salient_map[salient_map!=255] = 0

    
    # print(salient_map.shape)
    # salient_map = cv2.cvtColor(salient_map,cv2.COLOR_BGR2GRAY)
    # ret1, th1 = cv2.threshold(salient_map, 0, 255, cv2.THRESH_OTSU) 

    # salient_map[salient_map<200] = 255
    # salient_map[salient_map!=255] = 0

    return salient_map



if __name__ == "__main__":
    import copy
    
    # i = cv2.imread(image_ROOT+os.sep+"4-5.jpg")
    i = cv2.imread(BASE_DIR+os.sep+"1-2weld.jpg")
    

    # i = i.astype(np.uint8)

    i2 = FT(i)
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

    print(times)

    # i3 = copy.deepcopy(i2)

    print(np.max(i2))

    i3 = np.array(i3,dtype=np.float32)

    # baseLine = i2[:,times] 
    p1 = i3[:,0:times]/255
    p2 = i3[:,times+1:] /255

    l1 = np.sum(p1,1)
    l2 = np.sum(p2,1)
    print(len(l1-l2))

    print(l1 - l2)






    cv2.imwrite(BASE_DIR+os.sep+"1122.jpg",i2)