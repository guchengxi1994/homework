# -*- coding: utf-8 -*-
import cv2
import glob 
import os
import numpy as np 






BASE_DIR = os.path.abspath(os.curdir)
defect_model_ROOT = os.path.join(BASE_DIR, 'static'+os.sep+"def")
weld_model_ROOT = os.path.join(BASE_DIR, 'static'+os.sep+"MyWeld")

image_ROOT = os.path.join(BASE_DIR,"ori")
resultImgRoot = os.path.join(BASE_DIR,"result")
bResultRoot = os.path.join(BASE_DIR,'bResult')

#遍历目标文件夹
PATH= image_ROOT
# PATH = "./testORI"
paths = glob.glob(os.path.join(PATH, r'*.jpg'))


def cal_dist(hist):
    dist = {}
    for gray in range(256):
        value = 0.0
        for k in range(256):
            value += hist[k][0] * abs(gray - k)
        dist[gray] = value
    return dist



def LC(image_gray):
    image_height = image_gray.shape[0]
    image_width = image_gray.shape[1]
    image_gray_copy = np.zeros((image_height, image_width))
    hist_array = cv2.calcHist([image_gray], [0], None, [256], [0.0, 256.0])  # 直方图，统计图像中每个灰度值的数量
    gray_dist = cal_dist(hist_array)  # 灰度值与其他值的距离
    # print(gray_dist)
    for i in range(image_width):
        for j in range(image_height):
            temp = image_gray[j][i]
            image_gray_copy[j][i] = gray_dist[temp]
    image_gray_copy = (image_gray_copy - np.min(image_gray_copy)) / (np.max(image_gray_copy) - np.min(image_gray_copy))

    # print(np.max(image_gray_copy))
    # print(np.min(image_gray_copy))

    image_gray_copy = image_gray_copy * 255
    image_gray_copy = np.array(image_gray_copy,dtype=np.uint8)
    return image_gray_copy


def FT(src):
    lab = cv2.cvtColor(src,cv2.COLOR_BGR2LAB)
    gaussian_blur=cv2.GaussianBlur(src,(5,5),0)

    mean_lab = np.mean(lab,axis=(0,1))
    # print(mean_lab.shape)

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


    
    # xs = np.reshape(salient_map,(-1,1))
    # xs = np.reshape(xs,(1,len(xs)))

    # plt.hist(salient_map.ravel(), 256, [0, 256])
    # plt.show()



    # salient_map[salient_map ==0] = 255
    # salient_map[salient_map!=255] = 0

    
    # print(salient_map.shape)
    # salient_map = cv2.cvtColor(salient_map,cv2.COLOR_BGR2GRAY)
    # ret1, th1 = cv2.threshold(salient_map, 0, 255, cv2.THRESH_OTSU) 

    salient_map[salient_map<120] = 255
    salient_map[salient_map!=255] = 0

    return salient_map






if __name__ == "__main__":
    for i in range(0,len(paths)):
    # for i in range(0,1):
        path = "".join(paths[i])
        # print(path)
        mask_path = path.replace(image_ROOT,resultImgRoot)[:-4] + "_mask.jpg"
        bPath = path.replace(image_ROOT,bResultRoot)[:-4] 
        # print(mask_path)

        oriImg = cv2.imread(path)
        oriImg = cv2.cvtColor(oriImg,cv2.COLOR_BGR2GRAY)
        maskImg = cv2.imread(mask_path)
        maskImg = cv2.cvtColor(maskImg,cv2.COLOR_BGR2GRAY)
        _, _, stats, _ = cv2.connectedComponentsWithStats(maskImg)
        cropImgList = []

        x,y = stats.shape

        for j in range(1,x):
            param = stats[j,:]

            if param[4]<200:
                pass 
            else:
                cropImgList.append(oriImg[param[1]:param[1]+param[3],param[0]:param[0]+param[2]])
        # cv2.imwrite(bPath+".jpg",image_gray_copy)
        
        for j in range(0,len(cropImgList)):
            (x,y) = cropImgList[j].shape
            resimg = np.zeros((x,y,3),dtype=np.uint8)
            image_gray_copy = LC(cropImgList[j])
            resimg[:,:,0] = image_gray_copy
            resimg[:,:,1] = image_gray_copy
            resimg[:,:,2] = image_gray_copy

            # print(resimg.shape)
            image_gray_copy = FT(resimg)
            # cv2.imwrite(bPath+".jpg",cropImgList[j])
            # bPath = path.replace(image_ROOT,bResultRoot)[:-4] + "_"+str(j)+".jpg"
            BPath = bPath +"_B"+ str(j) + ".jpg"
            OPath = bPath +"_A"+ str(j) + ".jpg"
            print(BPath)

            cv2.imwrite(BPath,image_gray_copy)
            cv2.imwrite(OPath,cropImgList[j])

    



    # print(stats[1:,])
    
    # maskImg[maskImg==255] = 1
    # maskImg = maskImg[0]
    # print(np.max(maskImg))
    # print(oriImg.shape)
    # print(maskImg.shape)


