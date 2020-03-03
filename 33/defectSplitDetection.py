# -*- coding: utf-8 -*-

"""
Read images
Padding images
Cut images

@author: xiaoshuyui
"""

import cv2
import glob
import os
import utils.defectPadding as pd
import utils.splitimgs as sp
from utils.splitimgs import Img_ID
import utils.recom  as recom
import utils.opencv_based_img as ob 
import utils.opencv_based_weld as obw
import utils.opencv_based_weld_params as obp
import numpy as np 


BASE_DIR = os.path.abspath(os.curdir)
defect_model_ROOT = os.path.join(BASE_DIR, 'static'+os.sep+"def")
weld_model_ROOT = os.path.join(BASE_DIR, 'static'+os.sep+"MyWeld")

image_ROOT = os.path.join(BASE_DIR,"ori")
# WELD_path = model_ROOT+os.sep+'MyWeld'+os.sep+'yolov3-voc.backup'
# WELD_labels_path = model_ROOT+os.sep+'MyWeld'+os.sep+'voc.names'
# WELD_config_path = model_ROOT+os.sep+'MyWeld'+os.sep+'yolov3-voc.cfg'
# WELD = cv2.dnn.readNetFromDarknet(WELD_config_path,WELD_path)
# WordLabels = open(WELD_labels_path).read().strip().split('\n')




#defect model
# weights_path='/home/junrui123/defect/models/small/yolov3-voc.backup'
# config_path='/home/junrui123/defect/models/yolov3-voc.cfg'
# label_path='/home/junrui123/defect/models/mysmall.names'

weights_path = defect_model_ROOT+os.sep +'yolov3-voc.backup'
config_path = defect_model_ROOT+os.sep + 'yolov3-voc.cfg'
label_path = defect_model_ROOT+os.sep + 'mysmall.names'
LABELS = open(label_path).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

#weld model

# weld_weights_path='./MyWeld/yolov3-voc.backup'
# weld_config_path='./MyWeld/yolov3-test.cfg'
# weld_label_path='./MyWeld/voc.names'
weld_weights_path = weld_model_ROOT+os.sep+'yolov3-voc.backup'
weld_config_path = weld_model_ROOT+os.sep+'yolov3-test.cfg'
weld_label_path = weld_model_ROOT+os.sep+'voc.names'
weld_LABELS = open(weld_label_path).read().strip().split("\n")
weld_net = cv2.dnn.readNetFromDarknet(weld_config_path, weld_weights_path)


# big defect model



class loc(object):
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y 
        self.w = w 
        self.h = h


#遍历目标文件夹
PATH= image_ROOT
# PATH = "./testORI"
paths = glob.glob(os.path.join(PATH, r'*.jpg'))
#paths.append(glob.glob(os.path.join(PATH, r'*.jpg')))
# print(paths)
for i in range(0,len(paths)):
    path = "".join(paths[i])
    print(path)
    print("......")
    temp = path.split(os.sep)
    imgpath = temp[-1]
    temp = imgpath.split(".")
    imgname = temp[0]
    img = cv2.imread(path)

    # weldRes = obw.yolo_detect(None, None, None, nnnetwork=weld_net, nnlabels=weld_LABELS,oriimg=img)

    

    # LOCs = []
    # if len(weldRes)>0:
    #     for wR in weldRes:
    #         if "IQI" == wR.name:
    #             pass
    #         else:
    #             x,y = wR.location[0] ,wR.location[1]
    #             w,h = wR.w ,wR.h

    #             temp = loc(x,y,w,h)
    #             LOCs.append(temp)


  



    (x,y,z) = img.shape
    # print(imgshape)
    padding_img,imgtype,flag = pd.reshape(img)

    if flag:
        imgshape = (y,x,z)
    else:
        imgshape = (x,y,z)
    imgList,imgListAnother,_ = sp.imgsplit2(padding_img,imgname,imgtype)

    # DefectList = []

    res = recom.putTogether_small(imgList,imgtype)
    res2 = np.zeros(res.shape,dtype=np.uint8)
    # print(res.shape)
    imgList.extend(imgListAnother)

    #神经网络
    for i in imgList:
        result = obp.yolo_detect(None, None, None, nnnetwork=net, nnlabels=LABELS,oriimg=i.img)
        # temp = Img_ID(Im,i.ID)
        # DefectList.append(temp)

        for r in result:
            w = r.w
            h = r.h
            x,y = r.location[0],r.location[1]
            # print(r.name)
            

            x = int((i.ID-1)*416+x)

            COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

            color = [int(c) for c in COLORS[int(i.ID)]]
            res2[y:y+h,x:x+w] = 255
            # text = "{}: {:.4f}".format(r.name, r.confidence)

            cv2.rectangle(res, (x, y), (x + w, y + h), color, 1)  # 画框
            # cv2.putText(res, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # cv2.rectangle(res, (x, y), (x + w, y + h), color, 1)  # 画框



    


    # for i in imgListAnother:
    #     result = obp.yolo_detect(None, None, None, nnnetwork=net, nnlabels=LABELS,oriimg=i.img)

    #     for r in result:
    #         w = r.w
    #         h = r.h
    #         x,y = r.location[0],r.location[1]

    #         x = int((i.ID-1)*416+x)

    #         COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

    #         color = [int(c) for c in COLORS[int(i.ID)]]

    #         cv2.rectangle(res, (x, y), (x + w, y + h), color, 1)  # 画框







    res = cv2.resize(res, (imgshape[1],imgshape[0]))
    res2 = cv2.resize(res2, (imgshape[1],imgshape[0]))


    # if len(LOCs)>0:
    #     # print(True)
    #     for l in LOCs:
    #         x = l.x 
    #         y = l.y 
    #         w = l.w 
    #         h = l.h 
    #         img[ int(y) if int(y)>0 else 0 :int(y+h) if int(y+h) < imgshape[0] else imgshape[0] , int(x) if int(x)>0 else 0 :int(x+w) if int(x+w) < imgshape[1] else imgshape[1] ] = res[ int(y) if int(y)>0 else 0 :int(y+h) if int(y+h) < imgshape[0] else imgshape[0] , int(x) if int(x)>0 else 0 :int(x+w) if int(x+w) < imgshape[1] else imgshape[1] ]

    #         res = img




    cv2.imwrite(BASE_DIR+os.sep+"result2"+os.sep+imgname+".jpg",res)
    cv2.imwrite(BASE_DIR+os.sep+"result2"+os.sep+imgname+"_mask.jpg",res2)

    # print(len(imgList))

    # imgList.sort(key=lambda x:x.ID,reverse=False)



