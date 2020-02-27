import os
from utils import opencv_based_weld2 as ob
from utils import getStrings as gs
from utils import getWeld as gw
import cv2
import numpy as np 
import time

BASE_DIR = os.path.abspath(os.curdir)
model_ROOT = os.path.join(BASE_DIR, 'static')
image_ROOT = os.path.join(BASE_DIR,"images")
WORDNET_path = model_ROOT+os.sep+'Words'+os.sep+'yolov3-voc_last.weights'
WORDNET_labels_path = model_ROOT+os.sep+'Words'+os.sep+'voc.names'
WORDNET_config_path = model_ROOT+os.sep+'Words'+os.sep+'yolov3-voc.cfg'
WORDNET = cv2.dnn.readNetFromDarknet(WORDNET_config_path,WORDNET_path)
WordLabels = open(WORDNET_labels_path).read().strip().split('\n')


time1 = time.time()


pipeCode = "300ME301013B3421H031"

def dpi2mm(distance,scale=300):

    return distance*25.4/scale

def getR(d):
    return d/2/3.14


imagePaths = [image_ROOT+os.sep+"1-2.jpg",image_ROOT+os.sep+"2-3.jpg",image_ROOT+os.sep+"3-4.jpg",\
    image_ROOT+os.sep+"4-5.jpg",image_ROOT+os.sep+"5-6.jpg",image_ROOT+os.sep+"6-1.jpg"]


result = []

for i in imagePaths:
    image = cv2.imread(i)
    
    imgshape = image.shape

    if imgshape[0]>imgshape[1]:
        trans_img = cv2.transpose(image)
        image = cv2.flip(trans_img, 1)
    
    t1 = cv2.flip(image,0)
    t2 = cv2.flip(image,1)
    t3 = cv2.flip(image,2)
    index = 0
    lis = []

    defaultSim = 0
    # picLoc = []
    pp = None
    code = ""
    for p in [image,t1,t2,t3]:
        res = ob.yolo_detect(None,None,None,nnnetwork=WORDNET,nnlabels=WordLabels,oriimg=p)
        words,picLis = gs.getStringByThresh(res,imgshape=imgshape)
        
        r = gw.getCorrectCodeOnlySim(words,pipeCode,index)
        if r>defaultSim:
            defaultSim = r 
            code = words 
            # picLoc = picLoc.append(picLis)
            pp = picLis
    result.append(pp)
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(code)
    # for i in picLoc:
    #     print(i)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        # lis.append(r)
        # index = index + 1

    # lis.sort(key=lambda x:x.sim ,reverse=True)

    # result.append(lis[0].words)
d = 0
for i in result:
    # print(i[0])
    # print(i[1])
    d = d + abs(i[0].location[0]-i[1].location[0])

lenth = dpi2mm(d)

print(lenth)

R = getR(lenth)
print(R)

time2 = time.time()

print(time2-time1)


# print(result)



