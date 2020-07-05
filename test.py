import xml.etree.ElementTree as ET
from skimage import io
import cv2
import os,glob
import numpy as np
from skimage import exposure

NoneHighLightWeld = set()
__weidth__ = 400    # or 350

def _his(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if len(img.shape) == 1:
        
        equ = clahe.apply(img)
    else:
        b,g,r = cv2.split(img)
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)

        equ = cv2.merge([b,g,r])
    
    return equ

class HighLight(object):
    def __init__(self,name:str,left_top,width,height):
        self.name = name
        self.left_top = left_top
        self.width = width
        self.height = height

    def __sub__(self,other):
        if self.name != other.name:
            dis  = 10000
        else:
            dis = self.left_top[1] - other.left_top[1]
        return dis
    
    
class P_P_S(object):
    def __init__(self,p1,p2,s):
        self.p1 = p1
        self.p2 = p2
        self.s = s

    def __eq__(self,other):
        if self.p1 == other.p1 and self.p2 == other.p2:
            return True
        elif self.p1 == other.p2 and self.p2 == other.p1:
            return True
        else:
            return False
    
    def __hash__(self):
        return hash(self.p1) + hash(self.p2)

    def __str__(self):
        return self.p1 +'>>>>>>>>>' +self.p2 +'>>>>>>>>>'+ str(self.s) +'\n'


class Weld(object):
    def __init__(self,name:str,imgShape:tuple,highLight:list):
        self.name = name
        self.imgShape = imgShape
        self.highLight = highLight

    def __hash__(self):
        return hash(self.name)

    def __eq__(self,other):
        return self.name == other.name

    def __sub__(self,other):
        if not isinstance(other,self.__class__):
            return 0
        elif max(self.imgShape)/max(other.imgShape)>2 or \
            max(self.imgShape)/max(other.imgShape)<0.5:
            return 0
        elif len(self.highLight) == 0 or len(other.highLight) == 0:
            global NoneHighLightWeld
            NoneHighLightWeld.add(self)
            return 0
        else:
            res = []
            figureList = []
            flag = False
            self.highLight.sort(key=lambda  x: x.left_top[1])
            other.highLight.sort(key=lambda  x: x.left_top[1])
            for i in self.highLight:
                for j in other.highLight:
                    # print(i-j)
                    if abs(i-j)<__weidth__:
                        # figureList.append(i-j)
                        res.append((i,j))
            
            if len(res)>0:
                img1 = io.imread(self.name)
                img1 = _his(img1)
                img2 = io.imread(other.name)
                img2 = _his(img2)
                for i in res:
                    sH = i[0]
                    oH = i[1]

                    # time1 
                    left_top = sH.left_top
                    # print(left_top)
                    w = sH.width
                    h = sH.height
                    # x = img1[left_top[0]-50:left_top[0]+50+h,left_top[1]-100:left_top[1]+100+w]

                    x = img1[left_top[0]-50:left_top[0]+50+h,left_top[1]-50:left_top[1]+50+w]
                    
                    match = img2[:,left_top[1]-__weidth__ if left_top[1]-__weidth__>0 else 0 \
                        :left_top[1]+w+__weidth__ if left_top[1]+w+__weidth__<img2.shape[1] else img2.shape[1]]

                    # x = _his(x)
                    # match = _his(match)
                    
                    result = cv2.matchTemplate(match, x, cv2.TM_CCOEFF_NORMED)
                    _, max_val1, _, _ = cv2.minMaxLoc(result)
                    
                    figureList.append(max_val1)



                    left_top = oH.left_top
                    # print(left_top)
                    w = oH.width
                    h = oH.height
                    x = img2[left_top[0]-50:left_top[0]+50+h,left_top[1]-50:left_top[1]+50+w]

                    # x = img2[left_top[0]-50:left_top[0]+50+h,left_top[1]-100:left_top[1]+100+w]                    

                    match = img1[:,left_top[1]-__weidth__ if left_top[1]-__weidth__>0 else 0 \
                        :left_top[1]+w+__weidth__ if left_top[1]+w+__weidth__<img1.shape[1] else img1.shape[1]]

                    
                    # x = _his(x)
                    # match = _his(match)
                    
                    result = cv2.matchTemplate(match, x, cv2.TM_CCOEFF_NORMED)
                    _, max_val2, _, _ = cv2.minMaxLoc(result)
                    
                    figureList.append(max_val2)

                    if 0.5*(max_val1+max_val2)>0.95 and len(res)>=3:
                        flag = True
                
                print(figureList)

                

                return np.min(figureList) if flag==False else np.max(figureList)
            
            else:
                return 0







def readXml(path):
    tree = ET.ElementTree()
    tree.parse(path)
    root = tree.getroot()
    objects_size = root.findall('size')
    image_width = int(objects_size[0].find('width').text)
    image_height = int(objects_size[0].find('height').text)
    fileName = root.find('path').text
    objs = root.findall('object')
    hilights = []
    if len(objs)>0:
        for i in objs:
            bbox = i.find('bndbox')
            className = i.find('name').text
            # x是水平方向，y是垂直方向
            xmin = int(bbox.find('xmin').text.strip())  
            xmax = int(bbox.find('xmax').text.strip())
            ymin = int(bbox.find('ymin').text.strip())
            ymax = int(bbox.find('ymax').text.strip())

            width = xmax - xmin
            height = ymax - ymin

            left_top = (ymin,xmin)

            h = HighLight(className,left_top,width,height)

            hilights.append(h)
    else:
        hilights = []
    
    weld = Weld(fileName,(image_height,image_width),hilights)

    return weld




def _testReadXml():
    x = readXml('d:\\Desktop\\faketest\\labels\\XQⅡ-CF082-M005-W-01.xml')
    return x


if __name__ == "__main__":
    # img = io.imread('d:\\Desktop\\faketest\\imgs\\XQⅡ-CF082-M005-W-01.jpg')
    # weld = _testReadXml()

    # hi = weld.highLight

    # for i in range(0,len(hi)):
    #     left_top = hi[i].left_top  # top , left

    #     w = hi[i].width
    #     h = hi[i].height

    #     x = img[left_top[0]:left_top[0]+h,left_top[1]:left_top[1]+w]

        # io.imsave('d:\\Desktop\\faketest\\{}.jpg'.format(i),x)


    

    img1 = readXml('d:\\Desktop\\faketest\\labels\\XQⅡ-GH008+M197-02.xml')

    img2 = readXml('d:\\Desktop\\faketest\\labels\\XQⅡ-GH009+M048-02.xml')

    x = img1 - img2


    

    # imgs = glob.glob('d:\\Desktop\\faketest\\labels\\*.xml')
    # welds = []
    # s = set()
    # for i in imgs:
    #     welds.append(readXml(i))
    # f = open('d:\\Desktop\\faketest\\705.txt','a')
    # for i in range(0,len(welds)):
    #     for j in range(1,len(welds)):
    #         if i!=j :
    #             sim = welds[i]-welds[j]
    #             if welds[i]-welds[j]>0.9:
    #             # f.write(welds[i].name+'>>>>>>>>>>'+welds[j].name+'>>>>>>>>>>>>>'+str(welds[i]-welds[j])+'\n')
    #                 s.add(P_P_S(welds[i].name,welds[j].name,sim))


    # ss = list(s)
    # ss.sort(key=lambda  x: float(x.s),reverse=True)
    # for i in ss:
    #     f.write(str(i))
    
    # f.close()






