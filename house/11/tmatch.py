import cv2 
import numpy as np


def preFix(img1,img2):
    ishape1 = img1.shape
    ishape2 = img2.shape
     
    x1 ,y1 = ishape1[0],ishape1[1]
    x2 ,y2 = ishape2[0],ishape2[1]


    fixX = min(x1,x2)
    fixY = min(y1,y2)

    img1 = img1[0:fixX,0:fixY]
    img2 = img2[0:fixX,0:fixY]

    return img1,img2



template  = cv2.imread('D:\\testAlg\\homework\\house\\11\\tem.jpg')
# template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
# print(template.shape)
img = cv2.imread("D:\\testAlg\\homework\\house\\11\\6-1.jpg")
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(img.shape)
h, w = template.shape[:2]

img2 = cv2.imread("D:\\testAlg\\homework\\house\\11\\5-6.jpg")

# img,img2 = preFix(img,img2)





# 相关系数匹配方法：cv2.TM_CCOEFF
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

left_top1 = max_loc  # 左上角
# right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角

res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

left_top2 = max_loc  # 左上角
# right_bottom2 = (left_top2[0] + w, left_top2[1] )

# leftImg = img2[:,0:left_top2[0]]
# rightImg = img[:,left_top1[0]:]

# res = np.concatenate( (leftImg,rightImg),axis=1)


#修正两个左上角

topLeft = 0
bottomLeft = 0
topRight = 0
bottomRight = 0


#   左边             #右边
if left_top2[1] == left_top1[1]:
    # pass 
    bottomLeft = bottomRight = min(img.shape[1],img2.shape[1])
elif left_top2[1]>left_top1[1]:
    topLeft = left_top2[1] - left_top1[1] +2
    bottomLeft = bottomRight = min(img.shape[1],img2.shape[1]-topLeft )
else:
    topRight = left_top1[1] - left_top2[1] +2
    bottomLeft = bottomRight = min(img.shape[1] -topRight ,img2.shape[1] )


leftImg = img2[topLeft:bottomLeft,0:left_top2[0]]
rightImg = img[topRight:bottomRight,left_top1[0]:]

print(leftImg.shape)
print(rightImg.shape)

res = np.concatenate( (leftImg,rightImg),axis=1)  



tem = np.zeros(res.shape)

tem[:,left_top2[0]-10:left_top2[0]+10] = 255

tem = np.array(tem,dtype=np.uint8)

# print(tem.shape)


# res = np.zeros((max(img.shape[0],img2.shape[0]),img.shape[1]+img2.shape[1]))

# res[0:img2.shape[0],0:left_top2[0]] = img2[:,0:left_top2[0]]
cv2.imwrite("D:\\testAlg\\homework\\house\\11\\212121212.jpg",res)

dd = cv2.inpaint(res[:,:,0],tem[:,:,0],3,cv2.INPAINT_TELEA)

cv2.imwrite("D:\\testAlg\\homework\\house\\11\\12121212.jpg",dd)


# cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置
# cv2.imshow("aaa",img)
# cv2.waitKey(0)