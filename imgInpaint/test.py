import cv2
import numpy as np

global img
global point1, point2
def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
        cv2.imshow('image', img2)
        min_x = min(point1[0],point2[0])     
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        # print(point1)
        # print(point2)
        # # cut_img = img[min_y:min_y+height, min_x:min_x+width]
        # # cv2.imwrite('D:\\testALg\\homework\\imgInpaint\\lena3.jpg', cut_img)

        # return point1,point2

def mancrop():
    global img,point1, point2
    img = cv2.imread('D:\\testALg\\homework\\imgInpaint\\lenna.png')
    
    imgShape = img.shape 
    x,y = imgShape[0],imgShape[1]

    tem = np.zeros((x,y),dtype=np.uint8)



    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    # print(point1)

    tem[point1[1]:point2[1],point1[0]:point2[0]] = 255

    dd = cv2.inpaint(img[:,:,0],tem,3,cv2.INPAINT_TELEA)

    cv2.imshow('image2', dd)
    cv2.waitKey(0)


def autoFix():
    tem = cv2.imread("D:\\testALg\\homework\\imgInpaint\\tem.jpg")
    ori = cv2.imread("D:\\testALg\\homework\\imgInpaint\\ori.jpg")

    dd = cv2.inpaint(ori[:,:,0],tem[:,:,0],3,cv2.INPAINT_TELEA)
    cv2.imwrite("D:\\testALg\\homework\\imgInpaint\\dd.jpg",dd)






if __name__ == '__main__':
    # mancrop()
    autoFix()