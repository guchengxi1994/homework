import cv2,os
import numpy as np



BASE_DIR = os.path.abspath(os.curdir)
model_ROOT = os.path.join(BASE_DIR, 'static')
image_ROOT = os.path.join(BASE_DIR,"images")

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

    print(np.max(image_gray_copy))
    print(np.min(image_gray_copy))

    image_gray_copy = image_gray_copy * 255
    image_gray_copy = np.array(image_gray_copy,dtype=np.uint8)
    return image_gray_copy

if __name__ == '__main__':
    img = cv2.imread(image_ROOT+os.sep+"1-2.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    saliency_image = LC(img_gray)

    cv2.imwrite(BASE_DIR+os.sep+"1-2wwww.jpg",saliency_image)