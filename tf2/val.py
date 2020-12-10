'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-10-19 16:00:20
LastEditors: xiaoshuyui
LastEditTime: 2020-10-19 16:49:16
'''
from keras.models import load_model
from skimage import io
import cv2
import numpy as np

model = load_model('D:/tf2/my_model.h5')

image = io.imread('D:\\cut\\classfier\\val\\0\\XQⅡ-EK009+M110-01.jpg')
# image = io.imread('D:\\cut\\classfier\\val\\1\\XQⅡ-EG008+M011-W-02.jpg')


image = cv2.resize(image,(648,648))
image = [image,]
image = np.array(image)
print(image.shape)

y = model.predict(image)

print(y)



