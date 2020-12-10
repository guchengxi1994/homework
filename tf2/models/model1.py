'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-08-25 09:40:36
LastEditors: xiaoshuyui
LastEditTime: 2020-09-23 15:28:12
'''
from tensorflow.keras import  layers, models

def create_model(input_shape=(32,32,3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
    print(model.output_shape)
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))

    return model