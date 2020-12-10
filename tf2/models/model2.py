'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-08-25 10:17:10
LastEditors: xiaoshuyui
LastEditTime: 2020-09-09 16:38:44
'''
from tensorflow.keras import  layers, models
import tensorflow.keras as keras

def create_model(input_shape,num_classes):
    model = keras.Sequential(layers.experimental.preprocessing.Rescaling(1./255,input_shape=input_shape))
    # model = model.add(layers.experimental.preprocessing.Rescaling(1./255,input_shape=input_shape))
    model.add(layers.Conv2D(16,3,padding='same',activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(32,3,padding='same',activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,3,padding='same',activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(num_classes))

    return model



