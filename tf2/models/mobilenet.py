'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-09-23 14:46:40
LastEditors: xiaoshuyui
LastEditTime: 2020-09-23 15:34:11
'''
from tensorflow.keras import layers,models

import tensorflow.keras as keras

def mobileV1(inputShape=(224,224,3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),input_shape=inputShape,activation='relu',padding='same'))  # 224*224

    model.add(layers.DepthwiseConv2D((3,3),strides=(2,2),padding='same'))      #112

    # print(model.output_shape)

    model.add(layers.Conv2D(64,(1,1),activation='relu'))       # 112
    model.add(layers.DepthwiseConv2D((3,3),strides=(2,2),padding='same'))     # 56
    model.add(layers.Conv2D(128,(1,1),activation='relu'))      #56
    model.add(layers.DepthwiseConv2D((3,3),padding='same'))                    # 56
    model.add(layers.Conv2D(128,(1,1),activation='relu'))       # 56
    model.add(layers.DepthwiseConv2D((3,3),strides=(2,2),padding='same'))                   # 28
    model.add(layers.Conv2D(256,(1,1),activation='relu'))            # 28      
    model.add(layers.DepthwiseConv2D((3,3),padding='same'))               # 28
    model.add(layers.Conv2D(256,(1,1),activation='relu'))         #28
    model.add(layers.DepthwiseConv2D((3,3),strides=(2,2),padding='same'))    # 14

    # model.add(layers.DepthwiseConv2D((3,3),padding='same'))
    model.add(layers.Conv2D(512,(1,1),activation='relu'))

    model.add(layers.DepthwiseConv2D((3,3),padding='same'))
    model.add(layers.Conv2D(512,(1,1),activation='relu'))

    model.add(layers.DepthwiseConv2D((3,3),padding='same'))
    model.add(layers.Conv2D(512,(1,1),activation='relu'))

    model.add(layers.DepthwiseConv2D((3,3),padding='same'))
    model.add(layers.Conv2D(512,(1,1),activation='relu'))

    model.add(layers.DepthwiseConv2D((3,3),padding='same'))
    model.add(layers.Conv2D(512,(1,1),activation='relu'))

    # print(model.output_shape)

    model.add(layers.DepthwiseConv2D((3,3),strides=(2,2),padding='same'))    #7

    model.add(layers.Conv2D(1024,(1,1),activation='relu'))
    model.add(layers.DepthwiseConv2D((3,3),padding='same'))
    model.add(layers.Conv2D(1024,(1,1),activation='relu'))

    model.add(layers.AveragePooling2D(pool_size=(7,7)))

    # model.add(layers.)
    return model



if __name__ == "__main__":
    m = mobileV1()

    m.add(layers.Flatten())
    m.add(layers.Dense(1024, activation='relu'))
    m.add(layers.Dense(1000))

    m.summary()

    