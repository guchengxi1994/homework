'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-12-11 15:58:20
LastEditors: xiaoshuyui
LastEditTime: 2020-12-11 16:16:58
'''
from tensorflow.keras import layers
from tensorflow.keras import models

def vgg16(input_shape=(224,224,3)):
    model = models.Sequential()
    model.add(layers.Conv2D(64,(3,3),padding='same',input_shape=input_shape))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(64,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(padding='same'))
    # print(model.output_shape)

    model.add(layers.Conv2D(128,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(128,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(padding='same'))

    model.add(layers.Conv2D(256,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(padding='same'))

    model.add(layers.Conv2D(512,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(padding='same'))

    model.add(layers.Conv2D(512,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(padding='same'))

    return model



def vgg19(input_shape=(224,224,3)):
    model = models.Sequential()
    model.add(layers.Conv2D(64,(3,3),padding='same',input_shape=input_shape))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(64,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(padding='same'))
    # print(model.output_shape)

    model.add(layers.Conv2D(128,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(128,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(padding='same'))

    model.add(layers.Conv2D(256,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(padding='same'))

    model.add(layers.Conv2D(512,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(padding='same'))

    model.add(layers.Conv2D(512,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512,(3,3),padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(padding='same'))

    return model

    


if __name__ == "__main__":
    model = vgg19()
    # model = vgg16()
    model.add(layers.Flatten())
    model.add(layers.Dense(4096))
    model.add(layers.Dense(4096))
    model.add(layers.Dense(1000))

    model.add(layers.Dense(10,activation='softmax'))
    model.compile()
    model.summary()

    