'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-12-09 15:29:24
LastEditors: xiaoshuyui
LastEditTime: 2020-12-10 11:29:15
'''
from tensorflow.keras import layers
from tensorflow.python.keras.layers.merge import Add


def getEntryFlow(inputParams):
    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      activation='relu',
                      padding='same')(inputParams)
    # x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    kernelNumbers = [128, 256, 728]
    for i in range(0, 3):
        y = layers.Conv2D(kernelNumbers[i], (1, 1),
                          strides=(2, 2),
                          padding='same')(x)

        x = layers.SeparableConv2D(kernelNumbers[i],
                                   kernel_size=(3, 3),
                                   padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.SeparableConv2D(kernelNumbers[i],
                                   kernel_size=(3, 3),
                                   padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.SeparableConv2D(kernelNumbers[i], (3, 3),
                                   strides=(2, 2),
                                   padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = Add()([x, y])
    return x


def getMiddleFlow(inputParams):
    x = inputParams
    for _ in range(0, 16):
        x = layers.Activation('relu')(x)

        x = layers.SeparableConv2D(728, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.SeparableConv2D(728, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.SeparableConv2D(728, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = Add()([x, inputParams])

    return x


def getExitFlow(inputParams):
    y = x = inputParams
    y = layers.Conv2D(1024, (1, 1), strides=(2, 2), padding='same')(y)

    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(728, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(1024, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(1024, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = Add()([x, y])

    x = layers.SeparableConv2D(1536, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(1536, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(2048, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def getModel(inputShape=(299, 299, 3)):
    inputParams = layers.Input(shape=inputShape)
    entry = getEntryFlow(inputParams)
    middle = getMiddleFlow(entry)
    exitFlow = getExitFlow(middle)
    return exitFlow


if __name__ == "__main__":
    model = getModel()
    print(model.shape)