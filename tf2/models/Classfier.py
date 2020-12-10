'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-10-16 15:13:03
LastEditors: xiaoshuyui
LastEditTime: 2020-12-10 08:42:48
'''
import tensorflow as tf

# print(tf.__version__)
tfVersion = tf.__version__
del tf

if int(tfVersion[0]) > 1:
    from tensorflow.keras import layers, models
else:
    # pass
    from keras import layers, models


def create_model(input_shape=(648, 648, 3)):
    model = models.Sequential()
    model.add(
        layers.Conv2D(16, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    # print(model.output_shape)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    # print(model.output_shape)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    # print(model.output_shape)
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    return model


# if __name__ == "__main__":
#     model = create_model()
#     model.summary()
