'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-10-19 14:16:00
LastEditors: xiaoshuyui
LastEditTime: 2020-10-19 16:28:25
'''
from models.model2classfier import create_model
import tensorflow as tf 

if tf.__version__.startswith('2'):
    import tensorflow.keras as keras
else:
    import keras

del tf

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

data_dir = '/home/aijr/cla/classfier'

batch_size = 64
img_height = 648
img_width = 648

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_ds = train_datagen.flow_from_directory(
    data_dir+'/train',
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_ds = test_datagen.flow_from_directory(
    data_dir+'/val',
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# train_ds = keras.preprocessing.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset='training',
#     seed=123,
#     image_size=(img_height,img_width),
#     batch_size=batch_size
# )

# val_ds = keras.preprocessing.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=(img_height, img_width),
#     batch_size=batch_size)

model = create_model((img_height,img_width,3))
model.summary()

model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

epochs = 20

history = model.fit_generator(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss=history.history['loss']
# val_loss=history.history['val_loss']

model.save('my_model.h5')

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()