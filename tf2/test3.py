'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-08-25 10:57:41
LastEditors: xiaoshuyui
LastEditTime: 2020-08-25 10:59:47
'''
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

dataset, info = tfds.load('oxford_iiit_pet:3.2.0', with_info=True)