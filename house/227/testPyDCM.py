'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-13 17:25:49
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-27 13:49:01
'''
import pydicom
from pydicom import dcmread
import numpy as np

filename = "D:\\dump\\338\\pipelineCode-17ZZ040-weldingCode-QJSG-NJ01-B-17ZZ040018-M_0001.dcm"
dataset = dcmread(filename)
print(dataset.pixel_array.shape)

print(np.max(dataset.pixel_array))
print(np.min(dataset.pixel_array))

