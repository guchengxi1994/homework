'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-13 17:25:49
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-13 17:27:27
'''
import pydicom
from pydicom import dcmread

filename = "D:\\dump\\338\\pipelineCode-17ZZ040-weldingCode-QJSG-NJ01-B-17ZZ040018-M_0001.dcm"
dataset = dcmread(filename)
print(dataset.pixel_array)
