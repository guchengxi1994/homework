'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 10:31:45
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-18 11:24:41
'''
from django.db import models

# Create your models here.

class User(models.Model):
    username = models.CharField(max_length=20,verbose_name='用户名',unique=True)
    comment = models.CharField(max_length=200,verbose_name='评价',default='123456')

    class Meta:
        verbose_name = '成员'
        verbose_name_plural=verbose_name
