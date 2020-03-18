'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 17:02:44
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-18 17:23:55
'''
from django.db import models

# Create your models here.

class Task(models.Model):
    taskname = models.CharField(max_length=20,verbose_name='任务名',unique=True)
    comment = models.CharField(max_length=200,verbose_name='评价',default='123456')
    starter = models.CharField(max_length=20,verbose_name='发起人',default='')
    worker = models.CharField(max_length=20,verbose_name='指派工人',default='')

    class Meta:
        verbose_name = '任务'
        verbose_name_plural=verbose_name

    def __str__(self):
        return self.username
