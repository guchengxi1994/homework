'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 10:31:45
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-19 12:46:01
'''
from django.db import models
import uuid

# Create your models here.

class User(models.Model):
    username = models.CharField(max_length=20,verbose_name='用户名',unique=True)
    comment = models.CharField(max_length=200,verbose_name='评价',default='123456')

    class Meta:
        verbose_name = '成员'
        verbose_name_plural=verbose_name

    def __str__(self):
        return self.username


class WorkerState(models.Model):
    statedetail = models.CharField(max_length=20,verbose_name='工作状态',default='空闲')

    def __str__(self):
        return self.statedetail




class Worker(models.Model):
    workername = models.CharField(max_length=20,verbose_name='姓名')
    workerID = models.CharField(max_length=100,verbose_name='身份认证',unique=True)
    jointime = models.DateField(auto_now=True,verbose_name='加入时间')
    # state = models.CharField(max_length=20,verbose_name='工作中')
    state = models.ForeignKey(WorkerState,on_delete=models.CASCADE,related_name='workerstate_id',default=1)
    wuuid = models.UUIDField(primary_key=True, default=uuid.uuid4(), editable=False)


    class Meta:
        verbose_name = '操作员'
        verbose_name_plural=verbose_name

    def __str__(self):
        return self.workername
