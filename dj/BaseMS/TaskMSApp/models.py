'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 17:02:44
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-19 17:22:10
'''
from django.db import models
from BaseApp_1.models import User,Worker
import uuid

# Create your models here.

class StateCode(models.Model):
    statedetails = models.CharField(max_length=20,verbose_name='状态',default='未完成')

    def __str__(self):
        return self.statedetails

class Task(models.Model):
    taskname = models.CharField(max_length=20,verbose_name='任务名',unique=True)
    comment = models.CharField(max_length=200,verbose_name='评价',default='123456')
    # starter = models.CharField(max_length=20,verbose_name='发起人',default='')
    starter = models.ForeignKey(User,on_delete=models.CASCADE,related_name='user_id',verbose_name='发起人')
    # worker = models.CharField(max_length=20,verbose_name='操作员',default='')
    worker = models.ForeignKey(Worker,on_delete=models.CASCADE,related_name='worker_uuid',verbose_name='操作员')
    state = models.ForeignKey(StateCode,on_delete=models.CASCADE,related_name='statecode_id',default=1,verbose_name='完成状态')
    tuuid = models.UUIDField(primary_key=True, default=uuid.uuid4(), editable=False)

    startTime = models.DateTimeField(auto_now=True,verbose_name='开始时间')
    endTime = models.DateTimeField(auto_now=False,verbose_name='结束时间',blank=True,editable=False, null=True)
    primary = models.IntegerField(verbose_name='优先级',default=1,max_length=3)
    # state = models.CharField(max_length=20,verbose_name='状态',default='0')

    class Meta:
        verbose_name = '任务'
        verbose_name_plural=verbose_name

    def __str__(self):
        return self.taskname
