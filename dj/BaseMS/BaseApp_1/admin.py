'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 10:31:45
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-19 09:43:43
'''
from django.contrib import admin
from xadmin import views

# Register your models here.
import xadmin 
from .models import User ,Worker
from xadmin.layout import Fieldset


class UserAdmin(object):
    # pass 
    list_display = ['username','comment']
    search_fields=['username']
    model_icon = 'fa fa-user'
    list_editable = ['comment']

    # form_layout = (
        # Fieldset(None,
        #          'pc_name','pc_icorn','pc_link','sort'
        #          ),
        # Fieldset(None,
        #          'password',**{"style":"display:None"}
        #          ),
    # )
xadmin.site.register(User,UserAdmin)







class WorkerAdmin(object):
    list_display = ['workername','workerID']
    search_fields=['workerID']
    model_icon = 'fa fa-users '
    # list_editable = ['comment']
    # actions = [VerifyAction]
xadmin.site.register(Worker,WorkerAdmin)















