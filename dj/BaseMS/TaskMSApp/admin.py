'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 17:02:44
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-18 17:42:16
'''
from django.contrib import admin
from xadmin import views

# Register your models here.
import xadmin 
from .models import Task 
from xadmin.layout import Fieldset

# Register your models here.

class TaskAdmin(object):
    # pass 
    list_display = ['taskname','starter','worker','state']
    search_fields=['taskname','starter','worker','state']
    model_icon = 'fa fa-book'
    list_editable = ['worker','comment']

xadmin.site.register(Task,TaskAdmin)


# class GlobalSettings(object):
#     site_title="后台管理系统"
#     site_footer="xiaoshuyui"
#     menu_style="accordion"
# xadmin.site.register(views.CommAdminView,GlobalSettings)


# class BaseSetting(object):
#     enable_themes = True    #添加主题选择功能
#     use_bootswatch = True    #添加多个主题到选择中
    
# xadmin.site.register(views.BaseAdminView, BaseSetting)
