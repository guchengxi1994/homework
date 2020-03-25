'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-19 08:39:15
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-19 08:41:26
'''
from django.contrib import admin
import xadmin 
from xadmin import views


class GlobalSettings(object):
    site_title="后台管理系统"
    site_footer="xiaoshuyui"
    menu_style="accordion"
xadmin.site.register(views.CommAdminView,GlobalSettings)




class BaseSetting(object):
    enable_themes = True    #添加主题选择功能
    use_bootswatch = True    #添加多个主题到选择中
    
xadmin.site.register(views.BaseAdminView, BaseSetting)