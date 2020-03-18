'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 10:18:43
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-18 10:20:14
'''

from xadmin import views

class GlobalSettings(object):
    site_title = "xxx后台管理系统"
    site_footer = "xxx......"

xadmin.site.register(views.CommAdminView, GlobalSettings)
