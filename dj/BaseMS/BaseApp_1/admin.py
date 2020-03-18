'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 10:31:45
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-18 16:05:14
'''
from django.contrib import admin
from xadmin import views

# Register your models here.
import xadmin 
from .models import User 
from xadmin.layout import Fieldset

class UserAdmin(object):
    # pass 
    list_display = ['username','comment']
    search_fields=['username']
    model_icon = 'fa fa-user'
    list_editable = ['comment']

    form_layout = (
        # Fieldset(None,
        #          'pc_name','pc_icorn','pc_link','sort'
        #          ),
        Fieldset(None,
                 'password',**{"style":"display:None"}
                 ),
    )


class GlobalSettings(object):
    site_title="后台管理系统"
    site_footer="xiaoshuyui"
    menu_style="accordion"
xadmin.site.register(views.CommAdminView,GlobalSettings)


xadmin.site.register(User,UserAdmin)

class BaseSetting(object):
    enable_themes = True    #添加主题选择功能
    use_bootswatch = True    #添加多个主题到选择中
    
xadmin.site.register(views.BaseAdminView, BaseSetting)
