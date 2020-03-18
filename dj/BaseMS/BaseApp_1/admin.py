'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 10:31:45
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-18 11:29:37
'''
from django.contrib import admin

# Register your models here.
import xadmin 
from .models import User 
from xadmin.layout import Fieldset

class UserAdmin(object):
    # pass 
    list_display = ['username']
    search_fields=['username']

    form_layout = (
        # Fieldset(None,
        #          'pc_name','pc_icorn','pc_link','sort'
        #          ),
        Fieldset(None,
                 'password',**{"style":"display:None"}
                 ),
    )

xadmin.site.register(User,UserAdmin)
