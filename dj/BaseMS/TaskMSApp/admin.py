'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 17:02:44
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-19 13:34:01
'''
from django.contrib import admin
from xadmin import views

# Register your models here.
import xadmin 
from .models import Task ,StateCode
from xadmin.layout import Fieldset
from xadmin.plugins.actions import BaseActionView

# Register your models here.



# class VerifyAction(BaseActionView): # 定义一个动作
#     action_name = "测试用权限1"  # 动作名
#     description = "参数化构建"  # 要显示的名字
#     model_perm = "测试用权限1"   # 该动作所需权限
#     icon = 'fa fa-bug'

#     delete_confirmation_template = None
#     delete_selected_confirmation_template = None
 
#     delete_models_batch = True


#     def do_action(self, queryset):  # 重载do_action()方法
#         try:
#             for i in queryset:
#                 pass
#             self.message_user(message="Done", level="success")  # level的值必须小写
#         except Exception as e:
#             self.message_user(e, "error")

class CompleteAction(BaseActionView):
    '''清空action'''
    action_name = "complete_mission"    # 相当于这个Action的唯一标示, 尽量用比较针对性的名字
    description = u'完成所选 %(verbose_name_plural)s'  # 出现在 Action 菜单中名称
    model_perm = 'change'       # 该 Action 所需权限
    icon = 'fa fa-gavel'

    # 执行的动作
    def do_action(self, queryset):
        for obj in queryset:
            # 需执行model对应的字段
            # s = StateCode
            # s.
            obj.state =  StateCode(2)    # 重置score为0
            obj.save()
        # return HttpResponse
        return None  # 返回的url地址


class DelayAction(BaseActionView):
    '''清空action'''
    action_name = "delay_mission"    # 相当于这个Action的唯一标示, 尽量用比较针对性的名字
    description = u'推迟所选 %(verbose_name_plural)s'  # 出现在 Action 菜单中名称
    model_perm = 'change'       # 该 Action 所需权限
    icon = 'fa fa-times-circle'

    # 执行的动作
    def do_action(self, queryset):
        for obj in queryset:
            # 需执行model对应的字段
            # s = StateCode
            # s.
            obj.state =  StateCode(3)    # 重置score为0
            obj.save()
        # return HttpResponse
        return None  # 返回的url地址





class TaskAdmin(object):
    # pass 
    list_display = ['taskname','starter','worker','state']
    search_fields=['taskname',]
    # fieldsets = [(None, {"fields": ['taskname','starter','worker','state']})]
    # model_icon = 'fa fa-book'
    # list_editable = ['worker','comment']
    actions = [CompleteAction,DelayAction,]

    def formfield_for_dbfield(self, db_field, **kwargs):
        from BaseApp_1.models import Worker
        if db_field.name == 'worker':
            # qs1 = Worker.objects.filter(state = 1)
            # ids = []
            # for i in qs1:
            #     ids.append(i['wuuid'])
            
            kwargs['queryset'] = Worker.objects.filter(state = 1)
        
        return db_field.formfield(**dict(**kwargs))


    # def queryset(self):
    #     from django.db.models import Q
    #     from BaseApp_1.models import Worker
    #     # from django.db import models
    #     # qs1 = Worker.objects.filter(~Q(state = 1))
    #     qs1 = Worker.objects.filter(state = 1)
    #     # print(qs1)
    #     # print(type(qs1))
    #     qs = super(TaskAdmin,self).queryset()
    #     print(qs.filter(worker=qs1.wuuid))

    #     return qs.filter(worker=qs1.wuuid)

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
