'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 16:20:04
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-18 16:31:18
'''
from BaseMS.xadmin.plugins.actions import BaseActionView
from django.shortcuts import HttpResponse
from django.contrib.auth import get_permission_codename

class ActionApproveAccept(BaseActionView):
    action_name = u'accept'
    model_perm = 'approve'
    description = '同意'
    global_actions = []
    
    def do_action(self,queryset):
        for obj in queryset:
            print(obj)
        return HttpResponse("已同意")


class ActionApproveReject(BaseActionView):
    action_name = u'reject'
    model_perm = 'approve'
    description = '拒绝'
    global_actions = []
    
    def do_action(self,queryset):
        for obj in queryset:
            print(obj)
        return HttpResponse("已拒绝")

class TaskApproveSettings(ViewOnlyMixin):
    actions = [ActionApproveAccept,ActionApproveReject]

    list_display = ['name','targets','services','create_by']
    readonly_fields = TaskFields.create+TaskFields.params+TaskFields.approve
    form_layout = (
        Main(
            Fieldset('',
                     *TaskFields.create,
                     css_class='unsort no_title'
                     ),

            Fieldset(('参数'),
                     *TaskFields.params
                     ),
        ),
        Side(
            Fieldset(('审批状态'),
                     *TaskFields.approve
                     ),
        )
    )
    def get_model_perms(self):
        return {
            'view': self.has_view_permission(),
            'add': self.has_add_permission(),
            'change': self.has_change_permission(),
            'delete': self.has_delete_permission(),
            'approve': self.has_approve_permission()
        }
    def has_approve_permission(self):
        codename = get_permission_codename('approve', self.opts)
        return ('approve' not in self.remove_permissions) and self.user.has_perm('%s.%s' % (self.app_label, codename))
