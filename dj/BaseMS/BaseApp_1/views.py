'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-18 10:31:45
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-18 15:07:04
'''
from django.shortcuts import render

# Create your views here.

def page_not_found(request,exception):
    #404
    return render(request,"errors/404.html",status=404)

def server_error(exception):
    
    return render("errors/500.html",status=500)

def permission_denied(request,exception):
    #403
    return render(request, "errors/403.html",status=403)

def bad_request(request,exception):
    #400
    return render(request, "errors/400.html",status=400)
