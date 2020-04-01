'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-31 08:56:38
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-31 08:59:42
'''
def get(dic:dict,key,default = None):
    if dic.__contains__(key):
        return dic[key]
    else:
        return default

if __name__ == "__main__":
    dic = dict()

    for i in range(1,10):
        dic[i] = "我是第"+str(i)+"个值"

    print(get(dic,-1,6))

