'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-03 08:28:56
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-03 08:37:52
'''
import json

if __name__ == "__main__":
    dic = {"a":1,"b":2222}

    js = json.dumps(dic)
    print(dic)
    print(js)
    print(type(js))