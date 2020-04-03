'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-03 08:40:24
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-03 09:18:23
'''
import json
import numpy as np

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj,MyClass):
            return obj.__dict__()
        return json.JSONEncoder.default(self, obj)

class MyClass(object):
    def __init__(self,name:str,age:int):
        self.name = name 
        self.age = age 

    def __dict__(self):
        dic = {}
        # dic['姓名'] = self.name
        # dic['年龄'] = self.age
        dic['name'] = self.name
        dic['age'] = str(self.age)

        return dic


if __name__ == "__main__":
    myclass = MyClass('xiaohua',18)
    js = myclass.__dict__()
    print(js)

    dic = {'class1':myclass}
    js2 = json.dumps(dic,cls=MyEncoder,indent=4)
    print(js2)


        