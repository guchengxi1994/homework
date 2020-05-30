'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-07 08:28:26
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-07 08:47:43
'''

import json

if __name__ == "__main__":
    dic = {'name':"xiaoshuyui",'sex':"male", \
        'age':18,'introduction':"handsome very"}
    
    j = json.dumps(dic)
    print(j)

    dic2 = json.loads(j)
    print(dic2)
    print(type(dic2))

    