'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-08 08:43:42
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-08 08:46:08
'''

def com_plus(a:complex,b:complex):
    return a + b


if __name__ == "__main__":
    a = complex(1,5)
    b = complex(10,2.6)

    print(a)
    print(b)

    print(com_plus(a,b))