'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-09 08:53:43
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-09 08:56:30
'''



def func2(a:int,b:str,c=None):
    print(b)
    if c is not None:
        print(c)
    


if __name__ == "__main__":
    a = "aaaa"
    b = "bbbb"
    func2(a=a,b=b)