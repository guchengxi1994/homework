'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-08 08:50:45
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-08 08:52:34
'''
def mySum(n:int):
    s = 0
    for i in range(1,n+1):
        s = s + i 
    return s

if __name__ == "__main__":
    
    print(mySum(10))