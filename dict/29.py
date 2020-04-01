'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-31 08:27:45
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-31 08:51:57
'''
if __name__=="__main__":
    dic = dict()

    for i in range(1,10):
        dic[i] = "我是第"+str(i)+"个值"
    
    print(dic.get(-1,6))

    # for i in dic.items():
    #     print(i)