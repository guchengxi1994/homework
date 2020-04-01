'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-01 08:30:12
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-01 08:41:37
'''
if __name__ == "__main__":
    dic = {}
    for i in range(1,10):
        dic[i] = "我是第"+str(i)+"个值"

    print(dic)

    del dic
    print(dic)

    # dic.clear()
    # print(dic)
    # del dic[-1]
    # print(dic)