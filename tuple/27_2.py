'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-03-27 08:47:01
@LastEditors: xiaoshuyui
@LastEditTime: 2020-03-27 08:52:06
'''
if __name__ == "__main__":
    t = (1,2,3,[4,5])
    print(t)
    print(t[-1])
    t[-1][0] = "Y"
    t[-1][1] = "X"
    print(t)

    # s = set()
    # s.add('C')
    # s.add('B')

    # t[-1] = s
    # print(t)

    t[-1].append(6)
    print(t)