'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-02 08:22:09
@LastEditors: xiaoshuyui
@LastEditTime: 2020-04-02 08:51:10
'''
if __name__ == "__main__":
    d1 = {"a":1,"b":2}
    d2 = {"c":3,"d":4}

    # d1.setdefault('e',8)

    # print(d1)

    # res = d2.pop("c")
    # print(d2)
    # print(res)

    res = d1.popitem()
    print(d1)
    print(res)

