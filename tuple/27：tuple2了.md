<!--
 * @lanhuage: python
 * @Descripttion: 
 * @version: beta
 * @Author: xiaoshuyui
 * @Date: 2020-03-27 08:36:09
 * @LastEditors: xiaoshuyui
 * @LastEditTime: 2020-03-27 08:54:17
 -->
&emsp;&emsp;上次讲了tuple基本应用，讲到了tuple不可更改，但是tuple这个类型是支持__add__()方法的。这个方法，顾名思义就是支持两者相加。

    if __name__ == "__main__":
        a = (1,2,3)
        b = (4,)
        print(type(a))
        print(type(b))
        print(a + b)

    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/tuple/27_1.py
    <class 'tuple'>
    <class 'tuple'>
    (1, 2, 3, 4)

&emsp;&emsp;再试试__sub__()方法。

    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/tuple/27_1.py
    <class 'tuple'>
    <class 'tuple'>
    Traceback (most recent call last):
    File "d:/testALg/homework/tuple/27_1.py", line 15, in <module>
        print(a - b)
    TypeError: unsupported operand type(s) for -: 'tuple' and 'tuple'

&emsp;&emsp;并不支持。

&emsp;&emsp;尽管tuple本身的元素不能改变，但是有些用不到的手段可以改变tuple元素中的元素。

    if __name__ == "__main__":
        t = (1,2,3,[4,5])
        print(t)
        print(t[-1])
        t[-1][0] = "Y"
        t[-1][1] = "X"
        print(t)

    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/tuple/27_2.py
    (1, 2, 3, [4, 5])
    [4, 5]
    (1, 2, 3, ['Y', 'X'])

&emsp;&emsp;tuple的最后一个元素是一个列表，尽管其他int类型元素不能改变，但是列表的中元素能够改变——包括最后一个列表也不能改变类型。

    s = set()
    s.add('C')
    s.add('B')

    t[-1] = s
    print(t)

    Traceback (most recent call last):
    File "d:/testALg/homework/tuple/27_2.py", line 22, in <module>
    t[-1] = s
    TypeError: 'tuple' object does not support item assignment

&emsp;&emsp;但是支持列表的append一类函数。

    t[-1].append(6)
    print(t)

    (1, 2, 3, ['Y', 'X', 6])

&emsp;&emsp;不支持直接赋值，但是允许变相修改，真是个傲娇的类型。

&emsp;&emsp;不过这种方法通常没啥用，tuple还是要保证其不能随意修改，才能保证程序比较稳健。

