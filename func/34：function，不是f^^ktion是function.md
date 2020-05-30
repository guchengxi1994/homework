<!--
 * @lanhuage: python
 * @Descripttion: 
 * @version: beta
 * @Author: xiaoshuyui
 * @Date: 2020-04-08 08:13:00
 * @LastEditors: xiaoshuyui
 * @LastEditTime: 2020-04-08 08:59:17
 -->
&emsp;&emsp;function在鸟语中的含义是“功能”，也可以被叫做“函数”。之前也讲过一些，不过只是皮毛。

&emsp;&emsp;f**ktion的意思......不知道，我胡诌的。不过从鸟语构词法来看没有任何问题。

&emsp;&emsp;function的作用，在python或者任何的编程语言中都差不多，就是把会重复利用到的一句或者一段代码封装起来，给个别名，有输入（可有可无）跟输出（也是可有可无）：

    def func1():
        print("today is Wednesday！")

    if __name__ == "__main__":
        func1()

    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/func/34.py
    today is Wednesday！

&emsp;&emsp;就是在函数外边把这个函数的名字喊出来，它就会出来展示一下自己的功能。

&emsp;&emsp;当然这只是很简单的用法。

    def plus(a:float,b:float):
        return a + b

    if __name__ == "__main__":
        print(plus(1.6,2.8))

    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/func/34_1.py
    4.4

&emsp;&emsp;或者更加高深一点：

    def com_plus(a:complex,b:complex):
        return a + b

    if __name__ == "__main__":
        a = complex(1,5)
        b = complex(10,2.6)

        print(a)
        print(b)

        print(com_plus(a,b))

    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/func/34_2.py
    (1+5j)
    (10+2.6j)
    (11+7.6j)

&emsp;&emsp;这里其实不用函数也能获得结果，但是为什么需要函数呢？可以考虑这样的场景——如果我不知道什么时候需要这样的功能，同时输入的值不确定，比方说是n，我要求从1到n的累加：

    def mySum(n:int):
        s = 0
        for i in range(1,n+1):
            s = s + i 
        return s

    if __name__ == "__main__":   
        print(mySum(10))

    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/func/34_3.py
    55

&emsp;&emsp;这是函数的写法，如果我没有把这段代码封装，那么实际运作过程中保不齐什么时候就会有变量名冲突或者其它这样那样的问题；如果用流程控制语句if...else...，就是程序可能执行可能不执行，但是什么时候执行的问题在哪执行的问题依旧没有解决，因为代码写在哪它就得在哪待着，没办法像函数这样方便。

&emsp;&emsp;当然，这只是通俗易懂的解释方法，函数的好处肯定不止这样，它至少能把主函数变得很精简。

&emsp;&emsp;这里要稍微讲一点点超纲的话，在python里边，“+”，“-”这两个运算符，也都是函数，不过是类函数，不像这里自己定义的那样，随便定义在哪里一样。尤其是“-”法会很有用，“+”法用的比较少。



