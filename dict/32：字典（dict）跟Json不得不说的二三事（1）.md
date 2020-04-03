<!--
 * @lanhuage: python
 * @Descripttion: 
 * @version: beta
 * @Author: xiaoshuyui
 * @Date: 2020-04-03 08:26:22
 * @LastEditors: xiaoshuyui
 * @LastEditTime: 2020-04-03 09:19:38
 -->
&emsp;&emsp;这次讲讲json跟python字典不得不说的那些事情。

        JSON(JavaScript Object Notation, JS 对象简谱) 是一种轻量级的数据交换格式。它基于 ECMAScript (欧洲计算机协会制定的js规范)的一个子集，采用完全独立于编程语言的文本格式来存储和表示数据。简洁和清晰的层次结构使得 JSON 成为理想的数据交换语言。 易于人阅读和编写，同时也易于机器解析和生成，并有效地提升网络传输效率。

&emsp;&emsp;以上是百度百科的定义，通俗点讲，json就是传输数据的一种结构化规范，主要是用在网络通信传输过程中。这种规范化的样式跟dict很像，所以python中是支持两者的转化：

    import json

    if __name__ == "__main__":
        dic = {"a":1,"b":2222}

        js = json.dumps(dic)
        print(dic)
        print(js)

    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/dict/32.py
    {'a': 1, 'b': 2222}
    {"a": 1, "b": 2222}
    <class 'str'>

&emsp;&emsp;乍一看，好像这个规范只是把单引号转化成了双引号，同时把dict对象转化成了str对象。但是，因为网络通信过程中，是不能够传输一个python内置的dict对象的，必须要传输字符串对象，也就是说必须要是str类型。事实上因为json是javascript中使用的，所以其在javascript中也可以被当成对象解析，通俗易懂地讲就是跟python中字典解析（增删改查，主要是查）差不多，而如果是字符串的话，那么解析起来就很麻烦。所以一般开发过程中如果没有dict跟json这一层关系，会拼一个json的字符串进行信息传输。

&emsp;&emsp;举个可能超纲的例子：

    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bytes):
                return str(obj, encoding='utf-8')
            elif isinstance(obj,MyClass):
                return obj.__dict__()
            return json.JSONEncoder.default(self, obj)

    class MyClass(object):
        def __init__(self,name:str,age:int):
            self.name = name 
            self.age = age 

        def __dict__(self):
            dic = {}
            # dic['姓名'] = self.name
            # dic['年龄'] = self.age
            dic['name'] = self.name
            dic['age'] = str(self.age)

            return dic


    if __name__ == "__main__":
        myclass = MyClass('xiaohua',18)
        js = myclass.__dict__()
        print(js)

        dic = {'class1':myclass}
        js2 = json.dumps(dic,cls=MyEncoder,indent=4)
        print(js2)


    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/dict/32_2.py
    {'name': 'xiaohua', 'age': '18'}
    {
        "class1": {
            "name": "xiaohua",
            "age": "18"
        }
    }

