<!--
 * @lanhuage: python
 * @Descripttion: 
 * @version: beta
 * @Author: xiaoshuyui
 * @Date: 2020-04-07 08:20:30
 * @LastEditors: xiaoshuyui
 * @LastEditTime: 2020-04-07 08:48:10
 -->
&emsp;&emsp;上回提到，Json乃一游离于前端以及后端两面三刀者，且与dict同源，两者沆瀣一气，狼狈为奸之事，昭然若揭。

&emsp;&emsp;上回说的是把一个python dict类型的对象，转化成标准的Json格式供前端读取。除了一般的python对象，比如str类型，数字类型等，还超纲地使用了自定义类对象，使用json包下，json.dumps（这里提一句，一般不会用json.dump这个方法，因为这个方法是直接作用到文件的，也就是把json的字符串写入文件）方法照样可以正常读取内容。

&emsp;&emsp;这次来说说怎么把json字符串变回一个python可读取的dict对象。

    import json

    if __name__ == "__main__":
        dic = {'name':"xiaoshuyui",'sex':"male", \
            'age':18,'introduction':"handsome very"}
        
        j = json.dumps(dic)
        print(j)

        dic2 = json.loads(j)
        print(dic2)
        print(type(dic2))

    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/dict/33.py
    {"name": "xiaoshuyui", "sex": "male", "age": 18, "introduction": "handsome very"}
    PS C:\Users\admin> & D:/anaconda/python.exe d:/testALg/homework/dict/33.py
    {"name": "xiaoshuyui", "sex": "male", "age": 18, "introduction": "handsome very"}
    {'name': 'xiaoshuyui', 'sex': 'male', 'age': 18, 'introduction': 'handsome very'}
    <class 'dict'>