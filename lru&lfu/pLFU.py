class LFUObject(object):
    def __init__(self,key,value,insertTime,times):
        self.key = key
        self.value = value
        self.insertTime = insertTime
        self.times = times 

    def __eq__(self,other):
        return self.key == other.key

    def __str__(self):
        return "key:"+str(self.key) +" and "+"value:"+str(self.value)+" and "+"times:"+str(self.times)

def iter2Obj(x):
    l = []
    for i in x:
        l.append(i)
    
    return l[0]


class LFU(list):
    def __init__(self,defaultSize=5):
        self.defaultSize = defaultSize
        self.cache = list()

    def _sort_(self):
        self.cache.sort(key=lambda x:(x.insertTime,x.times),reverse=True)
    
    def set(self,obj:LFUObject):

        if obj not in self.cache:
            if len(self.cache) == self.defaultSize:
                self._sort_()
                self.cache.pop()
            self.cache.append(obj)
            self._sort_()
        else:
            print("warning:存在索引相同的对象，将替换原对象")
            x = filter(lambda y:y.key == obj.key,self.cache)
            o = iter2Obj(x)

            self.cache.remove(o)
            self.cache.append(obj)
            self._sort_()


    def get(self,key):
        x = filter(lambda y:y.key == key,self.cache)

        if x is None:
            return None 
        else:
            o = iter2Obj(x)
            self.cache.remove(o)
            o.times = o.times + 1
            self.cache.append(o)
            self._sort_()
            return o.value

if __name__ == "__main__":
    import datetime,random
    test = LFU()
    for i in range(0,5):
        x = LFUObject(i,"aa"+str(i),datetime.datetime.now(),random.randint(0,10))

        test.set(x)
    
    # print(test)
    for i in test.cache:
        print(i)

    test.get(0)

    print("==================")
    
    for i in test.cache:
        print(i)

    
    x = LFUObject(0,"aa"+str(0),datetime.datetime.now(),15)

    print("==================")

    test.set(x)

    

    for i in test.cache:
        print(i)

    
    x = LFUObject(20,"aa"+str(20),datetime.datetime.now(),20)

    print("==================")

    test.set(x)

    for i in test.cache:
        print(i)
    






