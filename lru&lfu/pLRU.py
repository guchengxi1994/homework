from collections import OrderedDict

class LRU(OrderedDict):
    def __init__(self,defaultSize=5):
        self.defaultSize = defaultSize
        self.cache = OrderedDict()

    # def set(self,key,value):
    #     if key in self:
    #         del self[key]
    #     OrderedDict.
    # def __setitem__(self,key,value):
    #     if key in self:
    #         del self[key]
    #     OrderedDict.__setitem__(self,key,value)
    
    # def set()
    def set(self,key,value):
        if key in  self.cache:
            del self.cache[key]
            # self.cache.pop(key)
            self.cache[key] = value
        else:
            if len(self.cache) == self.defaultSize:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def get(self,key):
        if key in self.cache:
            val = self.cache[key]
            del self.cache[key]
            self.cache[key] = val 
            return val 
        else:
            return None
        











if __name__ == "__main__":
    test = LRU()

    for i in range(0,5):
        test.set(i,"aa"+str(i))

    print(test.cache)

    test.get(3)

    print(test.cache)

    test.set("tom","123")

    print(test.cache)


    # for i in test.cache:
    #     print(i)



        