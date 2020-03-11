
class Mylist:
    def __init__(self,*params):
        self.lis = list(params)

    def __sub__(self,other):
        if not isinstance(other,Mylist):
            return self.lis
        else:
            return list(set(self.lis).difference(set(other.lis)))

    def __add__(self,other):
        return self.lis + Mylist(other).lis


    def __str__(self):
        return str(self.lis)

    def append(self,pIn):
        if isinstance(pIn,list) :
            return self.lis + pIn
        elif  isinstance(pIn,Mylist):
            return list(set(self.lis + pIn.lis))
        else:
            return self.lis.append(pIn)

    def __hash__(self):
        # return hash(set(self.lis))
        s = 0
        for i in self.lis:           
            if isinstance(i,list) or isinstance(i,Mylist) or isinstance(i,tuple) \
                or isinstance(i,dict):
                pass 

            s = s+ hash(i)       
        return s

    
    def __eq__(self, other):
        if not isinstance(other,Mylist):
            return False
        else:
            return self.lis == other.lis









if __name__ == "__main__":

    lis1 = Mylist(1,2,3,4,5,6)
    print(lis1)
    lis2 = Mylist("a","b","v",1,2,6)
    print(lis1 + lis2)


    # lis = ['1','2','3',1,2,3]
    # lis2 = ['1']
    # print(lis-lis2)