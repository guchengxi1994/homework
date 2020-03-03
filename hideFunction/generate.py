import pickle
# from operator import methodcaller



# path = "D:\\testALg\\homework\\hideFunction\\map.txt"

# class HideMap(object):
#     def _load(self,path):
#         x = [line.split(' ')[:-1] for line in open(path).readlines()]
#         print(x)


# if __name__ == "__main__":
#     hm = HideMap()
#     hm._load(path)

dic = {'loadNN':'cv2.dnn.readNetFromDarknet'}
p = open('D:\\testALg\\homework\\hideFunction\\params.pickle', 'wb')
pickle.dump(dic,p)
p.close()


pp = open('D:\\testALg\\homework\\hideFunction\\params.pickle','rb')
ppd = pickle.load(pp)
print(ppd)

# print(dic['loadNN'])

