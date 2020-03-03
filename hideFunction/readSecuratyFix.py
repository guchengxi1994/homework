import cv2
import ast 
import os


imread = "cv2.imread"
imshow = "cv2.imshow"
imshow2 = 'plt.show'
plu = 'OO.plus'

class OO(object):
    def plus(self,a,b):
        return a + b


# oo = OO()
# xx = x.plus(1,2)
# print(xx)

# x = OO.plus(OO,1,2)
# print(x)


x = eval(imread,{},{'cv2':cv2})('D:\\DefectDemo\\defectImages\\5LDN200-00011.jpg')
print(type(x))

# x = eval(plu,{'plu':oo},{'plu':oo})(1,2)
# print(x)
# eval()

# eval(imshow,{'cv2':os},{'cv2':os})('test',x)
eval(imshow,{},{'cv2':cv2})('test',x)

cv2.waitKey(0)


