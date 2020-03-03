import pickle
import cv2
# from operator import methodcaller

pp = open('D:\\testALg\\homework\\hideFunction\\params.pickle','rb')
ppd = pickle.load(pp)
print(ppd)

imread = "cv2.imread"
imshow = "cv2.imshow"

x = eval(imread)('D:\\DefectDemo\\\defectImages\\5LDN200-00011.jpg')

eval(imshow)('test',x)
# # x = methodcaller(imread)
# print(type(x))
# # imshow(x)
# xx = x('D:\\DefectDemo\\\defectImages\\5LDN200-00011.jpg')
# r = methodcaller(imshow)
# r(xx)
cv2.waitKey(0)
