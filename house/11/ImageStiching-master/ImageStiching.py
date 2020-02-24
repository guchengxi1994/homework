from Stitcher import Stitcher
import cv2

# 读取拼接图片
imageA = cv2.imread("D:\\homework\\homework\\house\\11\\ImageStiching-master\\5-6.jpg")
imageB = cv2.imread("D:\\homework\\homework\\house\\11\\ImageStiching-master\\6-1.jpg")

# imageAShape = imageA.shape
# x1,y1 = imageAShape[0],imageAShape[1]
# imageB = cv2.resize(imageB,(x1,y1))

# print(imageB.shape)

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# 显示所有图片
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()