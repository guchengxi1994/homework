import cv2
import numpy as np
 
 
def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
def detectAndCompute(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, features) = sift.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])  # 得到的点需要进一步转换才能使用
    return (kps, features)
 
 
def matchKeyPoints(kpsA, kpsB, featuresA, featuresB, ratio = 0.75, reprojThresh = 4.0):
    #  ratio是最近邻匹配的推荐阈值
    #  reprojThresh是随机取样一致性的推荐阈值
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            matches.append((m[0].queryIdx, m[0].trainIdx))
    kpsA = np.float32([kpsA[m[0]] for m in matches])  # 使用np.float32转化列表
    kpsB = np.float32([kpsB[m[1]] for m in matches])
    (M, status) = cv2.findHomography(kpsA, kpsB, cv2.RANSAC, reprojThresh)
    return (M, matches, status)  # 并不是所有的点都有匹配解，它们的状态存在status中
 
 
def stich(imgA, imgB, M):
    imgA = np.array(imgA,dtype= np.float32)
    imgB = np.array(imgB,dtype= np.float32)
    result = cv2.warpPerspective(imgA, M, (int(imgA.shape[1] + imgB.shape[1]), int(imgA.shape[0])))
    result[0:imageA.shape[0], 0:imageB.shape[1]] = imageB
    # cv_show('result', result)
    cv2.imwrite('D:\\homework\\homework\\house\\11\\result.jpg', result)
 
 
def drawMatches(imgA, imgB, kpsA, kpsB, matches, status):
    (hA, wA) = imgA.shape[0:2]
    (hB, wB) = imgB.shape[0:2]
    # 注意这里的3通道和uint8类型
    drawImg = np.zeros((max(hA, hB), wA + wB, 3), 'uint8')
    drawImg[0:hB, 0:wB] = imageB
    drawImg[0:hA, wB:] = imageA
    for ((queryIdx, trainIdx),s) in zip(matches, status):
        if s == 1:
            # 注意将float32 --> int
            pt1 = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]))
            pt2 = (int(kpsA[trainIdx][0]) + wB, int(kpsA[trainIdx][1]))
            cv2.line(drawImg, pt1, pt2, (0, 0, 255))
    # cv_show("drawImg", drawImg)
    cv2.imwrite('D:\\homework\\homework\\house\\11\\asasa.jpg',drawImg)
 

def preFix(img1,img2):
    ishape1 = img1.shape
    ishape2 = img2.shape
     
    x1 ,y1 = ishape1[0],ishape1[1]
    x2 ,y2 = ishape2[0],ishape2[1]


    fixX = min(x1,x2)
    fixY = min(y1,y2)

    img1 = img1[0:fixX,0:fixY]
    img2 = img2[0:fixX,0:fixY]

    return img1,img2



# 读取图像
imageA = cv2.imread('D:\\homework\\homework\\house\\11\\6-1.jpg')
# cv_show("imageA", imageA)
imageB = cv2.imread('D:\\homework\\homework\\house\\11\\5-6.jpg')
# cv_show("imageB", imageB)

imageA,imageB = preFix(imageA,imageB)

# 计算SIFT特征点和特征向量
(kpsA, featuresA) = detectAndCompute(imageA)
(kpsB, featuresB) = detectAndCompute(imageB)
# 基于最近邻和随机取样一致性得到一个单应性矩阵
(M, matches, status) = matchKeyPoints(kpsA, kpsB, featuresA, featuresB)
# 绘制匹配结果
drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
# 拼接
stich(imageA, imageB, M)

