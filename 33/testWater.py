import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,color,data,filters
import cv2
import numpy as np
 
# image =color.rgb2gray(data.camera())
ori_image =cv2.imread('D:\\DefectDemo\\DEMO2\\defect\\bResult\\1149-B-55-0-0000_A2.jpg')
image = cv2.cvtColor(ori_image,cv2.COLOR_BGR2GRAY)
denoised = filters.rank.median(image, morphology.disk(2)) #过滤噪声
 
#将梯度值低于10的作为开始标记点
markers = filters.rank.gradient(denoised, morphology.disk(5)) <10
markers = ndi.label(markers)[0]
 
gradient = filters.rank.gradient(denoised, morphology.disk(2)) #计算梯度
labels =morphology.watershed(gradient, markers, mask=image) #基于梯度的分水岭算法

# labels = np.array(labels/np.max(labels)*255,dtype = np.uint8)
 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

# cv2.imshow('seg',labels)
# cv2.waitKey()
 
ax0.imshow(image,  interpolation='nearest')
ax0.set_title("Original")
ax1.imshow(gradient,  interpolation='nearest')
ax1.set_title("Gradient")
ax2.imshow(markers,  interpolation='nearest')
ax2.set_title("Markers")
ax3.imshow(labels,  interpolation='nearest')
ax3.set_title("Segmented")
 
for ax in axes:
    ax.axis('off')
 
fig.tight_layout()
plt.show()
