import cv2
import numpy as np
import scipy.signal


def dpi2mm(distance,scale=300):

    return distance*25.4/scale

def len2r(distance,PI=3.14):

    return distance*3/3.14/2



def findpeaks(data, spacing=1, limit=None):
    """Finds peaks in `data` which are of `spacing` width and >=`limit`.
    :param data: values
    :param spacing: minimum spacing to the next peak (should be 1 or more)
    :param limit: peaks should have value greater or equal
    :return:
    """
    len = data.size
    x = np.zeros(len+2*spacing)
    x[:spacing] = data[0]-1.e-6
    x[-spacing:] = data[-1]-1.e-6
    x[spacing:spacing+len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start : start + len]  # before
        start = spacing
        h_c = x[start : start + len]  # central
        start = spacing + s + 1
        h_a = x[start : start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind




i = cv2.imread("D:\\homework\\homework\\house\\weld333.jpg")
gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

h,w = gray.shape

ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel  = np.ones((3,99),dtype = np.float32)

# kernel[1,:] = 1

kernel = kernel/np.sum(kernel)

# print(kernel)
dst = cv2.filter2D(gray,-1,kernel)

dst = cv2.filter2D(dst,-1,kernel)
dst = cv2.filter2D(dst,-1,kernel)

dst = np.array(dst,dtype = np.uint8)





# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)


cv2.imwrite('D:\\homework\\homework\\house\\dst.jpg', dst)

_,th3 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('D:\\homework\\homework\\house\\ret.jpg', th2)

cv2.imwrite('D:\\homework\\homework\\house\\ret2.jpg', th3)

x = th3.sum(axis=1)/255

x[x<=np.mean(x)] = 1

print(x.shape)

# print(x)


# indexes = findpeaks(x,spacing = 1)
# print(indexes)

width = int(0.1*h)

indexes = scipy.signal.find_peaks_cwt(x,np.arange(1, width))
indexes = np.array(indexes) - 1

# print(indexes.shape)
print(indexes)


indList = []
for ind in indexes:
    # print(x[ind])
    if x[ind]>0.5*w:
        indList.append(ind)

print(indList)

if len(indList)<2:
    pass
else:
    print(dpi2mm(max(indList)-min(indList)))

    print(len2r(dpi2mm(max(indList)-min(indList))))


# lines = cv2.HoughLines(th3, 1, np.pi / 180, 300)


for ind in indList:
    x0 = ind
    y0 = 0
    x1 = ind 
    y1 = w
    cv2.line(i,(y0, x0), (y1, x1), (0, 0, 255), 2)

# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
#     cv2.line(i, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite('D:\\homework\\homework\\house\\houghlines3.jpg', i)