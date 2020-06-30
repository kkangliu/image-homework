import numpy as np
import cv2
import matplotlib.pyplot as plt

# 二值化
def _im2bw(f, thre):
    img = np.zeros_like(f)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if f[i][j] < thre:
                img[i][j] = 1
            else:
                img[i][j] = 0
    return img

# 腐蚀操作
def corro(h, k):
    _img = cv2.filter2D(h, -1, k)
    img = np.zeros_like(_img)
    img[np.where(_img == np.sum(k))] = 1
    return img

# 膨胀操作
def swell(h,k):
    _img = cv2.filter2D(h, -1, k)
    img = np.ones_like(_img)
    img[np.where(_img == 0)] = 0
    return img

# 击中击不中
def jizhong(FF, a, b):
    _F = 1 - FF
    m1 = corro(FF, a)   #a对二值图腐蚀
    m2 = corro(_F, b)   #b对二值图的补集腐蚀
    l = m1 + m2
    for i in range(0, l.shape[0]):
        for j in range(0, l.shape[1]):
            if l[i, j] != 2:
                l[i, j] = 0
            else:
                l[i, j] = 1
    g = FF-l
    return g

# 细化
def xihua(h):
    gg = h.copy()
    a = np.array([[[0, 0, 0], [0, 1, 0], [1, 1, 1]],
                 [[1, 0, 0], [1, 1, 0], [1, 0, 0]],
                 [[1, 1, 1], [0, 1, 0], [0, 0, 0]],
                 [[0, 0, 1], [0, 1, 1], [0, 0, 1]],
                 [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
                 [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                 [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
                 [[0, 0, 0], [0, 1, 1], [0, 1, 1]]])
    b = np.array([[[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                 [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                 [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
                 [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                 [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 1], [0, 1, 1]],
                 [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                 [[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
    last = gg.copy()
    while(1):
        for i in range(0, 8):
            print(i)
            gg = jizhong(gg, a[i], b[i])
            if ((last==gg).all()):
                return gg
            last = gg.copy()
    return gg

#边界提取
def border(f):
    k = np.ones((3, 3), np.uint8)
    result = f.copy()
    result -= corro(result, k)
    return result

# 距离变换
def distance(ff, FF):
    _pad = np.pad(ff, ((1, 1), (1, 1)), 'constant', constant_values=0)
    for i in range(_pad.shape[0]):
        for j in range(_pad.shape[1]):
            if not _pad[i][j]:
                _pad[i][j] = 100
    result = np.ones_like(ff) * 100
    for i in range(1, _pad.shape[0] - 1):
        for j in range(1, _pad.shape[1] - 1):
            if FF[i-1, j-1] == 0:
                result[i - 1, j - 1] = ff[i - 1, j - 1]
                continue
            temp0 = _pad[i, j]
            temp1 = _pad[i - 1, j - 1] + 4
            temp2 = _pad[i - 1, j] + 3
            temp3 = _pad[i - 1, j + 1] + 4
            temp4 = _pad[i, j - 1] + 3
            result[i - 1, j - 1] = min(temp0, temp1, temp2, temp3, temp4)
    pad = np.pad(result, ((1, 1), (1, 1)), 'constant', constant_values=0)
    for i in range(pad.shape[0] - 2, 0, -1):
        for j in range(pad.shape[1] - 2, 0, -1):
            if FF[i-1, j-1] == 0:
                result[i - 1, j - 1] = ff[i - 1, j - 1]
                continue
            temp0 = pad[i, j]
            temp1 = pad[i, j + 1] + 3
            temp2 = pad[i + 1, j - 1] + 4
            temp3 = pad[i + 1, j] + 3
            temp4 = pad[i + 1, j + 1] + 4
            result[i - 1, j - 1] = min(temp0, temp1, temp2, temp3, temp4)
    result = np.round(result/3).astype('uint8')
    return result

# 裁剪
def cut(p):
    b1 = np.array([[[0, 0, 0], [1, 1, 0], [0, 0, 0]],
                 [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
                 [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
                 [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                 [[0, 0, 1], [0, 1, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                 [[0, 0, 0], [0, 1, 0], [0, 0, 1]]])
    b2 = np.array([[[1, 1, 1], [0, 0, 1], [1, 1, 1]],
                 [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
                 [[1, 1, 1], [1, 0, 0], [1, 1, 1]],
                 [[1, 1, 1], [1, 0, 1], [1, 0, 1]],
                 [[0, 1, 1], [1, 0, 1], [1, 1, 1]],
                 [[1, 1, 0], [1, 0, 1], [1, 1, 1]],
                 [[1, 1, 1], [1, 0, 1], [0, 1, 1]],
                 [[1, 1, 1], [1, 0, 1], [1, 1, 0]]])
    H = np.array([[1,1,1],[1,1,1],[1,1,1]])
    for ii in range(3):
        for i in range(0, 8):
            p = jizhong(p, b1[i], b2[i])   # xihua
    x2 = jizhong(p,b1[0],b2[0])+jizhong(p,b1[1],b2[1])+jizhong(p,b1[2],b2[2])+jizhong(p,b1[3],b2[3])+jizhong(p,b1[4],b2[4])+jizhong(p,b1[5],b2[5])+jizhong(p,b1[6],b2[6])+jizhong(p,b1[7],b2[7])
    x3 = x2
    for ii in range(3):
        x3 = swell(x3,H)   # swell
        l = p + x3
        for i in range(0, l.shape[0]):
             for j in range(0, l.shape[1]):
                 if l[i, j] != 2:
                     l[i, j] = 0
                 else:
                     l[i, j] = 1
        x3 = l
    x4 = x3+p
    return x4

if __name__=='__main__':
    f = cv2.imread('zhiwen.png', 0)
    # 127为初始估计值T;阈值法对图像进行二值化
    F = _im2bw(f, 127)   # binary_image
    f_xihua = xihua(F)    # xihua
    imgborder = border(F)
    f_dis = distance(imgborder,F)   # distance
    f_cut_xihua = cut(f_xihua)    # cut_xihua
    f_cut_dis = cut(f_dis)     # cut_distance

    fig = plt.figure(figsize=(12, 8))  # figure 的大小
    ax1 = fig.add_subplot(231)
    plt.axis('off')
    plt.title('initial_image')
    plt.imshow(f, cmap='gray')
    ax2 = fig.add_subplot(232)
    plt.axis('off')
    plt.title('binary_image')
    plt.imshow(F, cmap='gray')
    ax3 = fig.add_subplot(233)
    plt.axis('off')
    plt.title('xihua_image')
    plt.imshow(f_xihua, cmap='gray')
    ax4 = fig.add_subplot(234)
    plt.axis('off')
    plt.title('distance_image')
    plt.imshow(f_dis, cmap='gray')
    ax5 = fig.add_subplot(235)
    plt.axis('off')
    plt.title('cut_xihua_image')
    plt.imshow(f_cut_xihua, cmap='gray')
    ax6 = fig.add_subplot(236)
    plt.axis('off')
    plt.title('cut_distance_image')
    plt.imshow(f_cut_dis, cmap='gray')
    plt.show()
