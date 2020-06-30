import numpy as np
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt


#图像二维卷积函数
def  twodConv(f,sig,method='zero'):
    m = math.ceil(3 * sig) * 2 + 1
    center = m//2
    gausskernel = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            x = i - center
            y = j - center
            norm = math.pow(x , 2) + pow(y, 2)
            gausskernel[i, j] = (1/2*np.pi*sig*sig)*math.exp(-norm/(2*math.pow(sig,2)))  # 求高斯卷积
    sum = np.sum(gausskernel)  # 求和
    w = gausskernel / sum  # 归一化

    a = int((w.shape[0] - 1) / 2)
    f_new = np.zeros([f.shape[0], f.shape[1]])   #建立新的图像矩阵存储
    if method == 'replicate':
        constant = np.zeros([f.shape[0] + 2 * a, f.shape[1] + 2 * a])
        for i in range(0, f.shape[0] + 2*a ):
            for j in range(0, f.shape[1] + 2*a ):
            #for j in range(a, f.shape[1] + a):
                if i<a :
                    if j<a:
                        constant[i, j] = f[0, 0]

                    elif j>(a+f.shape[1]-1):
                        constant[i, j] = f[0, f.shape[1]-1]
                    else:
                        constant[i, j] = f[0,j-a]
                elif i > (f.shape[0] + a-1):
                    if j<a:
                        constant[i, j] = f[f.shape[0]-1, 0]
                    elif j>(a+f.shape[1]-1):
                        constant[i, j] = f[f.shape[0]-1, f.shape[1]-1]
                    else:
                        constant[i, j] = f[f.shape[0]-1, j-a]
                elif j<a:
                    constant[i, j] = f[i-a,0]
                elif j > (f.shape[1] + a-1):
                    constant[i, j] = f[i-a,f.shape[1]-1]
                else:
                    constant[i, j] = f[i-a, j-a]


    else:
        constant = np.zeros([f.shape[0]+2*a,f.shape[1]+2*a])
        for i in range(a, f.shape[0] + a ):
            for j in range(a, f.shape[1] + a ):
                constant[i,j] = f[i-a,j-a]


    for x in range(0,f_new.shape[0]):
        for y in range(0,f_new.shape[1]):
            # print(constant[x:x+w.shape[0], y:y+w.shape[1]])
            dataV = ((constant[x:x+w.shape[0], y:y+w.shape[1]])* w).sum()
            #防止出现灰度值<0和>255的情况
            if dataV < 0:
                dataV = 0
            if dataV > 255:
                dataV = 255
            f_new[x , y ] = dataV
    return f_new



if __name__=='__main__':
    f1 = cv2.imread('cameraman.tif', 0)
    f2 = cv2.imread('einstein.tif', 0)
    f3 = cv2.imread('lena512color_NTSC.tif', 0)
    f4 = cv2.imread('mandril_color_NTSC.tif', 0)

    g1 = twodConv(f1, sig=float(1.0),method = 'replicate')
    g2 = twodConv(f2, sig=float(2.0),method = 'replicate')
    g3 = twodConv(f3, sig=float(3.0),method = 'replicate')
    g4 = twodConv(f4, sig=float(5.0),method = 'replicate')
    g5 = cv2.GaussianBlur(f1,(7,7),1)
    g6 = twodConv(f4, sig=float(5.0))
    g7 = twodConv(f1, sig=float(2.0), method='replicate')
    g8 = twodConv(f1, sig=float(3.0), method='replicate')
    g9 = twodConv(f1, sig=float(5.0), method='replicate')
    g10 = g1 - g5
    g11 = g6 - g4



    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(141)
    plt.axis('off')
    plt.title('sig=1')
    ax1.imshow(g1, cmap='gray')


    ax1 = fig.add_subplot(142)
    plt.axis('off')
    plt.title('sig=2')
    ax1.imshow(g2, cmap='gray')

    ax1 = fig.add_subplot(143)
    plt.axis('off')
    plt.title('sig=3')
    ax1.imshow(g3, cmap='gray')

    ax1 = fig.add_subplot(144)
    plt.axis('off')
    plt.title('sig=5')
    ax1.imshow(g4, cmap='gray')

#sig=1,2,3,5在一张图片下的对比
    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(141)
    plt.axis('off')
    plt.title('sig=1')
    ax1.imshow(g1, cmap='gray')

    ax1 = fig.add_subplot(142)
    plt.axis('off')
    plt.title('sig=2')
    ax1.imshow(g7, cmap='gray')

    ax1 = fig.add_subplot(143)
    plt.axis('off')
    plt.title('sig=3')
    ax1.imshow(g8, cmap='gray')

    ax1 = fig.add_subplot(144)
    plt.axis('off')
    plt.title('sig=5')
    ax1.imshow(g9, cmap='gray')
#sig=1下与直接调用相关函数的比较
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(131)
    plt.axis('off')
    plt.title('sig=1')
    ax1.imshow(g1, cmap='gray')
    ax1 = fig.add_subplot(132)
    plt.axis('off')
    plt.title('filter')
    ax1.imshow(g5, cmap='gray')
    ax1 = fig.add_subplot(133)
    plt.axis('off')
    plt.title('comparison')
    ax1.imshow(g10, cmap='gray')
#einstein像素复制和补零下滤波结果差异
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(131)
    plt.axis('off')
    plt.title('replicate')
    ax1.imshow(g4, cmap='gray')
    ax1 = fig.add_subplot(132)
    plt.axis('off')
    plt.title('zero')
    ax1.imshow(g6, cmap='gray')
    ax1 = fig.add_subplot(133)
    plt.axis('off')
    plt.title('comparison')
    ax1.imshow(g11, cmap='gray')
    plt.show()
