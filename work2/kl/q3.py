import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#2.1图像二维卷积函数
def  twodConv(f,w ,method='zero'):
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
    f = cv2.imread('cameraman.tif',0)
    w = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]))   #卷积核
    g1 = twodConv(f,w,method='replicate')
    g2 = twodConv(f,w)

    print(g1.shape)
    print(g2.shape)
    print(f.shape)
    g1 = Image.fromarray(g1)
    g2 = Image.fromarray(g2)


    fig = plt.figure(figsize=(12, 4))   # figure 的大小
    ax1 = fig.add_subplot(131)
    plt.axis('off')     #不显示坐标轴
    plt.title('cameraman')
    ax1.imshow(f, cmap='gray')   #设置显示方式为灰度

    ax1 = fig.add_subplot(132)
    plt.axis('off')
    plt.title('replicate')
    ax1.imshow(g1, cmap='gray')

    ax1 = fig.add_subplot(133)
    plt.axis('off')
    plt.title('zero')
    ax1.imshow(g2, cmap='gray')
    plt.show()

