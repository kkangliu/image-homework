import numpy as np
import cv2
import matplotlib.pyplot as plt


def threshold(f,T):
    g1 = []
    g2 = []
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if f[i][j]<T:
                g1.append(f[i][j])   #G1由灰度值大于T的所有像素组成
            else:
                g2.append(f[i][j])   #G2由所有小于T的像素组成；
    g1 = np.array(g1)
    g2 = np.array(g2)
    m1 = g1.sum()/len(g1)
    m2 = g2.sum()/len(g2)
    T0 = int((m1+m2)/2)   #新的阈值
    return T0

def im2bw(f,T):
    img=np.zeros((f.shape[0],f.shape[1]),np.uint8)
    T0=T
    T1=threshold(f,T0)
    for k in range (50):   # 迭代次数
        if abs(T1-T0)==0:   # 新阈值减旧阈值差值为零，则为二值图最佳阈值
            for i in range (f.shape[0]):
                for j in range (f.shape[1]):
                    if f[i,j]>T1:
                        img[i,j]=255
                    else:
                        img[i,j]=0
            break
        else:
            T2=threshold(f,T1)
            T0=T1
            T1=T2
    return img


def corro(h,c):
     image = np.zeros([h.shape[0], h.shape[1]])
     d = sum(c!=0)
     d = d.sum()
     f_new = np.pad(h, ([1, 1], [1, 1]), mode='constant', constant_values=0)  # 补0
     for i in range(1, h.shape[0] + 1):
         for j in range(1, h.shape[1] + 1):
             image[i - 1, j - 1] = (f_new[i - 1: i + 2, j - 1: j + 2] * c).sum() / d
             if image[i - 1, j - 1] != 255:
                 image[i - 1, j - 1] = 0
             else:
                 image[i - 1, j - 1] = 255
    return image

def jizhong(FF,a,b):
    _F = 255-FF
    m1 = corro(FF,a)   #a对二值图腐蚀
    m2 = corro(_F,b)   #b对二值图的补集腐蚀
    l = m1+m2
    for i in range(0, l.shape[0]):
        for j in range(0, l.shape[1]):
            if l[i,j]!=510:
                l[i, j] = 0
            else:
                l[i, j] = 255
    g = FF-l
    return g

def xihua(F):
    gg = F
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
    for ii in range(0,4):
         for i in range(0,8):
             gg = jizhong(gg, a[i], b[i])
    return gg
'''
             if hh.all()==gg.all():
                 print(q)
                 return hh
             else:
                 q+=1
                 gg=np.repeat(hh)
'''



def distance(h):
    b = np.array([[1,1,1],[1,1,1],[1,1,1]])
    m = h-corro(h,b)   #提取边界

if __name__=='__main__':
    f = cv2.imread('gujiatiqu.png',0)
    F = im2bw(f,127)   #127为初始估计值T;阈值法对图像进行二值化
    g = xihua(F)

    fig = plt.figure(figsize=(8, 4))  # figure 的大小
    ax1 = fig.add_subplot(121)
    plt.axis('off')
    plt.imshow(F, cmap='gray')
    ax2 = fig.add_subplot(122)
    plt.axis('off')
    plt.imshow(g, cmap='gray')
    plt.show()

'''

    ax2 = fig.add_subplot(132)
    plt.axis('off')
    plt.title('zero')
    ax2.imshow(g1, cmap='gray')

    ax3 = fig.add_subplot(133)
    plt.axis('off')
    plt.title('edge')
    ax3.imshow(g2, cmap='gray')
    plt.show()
'''