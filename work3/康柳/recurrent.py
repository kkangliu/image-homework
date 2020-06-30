import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt


def center(f):
    #计算二维傅里叶之前，通过(-1)^(x+y)来中心化
    f1 = np.zeros([f.shape[0],f.shape[1]])
    for x in range(0,f.shape[0]):
        for y in range(0,f.shape[1]):
            f1[x,y] = f[x,y] * ((-1)**(x+y))
    f2 = dft2D(f1)
    return f2
def dft2D(f):
    #二维傅里叶
    F = np.zeros([f.shape[0], f.shape[1]], dtype=complex)
    for i in range(f.shape[0]):
        F[i,:] = np.fft.fft(f[i,:])
    for i in range(f.shape[1]):
        F[:,i] = np.fft.fft(F[:,i])
    return F


if __name__ == '__main__':
    f = np.zeros([512,512])
    for i in range(226,286):
        for j in range(251,261):
            f[i,j] = 1
    b = dft2D(f)
    b = np.abs(b)
    F = center(f)
    F = np.abs(F)
    S = np.log(1+np.abs(F))

    fig = plt.figure(figsize=(8,8))  # figure 的大小
    ax1 = fig.add_subplot(221)
    plt.axis('off')
    plt.title('a')
    plt.imshow(f,cmap='gray')

    ax1 = fig.add_subplot(222)
    plt.axis('off')
    plt.title('b')
    plt.imshow(b,cmap='gray')

    ax1 = fig.add_subplot(223)
    plt.axis('off')
    plt.title('c')
    plt.imshow(F,cmap='gray')

    ax1 = fig.add_subplot(224)
    plt.axis('off')
    plt.title('d')
    plt.imshow(S,cmap='gray')
    plt.show()
