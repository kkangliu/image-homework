import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import preprocessing


def dft2D(f):
    #二维傅里叶
    F = np.zeros([f.shape[0], f.shape[1]], dtype=complex)
    for i in range(f.shape[0]):
        F[i,:] = np.fft.fft(f[i,:])
    for i in range(f.shape[1]):
        F[:,i] = np.fft.fft(F[:,i])
    return F


if __name__=='__main__':
    f = cv2.imread('rose512.tif',0)
    f = f/255
    F = dft2D(f)
    F2 = np.abs(F)
    #F = np.log(np.abs(F)) #将图像转换成幅值谱,取log使数据落在0-255之间
    #FF = np.fft.fftshift(FF)  #中心化

    M = F.shape[0]
    N = F.shape[1]
    FF = np.mat(F).H.T  #取共轭
    f1 = dft2D(FF)
    f2 = f1/(M*N)
    ff = np.mat(f2).H.T
    g = np.abs(ff)
    d = f-g
    d = d.astype(int)



    fig = plt.figure(figsize=(12, 3))  # figure 的大小
    ax1 = fig.add_subplot(131)
    plt.axis('off')
    plt.title('question1:F(FFT)')
    plt.imshow(F2,cmap='gray')

    ax1 = fig.add_subplot(132)
    plt.axis('off')
    plt.title('question2:g(IFFT)')
    plt.imshow(g,cmap='gray')

    ax1 = fig.add_subplot(133)
    plt.axis('off')
    plt.title('question3:d = f-g')
    plt.imshow(d, cmap='gray')
    plt.show()

'''
    plt.imshow(F2, cmap='gray')
    # cv2.imwrite('FFT.png',F2)
    plt.imshow(f_new - f, cmap='gray')
'''


