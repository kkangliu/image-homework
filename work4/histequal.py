import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__=='__main__':
    f = cv2.imread('einstein.tif',0)
    imhist, bins = np.histogram(f.flatten(),256, normed=True)   #imhist是概率值,normed=True归一化
    _imhist = imhist.cumsum()   #累加
    _imhist = 255*_imhist / _imhist[-1]

    _ff = np.interp(f.ravel(), bins[:-1], _imhist)
    _ff = _ff.reshape(f.shape).astype('uint8')
    #img = exposure.equalize_hist(f)

    fig = plt.figure(figsize=(8,8))  # figure 的大小
    ax1 = fig.add_subplot(221)
    plt.title('initial_image')
    plt.imshow(f, cmap='gray')

    ax2 = fig.add_subplot(222)
    plt.title('later_image')
    plt.imshow(_ff, cmap='gray')

    ax3 = fig.add_subplot(223)
    plt.title('initial_histeq')
    plt.hist(f.ravel(), 256, [0, 256])  #ravel是把多维降维一维

    ax4 = fig.add_subplot(224)
    plt.title('later_histeq')
    plt.hist(_ff.ravel(), 256, [0, 256])
    plt.show()






