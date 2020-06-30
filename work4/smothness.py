import numpy as np
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt


#为图像添加椒盐噪声，即对某些像素值随机赋值为0或255
def noise(f, proportion=0.05):
    noise_img = np.zeros([f.shape[0], f.shape[1]], dtype='uint8')
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            noise_img[i,j] = f[i,j]
    height,width =f.shape[0],f.shape[1]
    num = int(height*width*proportion)   #一共有多少个添加椒盐噪声的像素点
    for i in range(num):
        w = random.randint(0,width-1)
        h = random.randint(0,height-1)
        if random.randint(0,1) ==0:   #选择随机生成0or1
            noise_img[h,w] = 0
        else:
            noise_img[h,w] = 255
    return noise_img

def yanmo(ff):
    # f_new = np.zeros([f.shape[0]+2, f.shape[1]+2])
    # for i in range(f.shape[0]-1):
    #     for j in range(f.shape[1]-1):
    #         f_new[i+2,j+2] = f[i,j]   #手动补两圈0
    a = np.array([[[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]],
    [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0]],
    [[1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0]]])
    U = np.ones(9)
    m = np.ones(shape=9)
    image = np.zeros([ff.shape[0], ff.shape[1]])
    for i in range(2,ff.shape[0]-2):
        for j in range(2, ff.shape[1]-2):
            for k in range(0,9):
                c = []    #计算不包括0掩模的方差
                for ii in range(ff[i - 2: i + 3, j - 2: j + 3].shape[0]):
                    for jj in range(ff[i - 2: i + 3, j - 2: j + 3].shape[1]):
                        if a[k][ii][jj] != 0:
                            c.append(ff[i - 2: i + 3, j - 2: j + 3][ii][jj] * a[k][ii][jj])
                U[k] = np.var(c)
                m[k] = len(c)
            index = np.argmin(U)
            image[i, j] = (ff[i - 2: i + 3, j - 2: j + 3] * a[index]).sum() / m[index]
    image = image[2:-2, 2:-2]
    return image


if __name__=='__main__':
    f = cv2.imread('cameraman.tif',0)
    noise_img = noise(f)
    f_new1 = np.pad(noise_img, ([2, 2], [2, 2]), mode='constant', constant_values=0)  #补0
    f_new2 = np.pad(noise_img, ([2, 2], [2, 2]), 'edge')   #补边界值
    g1 = yanmo(f_new1)
    g2 = yanmo(f_new2)   #g1与g2进行对比

    fig = plt.figure(figsize=(6, 6))  # figure 的大小
    ax1 = fig.add_subplot(221)
    plt.axis('off')
    plt.title('initial_image')
    plt.imshow(f, cmap='gray')

    ax2 = fig.add_subplot(222)
    plt.axis('off')
    plt.title('noise_image')
    ax2.imshow(noise_img, cmap='gray')

    ax3 = fig.add_subplot(223)
    plt.axis('off')
    plt.title('zero')
    ax3.imshow(g1, cmap='gray')

    ax4 = fig.add_subplot(224)
    plt.axis('off')
    plt.title('edge')
    ax4.imshow(g2, cmap='gray')
    plt.show()