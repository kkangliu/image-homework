import numpy as np
import cv2
import matplotlib.pyplot as plt


def lap(ff):
    a = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    image = np.zeros([ff.shape[0], ff.shape[1]])
    for i in range(1, ff.shape[0] - 1):
        for j in range(1, ff.shape[1] - 1):
            w = (ff[i - 1:i + 2, j - 1:j + 2] * a).sum().astype('uint8')
            image[i - 1, j - 1] = w
            #print(image)
    return image


if __name__=='__main__':
    f = cv2.imread('lena512color.tiff',0)
    f_new1 = np.pad(f, ([1, 1], [1, 1]), mode='constant', constant_values=0)
    f_new2 = np.pad(f, ([1, 1], [1, 1]), 'edge')
    g1 = lap(f_new1)
    g2 = lap(f_new2)

    fig = plt.figure(figsize=(12, 4))  # figure 的大小
    ax1 = fig.add_subplot(131)
    plt.axis('off')
    plt.title('initial_image')
    plt.imshow(f, cmap='gray')

    ax2 = fig.add_subplot(132)
    plt.axis('off')
    plt.title('zero')
    ax2.imshow(g1, cmap='gray')

    ax3 = fig.add_subplot(133)
    plt.axis('off')
    plt.title('edge')
    ax3.imshow(g2, cmap='gray')
    plt.show()
