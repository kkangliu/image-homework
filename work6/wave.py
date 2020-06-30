import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt


# 添加高斯噪声
def gaussiannoise(f, means, var):
    image = np.array(f / 255, dtype=float)
    noise = np.random.normal(means, var ** 0.5, image.shape)
    F = image + noise
    if F.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.0
    F = np.clip(F, low_clip, 1.0)
    F = np.uint8(F * 255)
    return F


# 调用 pywavelets 库 实现小波变换
def wave(f):
    coeffs = pywt.dwt2(f, 'bior4.4')
    LL, (LH, HL, HH) = coeffs
    return LL,LH,HL,HH,coeffs

if __name__=='__main__':
    f = cv2.imread('lena.tiff', 0)
    m,n = f.shape
    f_noise = gaussiannoise(f, 0, 0.01)
    f_n_LL, f_n_LH, f_n_HL, f_n_HH, c = wave(f_noise)
    # 求sigma噪声方差
    HH = abs(f_n_HH)
    sigma_n = np.median(HH)/0.6745
    #求信号方差
    sigma_1 = sum(sum(f_n_LL ** 2)) / (m/2 * n/2) - sigma_n ** 2
    sigma_2 = sum(sum(f_n_LH ** 2)) / (m/2 * n/2) - sigma_n ** 2
    sigma_3 = sum(sum(f_n_HL ** 2)) / (m/2 * n/2) - sigma_n ** 2
    sigma_4 = sum(sum(f_n_HH ** 2)) / (m/2 * n/2) - sigma_n ** 2
    x1 = sigma_1 / (sigma_1 + sigma_n ** 2) * f_n_LL
    x2 = sigma_2 / (sigma_2 + sigma_n ** 2) * f_n_LH
    x3 = sigma_3 / (sigma_3 + sigma_n ** 2) * f_n_HL
    x4 = sigma_4 / (sigma_4 + sigma_n ** 2) * f_n_HH
    coeffs2 = x1, (x2, x3, x4)
    # 小波逆变换
    x = pywt.idwt2(coeffs2, 'bior4.4')
    y = x- f_noise

    fig = plt.figure(figsize=(8,8))  # figure 的大小
    ax1 = fig.add_subplot(221)
    plt.axis('off')
    plt.title('f')
    plt.imshow(f, cmap='gray')

    ax2 = fig.add_subplot(222)
    plt.axis('off')
    plt.title('f_noise')
    plt.imshow(f_noise, cmap='gray')

    ax3 = fig.add_subplot(223)
    plt.axis('off')
    plt.title('f_no_noise')
    plt.imshow(x, cmap='gray')

    ax4 = fig.add_subplot(224)
    plt.axis('off')
    plt.title('f_no_noise - f_noise')
    plt.imshow(y, cmap='gray')

    plt.show()

