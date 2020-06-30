"""
9th homework
@Author: LuShun
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


def threshold_binary(f, threshold, background='white'):
    result = np.zeros_like(f)
    if background == 'white':
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if f[i][j] < threshold:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
    else:
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if f[i][j] < threshold:
                    result[i][j] = 0
                else:
                    result[i][j] = 1
    return result


def erosion(f, k):
    _result = cv2.filter2D(f, -1, k)
    result = np.zeros_like(_result)
    result[np.where(_result == np.sum(k))] = 1
    return result


def dilation(f, k):
    _result = cv2.filter2D(f, -1, k)
    result = np.zeros_like(_result)
    result[np.where(_result >= 1)] = 1
    return result


def thinning(f, a, b):
    result = f.copy()
    f1 = erosion(result, a)
    pad = np.ones_like(f1)
    f2 = erosion(pad - result, b)
    _index = np.where((f1 == 1) & (f2 == 1))
    print('Thinning iteration index[0] number:', len(_index[0]))
    _result = np.zeros_like(f)
    _result[_index] = 1
    _result_c = np.ones_like(_result)
    _result_c -= _result
    final_result = np.zeros_like(f)
    index = np.where((result == 1) & (_result_c == 1))
    final_result[index] = 1
    return final_result, len(_index[0])


def find_border(f, k):
    result = f.copy()
    result -= erosion(result, k)
    return result


def distance_trans(f_border, f_binary, padding='zero'):
    # method == 'border'
    if padding == 'border':
        _pad = np.pad(f_border, ((1, 1), (1, 1)), 'edge')
    # method == 'zero'
    else:
        _pad = np.pad(f_border, ((1, 1), (1, 1)), 'constant', constant_values=0)
    for i in range(_pad.shape[0]):
        for j in range(_pad.shape[1]):
            if not _pad[i][j]:
                _pad[i][j] = 100
    result = np.ones_like(f_border) * 100
    # left to right, up to down
    for i in range(1, _pad.shape[0] - 1):
        for j in range(1, _pad.shape[1] - 1):
            # set background as 0
            if f_binary[i-1, j-1] == 0:
                result[i - 1, j - 1] = f_border[i - 1, j - 1]
                continue
            temp0 = _pad[i, j]
            temp1 = _pad[i - 1, j - 1] + 4
            temp2 = _pad[i - 1, j] + 3
            temp3 = _pad[i - 1, j + 1] + 4
            temp4 = _pad[i, j - 1] + 3
            result[i - 1, j - 1] = min(temp0, temp1, temp2, temp3, temp4)
    # method == 'border'
    if padding == 'border':
        pad = np.pad(result, ((1, 1), (1, 1)), 'edge')
    # method == 'zero'
    else:
        pad = np.pad(result, ((1, 1), (1, 1)), 'constant', constant_values=0)
    # right to left, down to up
    for i in range(pad.shape[0] - 2, 0, -1):
        for j in range(pad.shape[1] - 2, 0, -1):
            if f_binary[i-1, j-1] == 0:
                result[i - 1, j - 1] = f_border[i - 1, j - 1]
                continue
            temp0 = pad[i, j]
            temp1 = pad[i, j + 1] + 3
            temp2 = pad[i + 1, j - 1] + 4
            temp3 = pad[i + 1, j] + 3
            temp4 = pad[i + 1, j + 1] + 4
            result[i - 1, j - 1] = min(temp0, temp1, temp2, temp3, temp4)
    result = np.round(result/3).astype('uint8')
    return result


def local_max(f, k, padding='zero'):
    # method == 'border'
    if padding == 'border':
        pad = np.pad(f, ((1, 1), (1, 1)), 'edge')
    # method == 'zero'
    else:
        pad = np.pad(f, ((1, 1), (1, 1)), 'constant', constant_values=0)
    w = k // 2
    h = k // 2 + 1
    result = np.zeros_like(f)
    for i in range(1, pad.shape[0] - 1):
        for j in range(1, pad.shape[1] - 1):
            if not pad[i, j]:
                result[i - 1, j - 1] = 0
            elif np.max(pad[i - w:i + h, j - w:j + h]) == pad[i, j]:
                result[i - 1, j - 1] = 1
            else:
                result[i - 1, j - 1] = 0
            # print(result)
    return result


def skele_morphology(f, k1, k2):
    result = f.copy()
    last = f.copy()
    flag = 0
    while True:
        for i in range(8):
            result, num = thinning(result, k1[i], k2[i])
            if not num:
                flag += 1
                print('flag:', flag)
                if flag == 8:
                    return last
            else:
                flag = 0
            last = result.copy()


def cut(f, k1, k2, k3):
    result = f.copy()

    # thinning
    for i in range(3):
        for j in range(0, 8):
            result, _ = thinning(result, k1[i], k2[i])
    X1 = result.copy()
    last_result = np.zeros_like(result)

    # compensate
    _result = result.copy()
    for i in range(8):
        _result, _ = thinning(_result, k1[i], k2[i])
        result_sum = np.zeros_like(_result)
        # or
        result_sum[np.where((_result == 1) | (last_result == 1))] = 1
        last_result = result_sum.copy()

    # condition dilation
    _result = result_sum.copy()
    for i in range(3):
        _result = dilation(_result, k3)
        result_sum = np.zeros_like(_result)
        result_sum[np.where((_result == 1) & (f == 1))] = 1
        _result = result_sum.copy()

    X3 = result_sum.copy()
    result = np.zeros_like(f)
    result[np.where((X1 == 1) | (X3 == 1))] = 1
    return result


if __name__ == '__main__':
    # read and binary
    img = cv2.imread('smallfingerprint.jpg', 0)
    img_binary = threshold_binary(img, threshold=127, background='white')
    K1 = np.array([[[0, 0, 0], [0, 1, 0], [1, 1, 1]],
                   [[1, 0, 0], [1, 1, 0], [1, 0, 0]],
                   [[1, 1, 1], [0, 1, 0], [0, 0, 0]],
                   [[0, 0, 1], [0, 1, 1], [0, 0, 1]],
                   [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
                   [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                   [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
                   [[0, 0, 0], [0, 1, 1], [0, 1, 1]]])
    K2 = np.array([[[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                   [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                   [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
                   [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                   [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
                   [[0, 0, 0], [0, 0, 1], [0, 1, 1]],
                   [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                   [[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
    K3 = np.ones((3, 3), np.uint8)
    K4 = np.array([[[0, 0, 0], [1, 1, 0], [0, 0, 0]],
                 [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
                 [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
                 [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                 [[0, 0, 1], [0, 1, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                 [[0, 0, 0], [0, 1, 0], [0, 0, 1]]])
    K5 = np.array([[[1, 1, 1], [0, 0, 1], [1, 1, 1]],
                 [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
                 [[1, 1, 1], [1, 0, 0], [1, 1, 1]],
                 [[1, 1, 1], [1, 0, 1], [1, 0, 1]],
                 [[0, 1, 1], [1, 0, 1], [1, 1, 1]],
                 [[1, 1, 0], [1, 0, 1], [1, 1, 1]],
                 [[1, 1, 1], [1, 0, 1], [0, 1, 1]],
                 [[1, 1, 1], [1, 0, 1], [1, 1, 0]]])

    # img_skele_morphology
    img_skele_morphology = skele_morphology(img_binary, K1, K2)
    # img_skele_dist_trans
    img_border = find_border(img_binary, K3)
    img_dist_trans = distance_trans(img_border, img_binary)
    img_local_max = local_max(img_dist_trans, 3)
    img_skele_morphology_cut = cut(img_skele_morphology, K4, K5, K3)
    img_skele_dist_cut = cut(img_local_max, K4, K5, K3)

    # imshow
    fig = plt.figure(figsize=(15, 15))
    # original img
    ax1 = fig.add_subplot(331)
    plt.axis('off')
    plt.title('a.img', fontsize=20)
    ax1.imshow(img, cmap='gray')
    # img_binary
    ax2 = fig.add_subplot(332)
    plt.axis('off')
    plt.title('b.img_binary', fontsize=20)
    ax2.imshow(img_binary, cmap='gray')
    # img_skele_morphology
    ax3 = fig.add_subplot(333)
    plt.axis('off')
    plt.title('c.img_skele_morphology', fontsize=20)
    ax3.imshow(img_skele_morphology, cmap='gray')
    # img_border
    ax4 = fig.add_subplot(334)
    plt.axis('off')
    plt.title('d.img_border', fontsize=20)
    ax4.imshow(img_border, cmap='gray')
    # img_dist_trans
    ax5 = fig.add_subplot(335)
    plt.axis('off')
    plt.title('e.img_dist_trans', fontsize=20)
    ax5.imshow(img_dist_trans, cmap='gray')
    # img_local_max
    ax6 = fig.add_subplot(336)
    plt.axis('off')
    plt.title('f.img_local_max', fontsize=20)
    ax6.imshow(img_local_max, cmap='gray')
    # img_skele_morphology_cut
    ax7 = fig.add_subplot(337)
    plt.axis('off')
    plt.title('g.img_skele_morphology_cut', fontsize=20)
    ax7.imshow(img_skele_morphology_cut, cmap='gray')
    # img_skele_dist_cut
    ax8 = fig.add_subplot(338)
    plt.axis('off')
    plt.title('h.img_skele_dist_cut', fontsize=20)
    ax8.imshow(img_skele_dist_cut, cmap='gray')
    plt.show()
