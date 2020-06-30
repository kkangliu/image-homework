
def get_dsf(image_size, angle, dis):
    PSF = np.zeros(image_size)
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2
    sin_val = math.sin(angle * math.pi / 180)
    cos_val = math.cos(angle * math.pi / 180)
    for i in range(dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1
    return PSF / PSF.sum()


def wiener(f, K = 0.01):
    F = np.fft.fft2(f)
    F_log = (np.log(1+abs(F)))**2
    H = np.real(np.fft.ifft2(F_log))
    H = np.fft.fftshift(H)
    for i in range(m):
        for j in range(n):
            if H[i][j] > 1:
                H[i][j] = 255
            else:
                H[i][j] = 0
    H_new = 1*H
    edges = cv2.Canny(np.array(H_new,dtype='uint8'), 0, 30, apertureSize=5)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)
    for rho, theta in lines[0]:
        print("rho:{}, theta:{}".format(rho, theta))
    PSF = get_dsf((512,512), 120, 26)
    PSF_fft = np.fft.fft2(PSF)
    M = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = np.fft.ifft2(F * M)
    r = np.abs(np.fft.fftshift(result))
    return H_new,r


if name__=='__main__':
    f = cv2.imread('test.png', 0)
    m,n = f.shape
    H, result = wiener(f)
    fig = plt.figure(figsize=(9,3))  
    ax1 = fig.add_subplot(131)
    plt.axis('off')
    plt.title('f')
    plt.imshow(f, cmap='gray')
    ax2 = fig.add_subplot(132)
    plt.axis('off')
    plt.title('daopu')
    plt.imshow(H, cmap='gray')
    ax3 = fig.add_subplot(133)
    plt.axis('off')
    plt.title('result')
    plt.imshow(result, cmap='gray')
    plt.show()

