import numpy as np
import math


def gaussKernel(sig,m):
    if m == 'no':     #此处一直忘了加 ''所以一直报错哎！
        m = math.ceil(3*sig)*2 + 1
    else:
        m = int(m)
        if m<(math.ceil(3*sig)*2 + 1):
           print('warring:m is too small')
    gausskernel=np.zeros((m,m))
    center = m // 2
    for i in range (0,m):
        for j in range (0,m):
            x = i-center
            y = j-center
            norm=math.pow(x,2)+pow(y,2)
            gausskernel[i,j]=(1/2*np.pi*sig*sig)*math.exp(-norm/(2*math.pow(sig,2)))   # 求高斯
    sum=np.sum(gausskernel)   # 求和
    l=gausskernel/sum   # 归一化
    return l


if __name__=='__main__':
    sig = 1.0
    print('输入‘no’或者是已知的数m')
    n = input()   #no代表m没有提供
    w = gaussKernel(sig,m=n)
    print(w)