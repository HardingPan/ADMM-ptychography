import numpy as np
from scipy.signal import correlate2d
from scipy.linalg import norm

def quancomp(object, obeject_Func):
    # 获取 object 的尺寸
    # m, n = object.shape
    #
    # # 计算归一化交叉相关性
    # correlation1 = correlate2d(abs(big_obj), abs(object), mode='same')
    # h1 = big_obj.shape[0] // 2, big_obj.shape[1] // 2
    #
    # # 寻找最大相关性
    # max1 = np.max(np.abs(correlation1[h1[0]-m:h1[0]+m, h1[1]-n:h1[1]+n]))
    # I = np.where(np.abs(correlation1) == max1)
    #
    # # 提取与 object 大小相同的区域
    # I1, I2 = I[0][0], I[1][0]
    # object1 = big_obj[I1-m+1:I1+1, I2-n+1:I2+1]

    # 计算归一化幅度和误差
    object1 = obeject_Func
    shift1 = norm(object1, 'fro')
    norm_amp = np.abs(object1 / shift1)
    err_amp = norm(norm_amp - np.abs(object), 'fro') / norm(np.abs(object), 'fro')

    # 计算归一化角度和误差
    shift2 = np.sum(np.conj(object1) * object)
    shift2 = shift2 / norm(shift2)
    norm_angle = np.angle(object1 * shift1)
    err_angle = norm(norm_angle - np.angle(object), 'fro') / norm(np.angle(object), 'fro')

    return err_amp, err_angle