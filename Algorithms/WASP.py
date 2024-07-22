"""
Created on 2024 06 25

@author: andyMaiden

Based on ""WASP: Weighted Average of Sequential Projections for ptychographic  phase retrieval," Optics Express 32(12), pp. 21327-21344, (2024).
"""

import matplotlib.pyplot as plt
import numpy as np

from util import Diff
from util.Propagate import Propagate
from util.center_find import center_mass_caculate, center_mass_correct
from util.evalution import complex_PNSR
from util.relative_err import relative_err


def Wasp(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object, lis, is_center_correct):
    # 定义函数WASP，用于pychographic相位恢复
    # 参数说明：
    # n：迭代次数
    # diffSet：衍射图像数据集
    # probe：初始化的探针图像
    # objectSize：物体函数的大小
    # positions：每个点的位置
    # illu_indy：光源在目标函数中的位置
    # illu_indx：光源在目标图像中的位置
    # type：传播类型
    # z：传播距离
    # wavelength：波长
    # pix：像素大小
    # object：原始图像
    # lis：光源位置列表
    # is_center_correct：是否中心对齐

    # 探针的大小
    ysize, xsize = probe.shape

    # 全一初始化复数物体
    objectFunc = np.ones(objectSize, dtype=np.complex64)

    # 初始化复数物体的照明对应部分
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)

    # 初始化算法中间变量
    g = np.zeros((ysize, xsize), dtype=np.complex64)
    gprime = np.zeros((ysize, xsize), dtype=np.complex64)
    G = np.zeros((ysize, xsize), dtype=np.complex64)
    Gprime = np.zeros((ysize, xsize), dtype=np.complex64)

    # 定义循环次数
    k = 0

    # 衍射强度和初始化，作为误差计算分母
    diffSet_num = 0
    # 初始化误差列表
    err_1 = []
    err_2 = [1]
    err_3 = [1]
    # 算法参数设置
    alpha = 2
    belta = 1
    while k < n:

        numP = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        denP = np.zeros(shape=(ysize, xsize), dtype=np.float64)
        numO = np.zeros(objectSize, dtype=np.complex64)
        denO = np.zeros(objectSize, dtype=np.float64)
        for pos in lis:
            # 选取出pos位置的照明部分图像
            objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]

            # 照明部分图像和照明探针耦合
            g = objectIlluminated * probe

            # 耦合图像进行传播
            G = Propagate(g, type, pix, wavelength, z)

            # 使用模量约束
            Gprime = np.sqrt(diffSet[pos]) * G / abs(G)

            # 进行你傅里叶变换传播
            gprime = Propagate(Gprime, type, pix, wavelength, -z)

            # 更新整个物体中的照明部分物体
            objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = objectIlluminated + (
                    gprime - g) * np.conj(probe) / (abs(probe) ** 2 + alpha * np.mean(abs(probe) ** 2))

            # 更新探针部分

            probe = probe + (gprime - g) * np.conj(objectIlluminated) /((abs(objectIlluminated)) ** 2 + belta)

            #计算论文中的分子和以及分母和
            numO[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = numO[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] +\
                                                                                 np.conj(probe)*gprime

            denO[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = denO[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] + \
                                                                                 abs(probe) ** 2

            numP = numP + np.conj(objectIlluminated) * gprime
            denP = denP + abs(objectIlluminated) ** 2
            # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]**2) - abs(G)**2 ))**2))  #intensity err
            # err_1.append(np.sum((abs(abs(diffSet[pos]) - abs(G) ** 2)) ** 2))  # amplitude err
            err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(G)))))
            # 计算衍射强度和
            if k == 0:
                #   #diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
                # diffSet_num += np.sum(abs(diffSet[pos]**2))
                diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])))
        #更新整个物体
        objectFunc = numO / (denO + 1e-10)
        probe = numP / (denP + 1e-10)

        #物体能量纠正
        objectFunc = np.where(abs(objectFunc) > 2, 2, abs(objectFunc)) * Diff.sign(objectFunc)

        # 探针位置纠正
        if is_center_correct == 1:
         x_shift, y_shift = center_mass_caculate(probe)
         probe = center_mass_correct(probe, x_shift, y_shift)
         objectFunc = center_mass_correct(objectFunc, x_shift, y_shift)

        k += 1
        print('Iteration %d starts' % k)

        err_2.append(np.sum(err_1) / diffSet_num)
        err_3.append(relative_err(object, objectFunc, 250))
        #
        del err_1[:]
    #计算峰值信噪比和结构相似度和均方根误差
    psnr_am, psnr_ph, ssim_am, ssim_ph, RMSE = complex_PNSR(object, objectFunc)
    print('WASP振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph, '\n相位RMSE=', RMSE)

    #绘图
    plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='green', label='WASP', alpha=0.7, markevery=30)
    plt.legend(loc=3, prop={'size': 9})
    plt.xlabel('iteration number')
    plt.ylabel('relative error')
    print(err_2[-1], err_3[-1])
     # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3