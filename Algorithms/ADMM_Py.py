import numpy as np
from util.Propagate import  Propagate
import matplotlib.pyplot as plt
from util.relative_err import relative_err

from util.evalution import complex_PNSR
from util import Diff
from util import Sampling
import regularization as re
from FFDNet_1.models import FFDNet
import torch
from util.center_find import center_mass_correct,center_mass_caculate

import cv2
def ADMM_Py_TV(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type,z,wavelength,pix,object):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    n1,n2 = objectFunc.shape
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 0
    err_2 = [1]
    miu = np.zeros((ysize, xsize), dtype=np.complex64)


    ################ ADMM part parameter seting################
    rho =0.1; epsilon =0.1;  lam =0.001 #global 0.03 0.03 0.03
    Z = np.zeros(shape=(n1, n2,2), dtype=np.complex64) + 1e-32
    Q =0
    V = np.zeros(shape=(n1,n2,2 ), dtype=np.complex64) + 1e-32
    W = 0
    probe_2 = probe
    objectFunc_2 = objectFunc

    while k < n:
        # if k!= 0:
        #      #probe = probe_N
        #      objectFunc = objectFunc_N
        for pos in positions:
            objectIlluminated = objectFunc[pos[0] + illu_indy, pos[1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # update object function
            miu = Propagate(g, type, pix, wavelength, z)

            miu_2 = miu - miu * np.sqrt(diffSet[diffSetIndex]) / abs(miu)

            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)

            objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated -2 * miu_3

            probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z)
            probe = probe - 0.2* probe_1 * np.conj(objectIlluminated)





            # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]) - abs( miu)**2)) ** 2))
            #
            # if k == 0:
            #  # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
            #  diffSet_num += np.sum(abs(diffSet[diffSetIndex]) ** 2)

            diffSetIndex += 1

        k += 1
        ################ ADMM part################
        objectFunc = objectFunc - rho * Diff.diff_xy_transpose(Diff.diff_xy(objectFunc) + V - Z)
        probe = probe - epsilon * (probe - Q + W)
        Z = Diff.sign(Diff.diff_xy(objectFunc) + V) * np.maximum(abs(Diff.diff_xy(objectFunc) + V) - lam / rho, 0)
        Q = probe
        V = V + Diff.diff_xy(objectFunc) - Z
        W = W + probe - Q


        # nesterov  apart ##########
        # if k > 1:
        #     probe = probe + (k / (k + 3)) * (probe - probe_N)
        #     objectFunc = objectFunc + (k / (k + 3)) * (objectFunc - objectFunc_N)
        #
        # objectFunc_N = objectFunc
        # probe_N = probe




        err_3.append(relative_err(object, objectFunc,130))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        # err_2.append(np.sum(err_1) / diffSet_num)
        # del err_1[:]

    psnr_am, psnr_ph, ssim_am, ssim_ph = complex_PNSR(object, objectFunc)
    print('ADMM_TV振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph)
    #plt.figure()
    plt.semilogy(err_3, linewidth=1.5, color='yellow', label='ADMM_TV', alpha=0.7)
    plt.legend()
    plt.plot(err_3, color='yellow')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')

    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3[-1]


def ADMM_Py_Aps(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type,z,wavelength,pix,object,lis,sigma,lamda,colour):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    n1, n2 = objectFunc.shape
    objectFunc = np.ones((objectSize), dtype=np.complex64)

    n1,n2 = objectFunc.shape
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 0
    err_2 = [1]
    weight = np.zeros(objectSize)
    # positions = sigma*positions

    #### pixl super-resolution ####


    illu_indy, illu_indx = np.indices((probe.shape))


    ################ ADMM part parameter seting################
    rho =0.1; epsilon =0.1;  lam =lamda
    Z = np.zeros(shape=(n1, n2,2), dtype=np.complex64) + 1e-32
    Q =0
    V = np.zeros(shape=(n1,n2,2 ), dtype=np.complex64) + 1e-32
    W = 0
    probe_2 = probe
    objectFunc_2 = objectFunc
    probe_3 = np.zeros(probe.shape, dtype=np.complex128)

    while k < n:

        for pos in lis:
            objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # update object function
            miu = Propagate(g, type, pix, wavelength, z)

            miu_2 = miu*Sampling.CT((1 - np.sqrt(diffSet[pos] / Sampling.C(abs(miu)**2,sigma))),sigma)


            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)

            objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = objectIlluminated - 2 * miu_3


            probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(objectIlluminated)
            probe = probe - 0.2 * probe_1

            # err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(miu)))))
            # #
            if k == 0:
            #  # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
             diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])))




        # # # ############# ADMM part################
        objectFunc = objectFunc - rho * Diff.diff_xy_transpose(Diff.diff_xy(objectFunc) + V - Z)
        Z = Diff.sign(Diff.diff_xy(objectFunc) + V) * np.maximum(abs(Diff.diff_xy(objectFunc) + V) - lam / rho, 0)
        Q = probe
        V = V + Diff.diff_xy(objectFunc) - Z
        W = W + probe - Q

        #
        # ##############nesterov  apart ##########
        if k > 2:
            objectFunc = objectFunc + (k / (k + 3)) * (objectFunc - objectFunc_N)
            probe = probe + (k / (k + 3)) * (probe - probe_N)
        objectFunc_N = objectFunc
        probe_N = probe


        k += 1



        # err_3.append(relative_err(object, objectFunc,250))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        err_2.append(np.sum(err_1) / diffSet_num)
        del err_1[:]

    # psnr_am, psnr_ph, ssim_am, ssim_ph, RMSE = complex_PNSR(object, objectFunc)
    # print('APSP-TV振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph, '\n相位RMSE=', RMSE)
    plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color=colour, label=lam, alpha=0.7)
    plt.legend()
    plt.plot(err_2, color=colour)
    plt.xlabel('iteration number')
    plt.ylabel('relative error')
    print(err_2[-1])
    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3
def ADMM_Py_ps(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type,z,wavelength,pix,object,lis,sigma):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    n1, n2 = objectFunc.shape
    objectFunc = np.ones((objectSize), dtype=np.complex64)

    n1,n2 = objectFunc.shape
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 1
    err_2 = [1]
    weight = np.zeros(objectSize)
    # positions = sigma*positions

    #### pixl super-resolution ####


    illu_indy, illu_indx = np.indices((probe.shape))


    ################ ADMM part parameter seting################
    rho =0.1; epsilon =0.1;  lam =0.001
    Z = np.zeros(shape=(n1, n2,2), dtype=np.complex64) + 1e-32
    Q =0
    V = np.zeros(shape=(n1,n2,2 ), dtype=np.complex64) + 1e-32
    W = 0
    probe_2 = probe
    objectFunc_2 = objectFunc
    probe_3 = np.zeros(probe.shape, dtype=np.complex128)
    #
    while k < n:

        for pos in positions:
            objectIlluminated = objectFunc[pos[0] + illu_indy, pos[1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # update object function
            miu = Propagate(g, type, pix, wavelength, z)

            miu_2 = miu*Sampling.CT((1 - np.sqrt(diffSet[diffSetIndex] / Sampling.C(abs(miu)**2,sigma))),sigma)


            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)

            objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated - 2 * miu_3


            probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(objectIlluminated)
            probe = probe - 2 * probe_1




            #err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]) - abs(miu)**2)) ** 2))
            # #
            # if k == 0:
            # #  # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
            #  diffSet_num += np.sum(abs(diffSet[diffSetIndex]) ** 2)

            diffSetIndex += 1

    #
    # while k < n:
    #
    #     objectFunc_gr = np.zeros(objectSize, dtype=np.complex64)
    #     temp1 = np.zeros(shape=(ysize, xsize))
    #     temp2 = np.zeros(objectSize)
    #     probe_2 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
    #     for pos in lis:
    #         objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
    #         # Guessed wave field from the object at position R
    #         g = objectIlluminated * probe
    #
    #         # update object function
    #         miu = Propagate(g, type, pix, wavelength, z)
    #         miu_2 = miu*Sampling.CT((1 - np.sqrt(diffSet[pos] / Sampling.C(abs(miu)**2,sigma))),sigma)
    #
    #         #
    #         # probe_1=0.5 * Propagate(miu_2, type, pix, wavelength, -z)
    #         # probe = probe - 0.2*probe_1*np.conj(objectIlluminated)
    #         probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(objectIlluminated)
    #         probe_2 += probe_1
    #         temp1 += abs(objectIlluminated) ** 2
    #
    #     probe = probe - 2 * probe_2 / (np.max(temp1))
    #     for pos in lis:
    #         objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
    #         # Guessed wave field from the object at position R
    #         g = objectIlluminated * probe
    #         # update object function
    #         miu = Propagate(g, type, pix, wavelength, z)
    #         miu_2 = miu * Sampling.CT((1 - np.sqrt(diffSet[pos] / Sampling.C(abs(miu) ** 2, sigma))), sigma)
    #         miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)
    #         objectIlluminated_gr = miu_3
    #         objectFunc_gr[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += objectIlluminated_gr
    #         temp2[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += abs(probe) ** 2
    #         if k == 0:
    #             # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
    #             # diffSet_num += np.sum(abs(diffSet[pos]) ** 2)
    #             diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])))
    #         # err_1.append(np.sum((abs(abs(diffSet[pos]) - abs(miu) ** 2)) ** 2))
    #         # err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(miu)))))
    #     objectFunc = objectFunc - 2 * objectFunc_gr / (np.max(temp2))
    #
        # # ############# ADMM part################
        objectFunc = objectFunc - rho * Diff.diff_xy_transpose(Diff.diff_xy(objectFunc) + V - Z)
        probe = probe - epsilon * (probe - Q + W)
        Z = Diff.sign(Diff.diff_xy(objectFunc) + V) * np.maximum(abs(Diff.diff_xy(objectFunc) + V) - lam / rho, 0)
        Q = probe
        V = V + Diff.diff_xy(objectFunc) - Z
        W = W + probe - Q

        #
        ##############nesterov  apart ##########

        # if k > 2:
        #     probe = probe + (k / (k + 3)) * (probe - probe_N)
        #     objectFunc = objectFunc + (k / (k + 3)) * (objectFunc - objectFunc_N)
        #
        # objectFunc_N = objectFunc
        # probe_N = probe

        k += 1



        err_3.append(relative_err(object, objectFunc,250))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        #err_2.append(np.sum(err_1) / diffSet_num)
        del err_1[:]

    psnr_am, psnr_ph, ssim_am, ssim_ph, RMSE = complex_PNSR(object, objectFunc)
    print('PSP-TV振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph, '\n相位RMSE=', RMSE)
    plt.figure()
    plt.semilogy(err_3, linewidth=1.5, color='red', label='Without Nesterov', marker='*', alpha=0.7, markevery=30)
    plt.legend()
    plt.plot(err_3, color='red')
    plt.xlabel('iteration number')
    plt.ylabel('relative error')
    print(err_3[-1])
    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3[-1]




def ADMM_net_denoise(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object,lis, is_center_correct):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    n1, n2 = objectFunc.shape
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 1
    err_2 = [1]

    print('Loading model ...\n')
    model = FFDNet(num_input_channels=1)
    model_fn = torch.load("FFDNet_1/models/net_gray.pth")

    # opt = parse_options(is_train=False)
    # model = create_model(opt)
    ################ ADMM part parameter seting################
    # rho = 0.1  #0.1
    # epsilon = 0.1  #0.1
    # lam =1.5e-4 # 5e-5
    # Z = np.zeros(shape=(n1, n2), dtype=np.complex64)
    # Q = 0
    # V = np.zeros(shape=(n1, n2), dtype=np.complex64)
    # W = 0

    h = re.denosise

    probe_2 = np.zeros(shape=(ysize, xsize), dtype=np.complex128)

    noise = np.zeros(shape=(ysize, xsize), dtype=np.float)
    noise2 = np.zeros(shape=(ysize, xsize), dtype=np.float)


    pho = 1e-9
    epsilon = 0.1   # 0.5
    belta = 2e-5 #1e-3
    s = np.ones(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    x = np.zeros(shape=(len(positions), ysize, xsize))
    s_last = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux_last = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    noise = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.float)
    eta = np.ones(shape=(len(positions), ysize, xsize), dtype=np.float)
    noise2 = np.zeros(shape=(ysize, xsize), dtype=np.float)
    wmiga = np.zeros(shape=(n1,n2), dtype=np.complex64)
    niu = np.zeros(shape=(n1,n2), dtype=np.complex64)
    # wmiga = np.zeros(shape=(n1, n2,2), dtype=np.complex64)
    # niu = np.zeros(shape=(n1, n2,2), dtype=np.complex64)
    f = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    abs_f = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    J= 100
    for pos in lis:
        objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
        g = objectIlluminated * probe
        s[pos, :, :] = g
        x[pos, :, :] = abs(s[pos, :, :])
        s_last[pos, :, :] = s[pos, :, :]
    is_tr = 0
    while k < n:


        temp1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp1_1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp2 = np.zeros(objectSize, dtype=np.complex64)
        temp2_1 = np.zeros(objectSize, dtype=np.complex64)
        temp3 = np.zeros(objectSize, dtype=np.complex64)
        is_adp = 0
        # 探针更新
        for pos in lis:
            miu2 = np.conj(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) * (
                    s[pos, :, :] - aux[pos, :, :])
            temp1 += miu2
            temp1_1 += abs(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) ** 2

        # probe = probe - 0.9 * temp1 / np.max(temp1_1)
        probe = pho*temp1 / (pho*temp1_1 + 1e-9)
        x_shift, y_shift = center_mass_caculate(probe)
        probe_correct= center_mass_correct( probe, x_shift, y_shift )
        # probe_correct = center_mass_correct(probe)
        probe = (probe + probe_correct) / 2  ###注意，这部分是论文中没有的，后面新加的防止探针移动的办法，很有效
        probe = np.where(abs(probe) > 1e8, 1e8, abs(probe)) * Diff.sign(probe)
        # probe = center_mass_correct(probe)
        # 物体更新
        for pos in lis:
            #
            miu = Propagate(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix,
                            wavelength, z)
            # miu1 = s[pos, :, :] - miu
            err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(miu)))))
            # err_1.append(np.sum((abs(abs(diffSet[pos]) - abs(miu) ** 2)) ** 2))

            miu2 = np.conj(probe) * (s[pos, :, :] - aux[pos, :, :])
            # miu2 = abs(probe) ** 2 * objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] - \
            #        np.conj(probe) * (s[pos, :, :] - aux[pos, :, :])
            temp2[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += miu2
            temp2_1[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += abs(probe) ** 2

            ##误差计算
            if k == 1:
                diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])))
            # diffSet_num += np.sum(abs(diffSet[pos]) ** 2)

        objectFunc = (pho*temp2 + 0.1*pho*(wmiga - niu))/ (pho*temp2_1 + 0.1*pho )
        objectFunc = np.where(abs(objectFunc) > 1e8, 1e8, abs(objectFunc)) * Diff.sign(objectFunc)
        if is_center_correct == 1:
         #探针位置纠正
         x_shift, y_shift = center_mass_caculate(probe)
         probe = center_mass_correct(probe, x_shift, y_shift)
         objectFunc = center_mass_correct(objectFunc, x_shift, y_shift)
        for pos in lis:

            miu = Propagate(
                objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe + aux[pos, :, :], type,
                pix,
                wavelength, z)

            f[pos, :, :] = np.sqrt(diffSet[pos] - noise[pos, :, :])* np.exp(1j*np.angle(miu))

            noise[pos, :, :] = diffSet[pos] - abs(f[pos, :, :])**2
            # f[pos, :, :] = np.sqrt(diffSet[pos] - noise[pos, :, :]) * np.exp(1j * np.angle(miu))
            miu2 = (f[pos, :, :] + pho * miu) / (1 + pho)
            s[pos, :, :] = Propagate(miu2, type, pix, wavelength, -z)
            aux[pos, :, :] = aux[pos, :, :] + objectFunc[
                positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe - s[pos, :, :]

        #     if np.sqrt(np.sum(abs(s[pos, :, :] - s_last[pos, :, :]) ** 2)) > 1e-2 / k or \
        #             np.sqrt(np.sum(abs(aux[pos, :, :] - aux_last[pos, :, :]) ** 2)) > 1e-2 / k or \
        #             np.sqrt(np.sum(abs(aux[pos, :, :]) ** 2)) > 1e+8:
        #         is_adp = 1
        if k < 100:
            pho = pho * 1.1
        # if np.sqrt(np.sum(abs(aux[pos, :, :] - aux_last[pos, :, :]))) / len(positions) / k < 1e-3:
        #     is_tr = 1
        if is_tr == 1:
            pho = 0.1
        if k > 200:
            pho = 1
        print(pho)
        s_last = s.copy()
        aux_last = aux.copy()
        # 噪声初始化
        # if k == 50:
        #     for pos in lis:
        #         temp = Propagate(
        #             objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix,
        #             wavelength, z)
        #         noise2 += diffSet[pos] - abs(temp) ** 2
        #     for pos in lis:
        #      noise[pos,:,:] = noise2 / len(positions)
        #     noise = np.where(noise > 0, noise, 0)
        # noise2 = noise /len(positions)
        # noise = 0
        # wmiga = Diff.sign(Diff.diff_xy(objectFunc) + niu) * np.maximum(abs(Diff.diff_xy(objectFunc) + niu) - belta /epsilon, 0)

        wmiga = h.FFDNetdeblur((objectFunc+niu), model=model,sigma = np.sqrt(belta / epsilon), model_fn =  model_fn)
        niu = niu + objectFunc - wmiga
        # niu =  niu + Diff.diff_xy(objectFunc) -wmiga
        k+=1





        err_3.append(relative_err(object, objectFunc,pixsum=250))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        err_2.append(np.sum(err_1) / diffSet_num)
        del err_1[:]
    #
    # objectFunc = h.FFDNetdeblur((objectFunc), model=model, sigma=np.sqrt(belta / epsilon), model_fn=model_fn)

    psnr_am, psnr_ph, ssim_am, ssim_ph, RMSE = complex_PNSR(object, objectFunc)
    print('T_ADMM_net振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph, '\n相位RMSE=', RMSE)
    # plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='red', label='T_ADMM_NET', alpha=0.7)
    plt.legend()
    plt.plot(err_2, color='red')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')

    # End of iterations
    print(err_2[-1], err_3[-1])
    print('End of iterations')
    return objectFunc, probe, err_3

"还未更改 这部分"
def ADMM_TV_denoise(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object, lis, is_center_correct):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    n1, n2 = objectFunc.shape
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 1
    err_2 = [1]

    # W = 0

    h = re.denosise

    probe_2 = np.zeros(shape=(ysize, xsize), dtype=np.complex128)

    noise = np.zeros(shape=(ysize, xsize), dtype=np.float)
    noise2 = np.zeros(shape=(ysize, xsize), dtype=np.float)


    pho = 1e-9
    epsilon = 0.1   # 0.5
    belta = 1e-4 #1e-3
    s = np.ones(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    x = np.zeros(shape=(len(positions), ysize, xsize))
    s_last = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux_last = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    noise = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.float)
    eta = np.ones(shape=(len(positions), ysize, xsize), dtype=np.float)
    noise2 = np.zeros(shape=(ysize, xsize), dtype=np.float)
    wmiga = np.zeros(shape=(n1,n2,2), dtype=np.complex64)
    niu = np.zeros(shape=(n1,n2,2), dtype=np.complex64)
    # wmiga = np.zeros(shape=(n1, n2,2), dtype=np.complex64)
    # niu = np.zeros(shape=(n1, n2,2), dtype=np.complex64)
    f = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    abs_f = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    J= 100
    for pos in lis:
        objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
        g = objectIlluminated * probe
        s[pos, :, :] = g
        x[pos, :, :] = abs(s[pos, :, :])
        s_last[pos, :, :] = s[pos, :, :]
    is_tr = 0
    while k < n:


        temp1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp1_1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp2 = np.zeros(objectSize, dtype=np.complex64)
        temp2_1 = np.zeros(objectSize, dtype=np.complex64)
        temp3 = np.zeros(objectSize, dtype=np.complex64)
        is_adp = 0
        # 探针更新
        for pos in lis:
            miu2 = np.conj(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) * (
                    s[pos, :, :] - aux[pos, :, :])
            temp1 += miu2
            temp1_1 += abs(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) ** 2

        # probe = probe - 0.9 * temp1 / np.max(temp1_1)
        probe = pho*temp1 / (pho*temp1_1 + 1e-9)
        x_shift, y_shift = center_mass_caculate(probe)
        probe_correct = center_mass_correct(probe, x_shift, y_shift)
        probe = (probe + probe_correct) / 2  ###注意，这部分是论文中没有的，后面新加的防止探针移动的办法，很有效
        probe = np.where(abs(probe) > 1e8, 1e8, abs(probe)) * Diff.sign(probe)
        # 物体更新
        for pos in lis:
            #
            miu = Propagate(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix,
                            wavelength, z)
            # miu1 = s[pos, :, :] - miu
            err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(miu)))))
            # err_1.append(np.sum((abs(abs(diffSet[pos]) - abs(miu) ** 2)) ** 2))

            # miu2 = np.conj(probe) * (s[pos, :, :] - aux[pos, :, :])
            miu2 = abs(probe) ** 2 * objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] - \
                   np.conj(probe) * (s[pos, :, :] - aux[pos, :, :])
            temp2[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += miu2
            temp2_1[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += abs(probe) ** 2

            ##误差计算
            if k == 1:
                diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])))
            # diffSet_num += np.sum(abs(diffSet[pos]) ** 2)

        # objectFunc = (pho*temp2 + 0.1*pho*Diff.diff_xy_transpose(wmiga - niu))/ (pho*temp2_1 + pho)
        objectFunc = objectFunc - 0.9*(temp2 + epsilon*(Diff.diff_xy_transpose(Diff.diff_xy(objectFunc) - wmiga + niu)))/ (np.max(temp2_1))
        objectFunc = np.where(abs(objectFunc) > 1e8, 1e8, abs(objectFunc)) * Diff.sign(objectFunc)
        if is_center_correct == 1:
         #探针位置纠正
         x_shift, y_shift = center_mass_caculate(probe)
         probe = center_mass_correct(probe, x_shift, y_shift)
         objectFunc = center_mass_correct(objectFunc, x_shift, y_shift)
        for pos in lis:

            miu = Propagate(
                objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe + aux[pos, :, :], type,
                pix,
                wavelength, z)

            f[pos, :, :] = np.sqrt(diffSet[pos] - noise[pos, :, :])* np.exp(1j*np.angle(miu))

            noise[pos, :, :] = diffSet[pos] - abs(f[pos, :, :])**2
            # f[pos, :, :] = np.sqrt(diffSet[pos] - noise[pos, :, :]) * np.exp(1j * np.angle(miu))
            miu2 = (f[pos, :, :] + pho * miu) / (1 + pho)
            s[pos, :, :] = Propagate(miu2, type, pix, wavelength, -z)
            aux[pos, :, :] = aux[pos, :, :] + objectFunc[
                positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe - s[pos, :, :]

        #     if np.sqrt(np.sum(abs(s[pos, :, :] - s_last[pos, :, :]) ** 2)) > 1e-2 / k or \
        #             np.sqrt(np.sum(abs(aux[pos, :, :] - aux_last[pos, :, :]) ** 2)) > 1e-2 / k or \
        #             np.sqrt(np.sum(abs(aux[pos, :, :]) ** 2)) > 1e+8:
        #         is_adp = 1
        if k < 100:
            pho = pho * 1.1
        # if np.sqrt(np.sum(abs(aux[pos, :, :] - aux_last[pos, :, :]))) / len(positions) / k < 1e-3:
        #     is_tr = 1
        if is_tr == 1:
            pho = 0.1
        if k > 200:
            pho = 1
        print(pho)
        s_last = s.copy()
        aux_last = aux.copy()
        # 噪声初始化
        # if k == 50:
        #     for pos in lis:
        #         temp = Propagate(
        #             objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix,
        #             wavelength, z)
        #         noise2 += diffSet[pos] - abs(temp) ** 2
        #     for pos in lis:
        #      noise[pos,:,:] = noise2 / len(positions)
        #     noise = np.where(noise > 0, noise, 0)
        # noise2 = noise /len(positions)
        # noise = 0
        wmiga = Diff.sign(Diff.diff_xy(objectFunc) + niu) * np.maximum(abs(Diff.diff_xy(objectFunc) + niu) - belta / epsilon, 0)

        # wmiga = h.FFDNetdeblur((objectFunc+niu), model=model,sigma = np.sqrt(belta / epsilon), model_fn =  model_fn)
        # niu = niu + objectFunc - wmiga
        niu =  niu + Diff.diff_xy(objectFunc) -wmiga
        k+=1





        err_3.append(relative_err(object, objectFunc,pixsum=250))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        err_2.append(np.sum(err_1) / diffSet_num)
        del err_1[:]
    #
    # objectFunc = h.FFDNetdeblur((objectFunc), model=model, sigma=np.sqrt(belta / epsilon), model_fn=model_fn)

    psnr_am, psnr_ph, ssim_am, ssim_ph, RMSE = complex_PNSR(object, objectFunc)
    print('T_ADMM_TV振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph, '\n相位RMSE=', RMSE)
    # plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='red', label='T_ADMM_TV', alpha=0.7)
    plt.legend()
    plt.plot(err_2, color='red')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')

    # End of iterations
    print(err_2[-1])
    print('End of iterations')
    return objectFunc, probe, err_3
