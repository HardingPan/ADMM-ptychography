# -*- coding: utf-8 -*-
"""
Created on 2023 03 22

@author: Sanna

Based on "A phase retrieval algorithm for shifting illumination" J.M. Rodenburg and H.M.L Faulkner, [App. Phy. Lett.  85.20 (2004)]
"""

import numpy as np
import matplotlib.pyplot as plt
from util.Propagate import Propagate
from util.subpixel_registration import dftregistration


def various_PIE(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, algorithm,
                object, lis):
    # size of probe and diffraction patterns
    ysize, xsize = probe.shape

    # initialize object. make sure it can hold complex numbers
    objectFunc = np.ones(objectSize, dtype=np.complex64)

    # initalize that illuminated part of the object
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)

    # initialize algorithm wave fields (fourier and real)

    g = np.zeros((ysize, xsize), dtype=np.complex64)
    gprime = np.zeros((ysize, xsize), dtype=np.complex64)
    G = np.zeros((ysize, xsize), dtype=np.complex64)
    Gprime = np.zeros((ysize, xsize), dtype=np.complex64)

    # define iteration counter for outer loop
    k = 0

    # figure for animation
    # fig = plt.figure()

    # Initialize vector for animation data
    ims = []

    # idex for iterating through the diffraction patterns
    diffSetIndex = 0
    diffSet_num = 0
    # initialize vector for error calculation
    err_1 = []
    err_2 = [1]
    err_3 = [1]
    # Start of ePIE iterations
    while k < n:
        # Start of inner loop: (where you iterate through all probe positions R)
        # for pos in positions:
        for pos in lis:
            # Cut out the part of the image that is illuminated at R(=(ypos,xpos)
            objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # fft the wave field at position R to Fourier space
            G = Propagate(g, type, pix, wavelength, z)

            # make |PSI| confirm with the diffraction pattern from R
            Gprime = np.sqrt(diffSet[pos]) * G / abs(G)

            # inverse Fourier transform
            gprime = Propagate(Gprime, type, pix, wavelength, -z)

            # update the TOTAL object function with the illuminated part
            # The update should be the differens of the last iteration and the new one

            if algorithm == 'ePIE':
                alpha = 1  # higher value == faster change
                objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = objectIlluminated + alpha * (
                            gprime - g) * np.conj(
                    probe) / (np.max(abs(probe)) ** 2)  # probe* annars blir det att man delar med massa nollor
                # objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated + alpha * (gprime - g) * np.conj(
                #       probe)  # probe* annars blir det att man delar med massa nollor
                # update probe function
                beta =0.2 # higher value == faster change
                probe = probe + beta * (gprime - g) * np.conj(objectIlluminated) / (np.max(abs(objectIlluminated)) ** 2)


                # probe = probe + beta * (gprime - g) * np.conj(objectIlluminated)
            elif algorithm == 'PIE':
                alpha = 0.3  # higher value == faster change
                objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = objectIlluminated + abs(
                    probe) * (gprime - g) * np.conj(
                    probe) / (np.max(abs(probe)) * (abs(probe) ** 2 + alpha * np.max(
                    abs(probe)) ** 2))  # probe* annars blir det att man delar med massa nollor

                # update probe function
                beta = 0.3  # higher value == faster change
                probe = probe + beta * (gprime - g) * np.conj(objectIlluminated) * abs(
                    objectIlluminated) / (np.max(abs(objectIlluminated)) * (
                            abs(objectIlluminated) ** 2 + beta * np.max(abs(objectIlluminated)) ** 2))
            elif algorithm == 'rPIE':
                alpha = 0.1  # higher value == faster change
                objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = objectIlluminated + (
                            gprime - g) * np.conj(probe) / (alpha * np.max(abs(probe)) ** 2 + (1 - alpha) * abs(
                    probe) ** 2)

                # update probe function
                beta = 1  # higher value == faster change
                probe = probe + (gprime - g) * np.conj(objectIlluminated) / (
                            beta * np.max(abs(objectIlluminated)) ** 2 + (1 - beta) * abs(objectIlluminated) ** 2)

            else:
                print('invalid algorithm,please change')
                return objectFunc, probe, err_3

            # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]**2) - abs(G)**2 ))**2))  #intensity err
            # err_1.append(np.sum((abs(abs(diffSet[pos]) - abs(G) ** 2)) ** 2))  # amplitude err     #amplitude err
            err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(G)))))
            # caculate diffraction image sum
            if k == 0:
                #   #diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
                # diffSet_num += np.sum(abs(diffSet[pos]**2))
                diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])))
            ########################
            # Apply further constraints:
            # These 2 constraints are for transmission
            ########################

            # constrain object amplitude to 1
            # temp_Oamp = abs(objectFunc)
            # temp_Oamp[temp_Oamp>1] = 1
            # temp = np.angle(objectFunc)
            # objectFunc = temp_Oamp * np.exp(1j* temp)
            #
            # #constraint object phase to negative or 0
            # temp_Ophase = np.angle(objectFunc)
            # temp_Ophase[temp_Ophase>0] = 0
            # objectFunc = abs(objectFunc) * np.exp(1j* temp_Ophase)
            #
            # animate
            # im = plt.imshow(np.angle(objectFunc))

            # ims.append([im])

            diffSetIndex += 1

        # err_3.append(relative_err(object, objectFunc,250))
        k += 1
        print('Iteration %d starts' % k)

        # reset inner loop index
        diffSetIndex = 0
        err_2.append(np.sum(err_1) / diffSet_num)
        # err_3.append(relative_err(object, objectFunc, 250))
        #
        del err_1[:]
    #
    # psnr_am, psnr_ph, ssim_am, ssim_ph, RMSE = complex_PNSR(object, objectFunc)
    # print('rPIE振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph, '\n相位RMSE=', RMSE)

    if algorithm == 'ePIE':

        # plt.figure()
        plt.semilogy(err_2, linewidth=1.5, color='green', label=algorithm, alpha=0.7,marker='h', markevery=30)
        plt.legend(loc=3, prop={'size': 9})
        plt.xlabel('iteration number')
        plt.ylabel('relative error')
        print(err_2[-1], err_3[-1])
        # End of iterations
        print('End of iterations')
    elif algorithm == 'rPIE':
        plt.figure()
        plt.semilogy(err_2, linewidth=1.5, color='purple', label=algorithm, alpha=0.7,marker='d', markevery=30)
        plt.legend(loc=3, prop={'size': 9})
        plt.xlabel('iteration number')
        plt.ylabel('relative error')
        print(err_2[-1], err_3[-1])
        # End of iterations
        print('End of iterations')
    else:

        # plt.figure()
        plt.semilogy(err_2, linewidth=1.5, color='green', label='PIE', alpha=0.7, markevery=30)
        plt.legend(loc=3, prop={'size': 9})
        plt.xlabel('iteration number')
        plt.ylabel('relative error')
        print(err_2[-1], err_3[-1])
        # End of iterations
        print('End of iterations')

    # animate reconstruction

    # ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True, repeat=False)  # interval 帧之间的延迟  #是否使用光点切割优化绘图

    # save animation
    # .mp4 requires mencoder or ffmpeg to be installed
    # ani.save('ePIE.gif')
    # print('Saving animation')

    # show animation
    # plt.colorbar()

    # plt.show()
    return objectFunc, probe, err_3


def dis_crrect(n, z0, z1, deltaz, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix,
               algorithm, object, lis):
    # size of probe and diffraction patterns
    ysize, xsize = probe.shape
    object_rows, object_cols = object.shape
    # initialize object. make sure it can hold complex numbers
    objectFunc = np.ones(objectSize, dtype=np.complex64)

    # initialize that illuminated part of the object
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)

    k = 0
    # index for iterating through the diffraction patterns
    diffSetIndex = 0
    diffSet_num = 0
    # initialize vector for error calculation
    err_1 = []
    err_2 = []
    hash_table = {}
    e = 0.01
    epsilon = 1
    xinxishang = []
    count = 0
    probe_2 = np.ones(shape=(ysize, xsize), dtype=np.complex64)

    # Start of ePIE iterations

    # Start of inner loop: (where you iterate through all probe positions R)
    # for pos in positions:
    for z in np.arange(z0, z1 + deltaz, deltaz):

        objectFunc = np.ones(objectSize, dtype=np.complex64)
        probe1 = probe.copy()
        while k < n:
            for pos in lis:
                # Cut out the part of the image that is illuminated at R(=(ypos,xpos)
                objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]

                # Guessed wave field from the object at position R
                g = objectIlluminated * probe

                # fft the wave field at position R to Fourier space
                G = Propagate(g, type, pix, wavelength, z)

                # make |PSI| confirm with the diffraction pattern from R
                Gprime = np.sqrt(diffSet[pos]) * np.exp(1j * np.angle(G))

                # inverse Fourier transform
                gprime = Propagate(Gprime, type, pix, wavelength, -z)

                # update the TOTAL object function with the illuminated part
                # The update should be the differens of the last iteration and the new one

                if algorithm == 'ePIE':
                    alpha = 0.9  # higher value == faster change
                    objectFunc[
                        positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = objectIlluminated + alpha * (
                            gprime - g) * np.conj(
                        probe) / (np.max(abs(probe)) ** 2)  # probe* annars blir det att man delar med massa nollor
                    # objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated + alpha * (gprime - g) * np.conj(
                    #       probe)  # probe* annars blir det att man delar med massa nollor
                    # update probe function
                    beta = 0.2  # higher value == faster change
                    probe = probe + beta * (gprime - g) * np.conj(objectIlluminated) / (
                                np.max(abs(objectIlluminated)) ** 2)

                err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(G)))))  # amplitude err
                # caculate diffraction image sum
                if k == 0:
                    # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**2)
                    diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])))

                diffSetIndex += 1

            probe1 = probe1 - 2 * probe_2 / len(positions)
            probe_2 = 0
            k += 1
            print('%f distance serch Iteration %d starts' % (z, k))
            err_2.append(np.sum(err_1) / diffSet_num)
            diffSetIndex = 0
            del err_1[:]
        k = 0
        diffSet_num = 0
        hash_table[z] = err_2[-1]
        del err_2[:]

    # 按值对哈希表进行排序
    sorted_items = sorted(hash_table.items(), key=lambda x: x[1])

    # 输出最小值和次小值对应的键
    if len(sorted_items) >= 1:
        min_key = sorted_items[0][0]
        print("Minimum key:", min_key)

    if len(sorted_items) >= 2:
        second_min_key = sorted_items[1][0]
        print("Second minimum key:", second_min_key)
    print(hash_table)


def pos_crrect(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, algorithm,
               object, lis, r):
    # size of probe and diffraction patterns
    ysize, xsize = probe.shape
    positions = positions.astype(float)
    # initialize object. make sure it can hold complex numbers
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    objectFunc_last = np.ones(objectSize, dtype=np.complex64)
    # initalize that illuminated part of the object
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)

    # initialize algorithm wave fields (fourier and real)

    g = np.zeros((ysize, xsize), dtype=np.complex64)
    gprime = np.zeros((ysize, xsize), dtype=np.complex64)
    G = np.zeros((ysize, xsize), dtype=np.complex64)
    Gprime = np.zeros((ysize, xsize), dtype=np.complex64)

    # define iteration counter for outer loop
    k = 0
    pos_iter = 1
    pr_iter = 0
    delta = 1 * np.ones(shape=(len(positions), 1), dtype=float)
    pos_shift_dir = np.zeros(shape=(len(positions), 2), dtype=float)
    shift_max = 0.5
    # figure for animation
    # fig = plt.figure()

    # Initialize vector for animation data
    ims = []

    # idex for iterating through the diffraction patterns
    diffSetIndex = 0
    diffSet_num = 0
    # initialize vector for error calculation
    err_1 = []
    err_2 = [1]
    err_3 = [1]
    # Start of ePIE iterations
    while k < n:
        # Start of inner loop: (where you iterate through all probe positions R)
        # for pos in positions:
        for pos in lis:
            # Cut out the part of the image that is illuminated at R(=(ypos,xpos)
            objectIlluminated = objectFunc[round(positions[pos][0]) + illu_indy, round(positions[pos][1]) + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # fft the wave field at position R to Fourier space
            G = Propagate(g, type, pix, wavelength, z)

            # make |PSI| confirm with the diffraction pattern from R
            Gprime = np.sqrt(diffSet[pos]) * np.exp(1j * np.angle(G))

            # inverse Fourier transform
            gprime = Propagate(Gprime, type, pix, wavelength, -z)

            # update the TOTAL object function with the illuminated part
            # The update should be the differens of the last iteration and the new one
            objectIlluminated_last = objectIlluminated
            alpha = 1  # higher value == faster change
            objectIlluminated = objectIlluminated + alpha * (
                    gprime - g) * np.conj(probe) / (np.max(
                abs(probe)) ** 2)  # probe* annars blir det att man delar med massa nollor
            # objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated + alpha * (gprime - g) * np.conj(
            #       probe)  # probe* annars blir det att man delar med massa nollor
            # update probe function
            beta = 0.02  # higher value == faster change
            if k > pr_iter:
                probe = probe + beta * (gprime - g) * np.conj(objectIlluminated) / (np.max(abs(objectIlluminated)) ** 2)
            objectFunc[round(positions[pos][0]) + illu_indy, round(positions[pos][1]) + illu_indx] = objectIlluminated

            if k > pos_iter:
                stat_x = round(ysize // 2 - r)
                end_x = round(ysize // 2 + r)

                out, grey = dftregistration(objectIlluminated_last[stat_x:end_x, stat_x:end_x],
                                            objectIlluminated[stat_x:end_x, stat_x:end_x], r=100)
                positions[pos][0] = positions[pos][0] + delta[pos] * out[2]
                positions[pos][1] = positions[pos][1] + delta[pos] * out[3]
                pos_new = np.array([out[2], out[3]])
                pos_shift_dir_tmp = np.array([pos_shift_dir[pos][0], pos_shift_dir[pos][1]])
                # if np.dot(pos_new.flatten(), pos_shift_dir_tmp.flatten()) > 0.3 * np.dot(pos_new.flatten(),pos_new.flatten()):
                #     delta[pos] = delta[pos]*1.1
                # elif np.dot(pos_new.flatten(),pos_shift_dir_tmp.flatten()) < -0.3 * np.dot(pos_new.flatten(),pos_new.flatten()):
                #     delta[pos] = delta[pos]*0.9
                if out[2] > 1 or out[3] > 1:
                    positions[pos][0] += out[2]
                    positions[pos][1] += out[3]
                else:
                    # print(positions[pos][1])
                    pos_shift_dir[pos][0] = 3 * out[2]
                    pos_shift_dir[pos][1] = 3 * out[3]
                pos_shift_dir[pos][0] = out[2]
                pos_shift_dir[pos][1] = out[3]
            # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]**2) - abs(G)**2 ))**2))  #intensity err
            err_1.append(np.sum(abs(diffSet[pos] - abs(G) ** 2) ** 2))  # amplitude err     #amplitude err
            # caculate diffraction image sum
            if k == 0:
                #   #diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
                diffSet_num += np.sum(diffSet[pos] ** 2)
            ########################
            # Apply further constraints:
            # These 2 constraints are for transmission
            ########################

            # constrain object amplitude to 1
            # temp_Oamp = abs(objectFunc)
            # temp_Oamp[temp_Oamp>1] = 1
            # temp = np.angle(objectFunc)
            # objectFunc = temp_Oamp * np.exp(1j* temp)
            #
            # #constraint object phase to negative or 0
            # temp_Ophase = np.angle(objectFunc)
            # temp_Ophase[temp_Ophase>0] = 0
            # objectFunc = abs(objectFunc) * np.exp(1j* temp_Ophase)
            #
            # animate
            # im = plt.imshow(np.angle(objectFunc))

            # ims.append([im])

            diffSetIndex += 1

            # err_3.append(relative_err(object, objectFunc,250))
        k += 1
        print('Iteration %d starts' % k)

        # reset inner loop index
        diffSetIndex = 0
        err_2.append(np.sum(err_1) / diffSet_num)
        #
        del err_1[:]

    # psnr_am, psnr_ph, ssim_am, ssim_ph = complex_PNSR(object, objectFunc)
    # print(algorithm,'振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph)
    if algorithm == 'ePIE':
        plt.figure()
        plt.semilogy(err_2, linewidth=1.5, color='green', label=algorithm, alpha=0.7)
        plt.legend()
        plt.plot(err_2, color='green')
        plt.xlabel('iteration number')
        plt.ylabel('relative residual norm(res)')
        print(err_2[-1])
        # End of iterations
        print('End of iterations')
    print(positions)
    return objectFunc, probe, err_2[-1]


def sigle_iter(diffSet, probe, objectFunc, positions, illu_indy, illu_indx, type, z, wavelength, pix, lis):
    for pos in lis:
        # Cut out the part of the image that is illuminated at R(=(ypos,xpos)
        objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]

        # Guessed wave field from the object at position R
        g = objectIlluminated * probe

        # fft the wave field at position R to Fourier space
        G = Propagate(g, type, pix, wavelength, z)

        # make |PSI| confirm with the diffraction pattern from R
        Gprime = np.sqrt(diffSet[pos]) * np.exp(1j * np.angle(G))

        # inverse Fourier transform
        gprime = Propagate(Gprime, type, pix, wavelength, -z)

        # update the TOTAL object function with the illuminated part
        # The update should be the differens of the last iteration and the new one

        alpha = 1  # higher value == faster change
        objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = objectIlluminated + alpha * (
                gprime - g) * np.conj(
            probe) / (np.max(abs(probe)) ** 2)  # probe* annars blir det att man delar med massa nollor

        # update probe function
        beta = 0.02  # higher value == faster change
        probe = probe + beta * (gprime - g) * np.conj(objectIlluminated) / (np.max(abs(objectIlluminated)) ** 2)
        return objectFunc, probe


if __name__ == '__main__':
    print('main prog')
