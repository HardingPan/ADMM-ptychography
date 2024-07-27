import numpy as np
from util.Propagate import Propagate
import matplotlib.pyplot as plt
from util.relative_err import relative_err
import regularization
from util.evalution import complex_PNSR
from util.subpixel_registration import dftregistration
from FFDNet_1.models import FFDNet
from util.center_find import  center_mass_caculate, center_mass_correct
from util import Diff, matrix_diffuse
from  math import pi


def wf(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object, lis):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 0
    err_2 = [1]
    probe_2 = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    probe_N_2 = probe
    objectFunc_2 = objectFunc
    alhpa = 0.05
    sigma = 0.5
    while k < n:
        #

        temp = np.ones(shape=(ysize, xsize))
        # for pos in positions:
        for pos in lis:

            objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # update object function
            miu = Propagate(g, type, pix, wavelength, z)

            miu_2 = (miu - miu * np.sqrt(diffSet[pos]) / (abs(miu)))
            # #
            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z)

            objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = objectIlluminated - 2 * np.conj(
                probe) * miu_3 / (abs((probe)) ** 2 + 2)
            probe_1 = miu_3 * np.conj(objectIlluminated)
            temp += abs(objectIlluminated) ** 2
            # probe =  probe - 2*probe_1
            probe_2 += probe_1
            # if k<100:
            # probe = probe - 2*probe_1/(abs((objectIlluminated))**2+1)

            # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]**2) - abs(miu)**2 ))**2))  #intensity err
            err_1.append(np.sum(abs(diffSet[pos] - abs(miu) ** 2) ** 2))  # amplitude err
            # caculate diffraction image sum
            if k == 0:
                #     # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
                diffSet_num += np.sum(diffSet[pos] ** 2)

            diffSetIndex += 1
        # if k>= 100:
        #
        probe = probe - 2 * probe_2 / (np.max(temp))
        probe_2 = 0
        k += 1
        probe = np.where(abs(probe) > 3, 3, abs(probe)) * Diff.sign(probe)
        objectFunc = np.where(abs(objectFunc) > 3, 3, abs(objectFunc)) * Diff.sign(objectFunc)

        #####*** nesterov  apart ##########
        if k > 1:
            probe = probe + (k / (k + 3)) * (probe - probe_N)
            objectFunc = objectFunc + (k / (k + 3)) * (objectFunc - objectFunc_N)

        objectFunc_N = objectFunc
        probe_N = probe

        # err_3.append(relative_err(object, objectFunc,pixsum=250))

        print('Iteration %d starts' % k)

        # reset inner loop index
        diffSetIndex = 0
        err_2.append(np.sum(err_1) / diffSet_num)

        del err_1[:]

    psnr_am, psnr_ph,ssim_am , ssim_ph = complex_PNSR(object, objectFunc)
    print('wf振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph,'\n振幅ssim=',ssim_am,' 相位ssim=',ssim_ph )
    plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='black', label='WF', alpha=0.7)
    plt.legend()
    plt.plot(err_2, color='black')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')
    print(err_2[-1])
    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3[-1]


def wf_nosie(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object, lis):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    n1, n2 = objectFunc.shape
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 0
    err_2 = [1]

    probe_2 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)

    noise = np.zeros(shape=(ysize, xsize), dtype=np.float)
    noise2 = np.zeros(shape=(ysize, xsize), dtype=np.float)
    while k < n:

        # for pos in positions:
        #     objectIlluminated = objectFunc[pos[0] + illu_indy, pos[1] + illu_indx]
        #
        #     # Guessed wave field from the object at position R
        #     g = objectIlluminated * probe
        #
        #     # update object function
        #     miu = Propagate(g, type, pix, wavelength, z)
        #
        #     miu_2 = miu - miu * np.sqrt(diffSet[diffSetIndex]) / abs(miu)
        #
        #     miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)
        #
        #     objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated - 2* miu_3
        #
        #     # probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z)* np.conj(objectIlluminated)
        #     # probe = probe - 0.02 * probe_1
        #     probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(objectIlluminated)
        #     probe_2 += probe_1
        #
        #
        #     # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]) - abs( miu)**2)) ** 2))
        #
        #     # if k == 0:
        #     #  # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
        #     #  diffSet_num += np.sum(abs(diffSet[diffSetIndex]) ** 2)
        #
        #     diffSetIndex += 1
        # probe = probe - 2 * probe_2 / len(positions)
        # probe_2 = 0
        # k += 1
        for pos in lis:

            objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # update object function
            miu = Propagate(g, type, pix, wavelength, z)
            # temp = diffSet[pos] - noise2**2
            # miu_2 = miu - miu * (np.sqrt(np.where(temp < 0, 0, temp))) / abs(miu)
            miu_2 = miu - miu * (np.sqrt(diffSet[pos]) - noise2) / abs(miu)
            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)

            objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = objectIlluminated - 2 * miu_3 / (
                    (probe) ** 2 + 1)

            probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(objectIlluminated)
            # noise[pos, :, :] = noise[pos, :, :] - 0.1*(np.sqrt(diffSet[pos]) - abs(miu) + noise[pos, :, :])
            if k > 30:
                # noise =  noise + np.sqrt(np.where(diffSet[pos]-abs(miu)**2 < 0, 0,diffSet[pos]-abs(miu)**2))

                noise += (np.sqrt(diffSet[pos]) - abs(miu))

            probe_2 += (probe_1 / ((objectIlluminated) ** 2 + 1))
            # probe_2 += probe_1
            # if k<100:
            # probe = probe - 0.2*probe_1

            # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]**2) - abs(miu)**2 ))**2))  #intensity err
            err_1.append(np.sum(abs(diffSet[diffSetIndex] - abs(miu) ** 2) ** 2))  # amplitude err
            # caculate diffraction image sum
            if k == 0:
                # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
                diffSet_num += np.sum(abs(diffSet[diffSetIndex]) ** 2)

            diffSetIndex += 1
            # if k>= 100:
            #
        probe = probe - 2 * probe_2 / len(positions)
        noise2 = noise / len(positions)
        noise2 = np.where(noise2 < 0, 0, noise2)

        noise = 0
        probe_2 = 0
        k += 1

        # nesterov  apart ##########

        if k > 1:
            probe = probe + (k / (k + 3)) * (probe - probe_N)
            objectFunc = objectFunc + (k / (k + 3)) * (objectFunc - objectFunc_N)

        objectFunc_N = objectFunc
        probe_N = probe

        # err_3.append(relative_err(object, objectFunc,pixsum=250))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        err_2.append(np.sum(err_1) / diffSet_num)
        del err_1[:]
    #
    psnr_am, psnr_ph, ssim_am, ssim_ph = complex_PNSR(object, objectFunc)
    print('wf_denoise振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph)
    plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='gray', label='WF_denoise', alpha=0.7)
    plt.legend()
    plt.plot(err_2, color='gray')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')

    # End of iterations
    print(err_2[-1])
    print('End of iterations')
    return objectFunc, probe, err_2[-1]


def wf_Nesterov(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 0
    err_2 = [1]
    probe_2 = probe
    objectFunc_2 = objectFunc
    miu = np.zeros((ysize, xsize), dtype=np.complex64)
    while k < n:

        for pos in positions:
            objectIlluminated = objectFunc[pos[0] + illu_indy, pos[1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # update object function
            miu = Propagate(g, type, pix, wavelength, z)
            out = np.where(np.abs(miu) > 0, miu / np.abs(miu), 0)
            miu_2 = miu - out * np.sqrt(diffSet[diffSetIndex])

            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)

            objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated - 1 * miu_3

            probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z)
            probe = probe - 1 * probe_1 * np.conj(objectIlluminated)

            # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]) - abs( miu)**2)) ** 2))
            #
            # if k == 0:
            #  # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
            #  diffSet_num += np.sum(abs(diffSet[diffSetIndex]) ** 2)

            diffSetIndex += 1

        k += 1

        # #####*** nesterov  apart ##########
        if k > 1:
            probe = probe + (k / (k + 3)) * (probe - probe_N)
            objectFunc = objectFunc + (k / (k + 3)) * (objectFunc - objectFunc_N)

        objectFunc_N = objectFunc
        probe_N = probe

        err_3.append(relative_err(object, objectFunc, 100))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        # err_2.append(np.sum(err_1) / diffSet_num)
        # del err_1[:]

    psnr_am, psnr_ph = complex_PNSR(object, objectFunc)
    print('wf_Nesterov振幅pnsr=', psnr_am, '\n相位pnsr=', psnr_ph)
    plt.figure()
    plt.semilogy(err_3, linewidth=1.5, color='lightgreen', label='WF_Nesterov', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.01, 0.2, 0))
    plt.plot(err_3, color='lightgreen')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')

    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3[-1]


def wf_global(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object, lis):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)

    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 0
    err_2 = [1]

    alhpa = 0.05

    while k < n:

        objectFunc_gr = np.zeros(objectSize, dtype=np.complex64)
        temp1 = np.zeros(shape=(ysize, xsize))
        temp2 = np.zeros(objectSize)
        probe_2 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        for pos in lis:
            objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # update object function
            miu = Propagate(g, type, pix, wavelength, z)
            miu_2 = miu - miu * np.sqrt(diffSet[pos]) / (abs(miu))

            #
            # probe_1=0.5 * Propagate(miu_2, type, pix, wavelength, -z)
            # probe = probe - 0.2*probe_1*np.conj(objectIlluminated)
            probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(objectIlluminated)
            probe_2 += probe_1
            temp1 += abs(objectIlluminated) ** 2

        probe = probe - 2 * probe_2 / (np.max(temp1))
        for pos in lis:
            objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
            # Guessed wave field from the object at position R
            g = objectIlluminated * probe
            # update object function
            miu = Propagate(g, type, pix, wavelength, z)
            miu_2 = miu - miu * np.sqrt(diffSet[pos]) / (abs(miu))
            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)
            objectIlluminated_gr = miu_3
            objectFunc_gr[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += objectIlluminated_gr
            temp2[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += abs(probe) ** 2
            if k == 0:
                # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
                #diffSet_num += np.sum(abs(diffSet[pos]) ** 2)
                diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])))
            #err_1.append(np.sum((abs(abs(diffSet[pos]) - abs(miu) ** 2)) ** 2))
            err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(miu)))))
        objectFunc = objectFunc - 2 * objectFunc_gr / (np.max(temp2) )
        #
        # if k > 1:
        #     probe = probe + (k / (k + 3)) * (probe - probe_N)
        #     objectFunc = objectFunc + (k / (k + 3)) * (objectFunc - objectFunc_N)
        objectFunc_N = objectFunc
        probe_N = probe
        probe = np.where(abs(probe) > 1e8, 1e8, abs(probe)) * Diff.sign(probe)
        objectFunc = np.where(abs(objectFunc) > 1e8, 1e8, abs(objectFunc)) * Diff.sign(objectFunc)
        k += 1
        # err_3.append(relative_err(object, objectFunc, 250))
        print('Iteration %d starts' % k)

        err_2.append(np.sum(err_1) / diffSet_num)
        del err_1[:]
    psnr_am, psnr_ph, ssim_am, ssim_ph = complex_PNSR(object, objectFunc)
    print('wf_globa振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph)

    plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='orange', label='WF_global', alpha=0.7)
    plt.legend()
    plt.plot(err_2, color='orange')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')
    # End of iterations
    print('End of iterations')
    print(err_2[-1])
    return objectFunc, probe, err_3[-1]


def wf_red_net(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 0
    err_2 = [1]

    regula = regularization.denosise
    lamda = 0.001
    delta = 0.01
    model = FFDNet(num_input_channels=1)
    # opt = parse_options(is_train=False)
    # model = create_model(opt)
    while k < n:
        for pos in positions:
            objectIlluminated = objectFunc[pos[0] + illu_indy, pos[1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # update object function
            miu = Propagate(g, type, pix, wavelength, z)

            miu_2 = miu - miu * np.sqrt(diffSet[diffSetIndex]) / abs(miu)
            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)

            objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated - 2 * miu_3

            probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z)
            probe = probe - 0.2 * probe_1 * np.conj(objectIlluminated)

            # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]) - abs( miu)**2)) ** 2))

            # if k == 0:
            #  # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
            #  diffSet_num += np.sum(diffSet[diffSetIndex] ** 2)

            diffSetIndex += 1

        objectFunc = objectFunc - lamda * (
                    objectFunc - regula.FFDNetdeblur(objectFunc, model=model, sigma=0.25))  # 正则化项

        k += 1
        err_3.append(relative_err(object, objectFunc, pixsum=230))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        # err_2.append(np.sum(err_1) / diffSet_num)
        # del err_1[:]

    psnr_am, psnr_ph, ssim_am, ssim_ph = complex_PNSR(object, objectFunc)
    print('wf_red_net振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph)
    plt.figure()
    plt.semilogy(err_3, linewidth=1.5, color='slategray', label='wf_red_net', alpha=0.7)
    plt.legend()
    plt.plot(err_3, color='slategray')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')

    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3[-1]


def wf_epie(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 0
    err_2 = [1]

    miu = np.zeros((ysize, xsize), dtype=np.complex64)
    while k < n:

        for pos in positions:
            objectIlluminated = objectFunc[pos[0] + illu_indy, pos[1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe
            G = Propagate(g, type, pix, wavelength, z)
            Gprime = np.sqrt(diffSet[diffSetIndex]) * np.exp(1j * np.angle(G))
            gprime = Propagate(Gprime, type, pix, wavelength, -z)
            # update object function
            miu = Propagate(g, type, pix, wavelength, z)
            miu_2 = miu - miu * np.sqrt(diffSet[diffSetIndex]) / (abs(miu))
            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)

            objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated - 2 * miu_3

            probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z)
            probe = probe + 1 * (gprime - g) * np.conj(objectIlluminated) / (np.max(abs(objectIlluminated)) ** 2)

            # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]) - abs( miu)**2)) ** 2))
            #
            # if k == 0:
            #  # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
            #  diffSet_num += np.sum(abs(diffSet[diffSetIndex]) ** 2)

            diffSetIndex += 1
        k += 1

        err_3.append(relative_err(object, objectFunc, 100))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        # err_2.append(np.sum(err_1) / diffSet_num)
        # del err_1[:]

    psnr_am, psnr_ph = complex_PNSR(object, objectFunc)
    print('wf_epie振幅pnsr=', psnr_am, '\n相位pnsr=', psnr_ph)
    # plt.figure()
    plt.semilogy(err_3, linewidth=1.5, color='peru', label='WF_epie', alpha=0.7)
    plt.legend()
    plt.plot(err_3, color='peru')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')

    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3[-1]


def rwf(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object, lis, is_center_correct = 1):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 0
    err_2 = [1]
    beta = 0.8
    alhpa = 0.05
    beta_2 = 0.9
    probe_2 = probe
    objectFunc_2 = objectFunc
    probe_2 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)

    while k < n:
        # if k != 0:
        #     probe = probe_N
        #     objectFunc = objectFunc_N

        for pos in lis:
            objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # update object function
            miu = Propagate(g, type, pix, wavelength, z)
            # w = abs(miu)**2/(abs(miu)**2+beta*diffSet[diffSetIndex])
            # w1 = 1/(alhpa*abs(probe)+(1-alhpa)*np.max(abs(probe)))
            w1 = 1 / ((1 - alhpa) * abs(probe) ** 2 + alhpa * np.max(abs(probe)) ** 2)
            # w2 = abs(miu)**2/ (abs(miu)**2+beta*diffSet[diffSetIndex])
            # w3 =  1/ ( beta_2*np.max(abs(objectIlluminated)) ** 2+(1- beta_2)*abs(objectIlluminated)**2)

            miu_2 = miu - miu * np.sqrt(diffSet[pos]) / abs(miu)
            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)

            objectFunc[
                positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = objectIlluminated - 2 * miu_3 * w1

            # probe_1 = 0.5 * Propagate(miu_2 , type, pix, wavelength, -z)
            # probe = probe -0.2*probe_1*np.conj(objectIlluminated)
            probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(objectIlluminated)
            probe = probe - 0.2 * probe_1

            # err_1.append(np.sum((abs(abs(diffSet[pos]) - abs( miu)**2)) ** 2))
            #
            #
            # if k == 0:
            #  # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
            #  diffSet_num += np.sum(diffSet[pos] ** 2)
            #
            # diffSetIndex += 1
        # probe = probe - 2 * probe_2 / len(positions)
        # probe_2 = 0
        k += 1
        if is_center_correct == 1:
            # 探针位置纠正
            x_shift, y_shift = center_mass_caculate(probe)
            probe = center_mass_correct(probe, x_shift, y_shift)
            objectFunc = center_mass_correct(objectFunc, x_shift, y_shift)
        # probe_N = probe + (k / (k + 3)) * (probe - probe_2)
        # objectFunc_N = objectFunc + (k / (k + 3)) * (objectFunc - objectFunc_2)/2
        # probe_2 = probe
        # objectFunc_2 = objectFunc

        err_3.append(relative_err(object, objectFunc, 250))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        err_2.append(np.sum(err_1) / diffSet_num)
        del err_1[:]

    psnr_am, psnr_ph, ssim_am, ssim_ph = complex_PNSR(object, objectFunc)
    print('rwf振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph)
    plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='purple', label='RWF', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.plot(err_2, color='purple')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')
    print(err_2[-1])
    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_2[-1]


def rwf_mean(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 0
    err_2 = [1]
    beta = 0.1
    alhpa = 0.05
    beta_2 = 0.9
    miu = np.zeros((ysize, xsize), dtype=np.complex64)
    regula = regularization.denosise
    lamda = 0.02
    delta = 0.001
    while k < n:
        for pos in positions:
            objectIlluminated = objectFunc[pos[0] + illu_indy, pos[1] + illu_indx]

            # Guessed wave field from the object at position R
            g = objectIlluminated * probe

            # update object function
            miu = Propagate(g, type, pix, wavelength, z)
            # w = abs(miu)**2/(abs(miu)**2+beta*diffSet[diffSetIndex])
            # w1 = 1/(alhpa*abs(probe)+(1-alhpa)*np.max(abs(probe)))
            w1 = 1 / ((1 - alhpa) * abs(probe) ** 2 + alhpa * np.max(abs(probe)) ** 2)
            w2 = abs(miu) ** 2 / (abs(miu) ** 2 + beta * diffSet[diffSetIndex])
            w3 = 1 / (beta_2 * np.max(abs(objectIlluminated)) ** 2 + (1 - beta_2) * abs(objectIlluminated) ** 2)

            miu_2 = miu - miu * np.sqrt(diffSet[diffSetIndex]) / abs(miu)
            miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z) * np.conj(probe)

            objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated - 2 * w1 * miu_3

            probe_1 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z)
            probe = probe - 2 * probe_1 * np.conj(objectIlluminated)

            probe = probe - lamda * (probe - regula.tvdeblurr(probe, 3))

            # err_1.append(np.sum((abs(abs(diffSet[diffSetIndex]) - abs( miu)**2)) ** 2))

            # if k == 0:
            #  # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
            #  diffSet_num += np.sum(diffSet[diffSetIndex] ** 2)

            diffSetIndex += 1
        objectFunc = objectFunc - lamda * (objectFunc - regula.tvdeblurr(objectFunc, 3))
        k += 1
        err_3.append(relative_err(object, objectFunc, 130))
        print('Iteration %d starts' % k)
        diffSetIndex = 0

        # err_2.append(np.sum(err_1) / diffSet_num)
        # del err_1[:]

    psnr_am, psnr_ph, ssim_am, ssim_ph = complex_PNSR(object, objectFunc)
    print('wf振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph)
    # plt.figure()
    plt.semilogy(err_3, linewidth=1.5, color='blue', label='RWF_mean', alpha=0.7)
    plt.legend()
    plt.plot(err_3, color='blue')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')

    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3[-1]


def ADMM(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object, lis, is_center_correct):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)

    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k =1
    err_2 = [1]
    probe_2 = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    probe_N_2 = probe
    objectFunc_2 = objectFunc

    belta = 0.2
    s = np.ones(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    x = np.zeros(shape=(len(positions), ysize, xsize))
    s1 = 1e-6
    s2 = 1e-6
    r1 = 1e-6
    r2 = 1e-3
    for pos in lis:
        objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
        g = objectIlluminated * probe
        s[pos, :, :] = Propagate(g, type, pix, wavelength, z)
        x[pos, :, :] = abs(s[pos, :, :])

    while k < n:

        temp1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp1_1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp2 = np.zeros(objectSize, dtype=np.complex64)
        temp2_1 = np.zeros(objectSize, dtype=np.complex64)
        # if k  < 200:
        #     belta = 0.01
        # else:
        #     belta = 1
        # 探针更新
        for pos in lis:
            # miu1 = s[pos, :, :] - Propagate(
            #     objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix, wavelength,
            #     z)

            miu2 = Propagate(s[pos, :, :] + aux[pos, :, :], type, pix, wavelength, -z)
            temp1 += miu2 * np.conj(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx])
            temp1_1 += abs(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) ** 2

        h1 = np.max(temp1_1)
        if h1 < s1:
            m1 = s1
        else:
            m1 = r1 * h1
        probe = ( temp1 + m1*probe) / (temp1_1 + m1)
        probe = np.where(abs(probe) > 1e+8, 1e+8, abs(probe)) * Diff.sign(probe)

        # if k % 100 == 0:
        #     probe = center_mass_correct(probe)
        # 物体更新
        for pos in lis:
            #
            miu = Propagate(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix,
                            wavelength, z)
            # miu1 = s[pos, :, :] - miu
            #err_1.append(np.sum((abs(abs(diffSet[pos]) - abs(miu) ** 2)) ** 2))
            err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(miu)))))

            miu2 = Propagate(s[pos, :, :] + aux[pos, :, :], type, pix,
                             wavelength, -z)
            temp2[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += miu2 * np.conj(probe)
            temp2_1[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += abs(probe) ** 2
            ##误差计算
            if k == 1:
                #diffSet_num += np.sum(abs(diffSet[pos]) ** 2)
                diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])) )
        h2 = np.max(temp2_1)
        if h2 < s2:
            m2 = s2
        else:
            m2 = r2 * h2
        objectFunc = (temp2 + m2*objectFunc) / (temp2_1 + m2)
        objectFunc = np.where(abs(objectFunc) > 1e+8, 1e+8, abs(objectFunc)) * Diff.sign(objectFunc)
        ##位置矫正
        if is_center_correct == 1 :
         x_shift, y_shift = center_mass_caculate(probe)
         probe = center_mass_correct(probe, x_shift, y_shift)
         objectFunc = center_mass_correct(objectFunc, x_shift, y_shift)
        #辅助变量对偶变量更新
        for pos in lis:
            miu = Propagate(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix,
                            wavelength, z)

            x[pos, :, :] = x[pos, :, :] - 0.5* (
                        (1 + belta - np.sqrt(diffSet[pos] + 1e-8 *np.max(diffSet[pos])) / np.sqrt(abs(x[pos, :, :])**2 + 1e-8*np.max(diffSet[pos]))) * x[pos, :, :] - belta * abs(
                    miu - aux[pos, :, :]))
            x[pos, :, :] = np.where(x[pos, :, :] < 0, 0, x[pos, :, :])
            s[pos, :, :] = x[pos, :, :] * Diff.sign(miu - aux[pos, :, :])
            aux[pos, :, :] = aux[pos, :, :] + s[pos, :, :] - miu
        # if k > 2:
        #  err_3.append(relative_err(object, objectFunc, 250))
        print(belta)
        k += 1
        print('Iteration %d starts' % k)

        # reset inner loop index
        diffSetIndex = 0
        err_2.append(np.sum(err_1) / diffSet_num)

        del err_1[:]
    # out1, objectFunc = dftregistration(object, objectFunc, r=50)
    # psnr_am, psnr_ph, ssim_am, ssim_ph, RMSE = complex_PNSR(object, objectFunc)
    # print('ADMM振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph, '\n相位RMSE=', RMSE)
    # plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='grey', label='ADMM', alpha=0.7, marker='^', markevery=30)
    plt.legend(loc=3, prop={'size': 9})
    plt.xlabel('iteration number')
    plt.ylabel('relative error')
    plt.savefig('admm_loss.png', dpi=300)
    print(err_2[-1], err_3[-1])
    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3


def L_ADMM(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object, lis, is_center_correct):

    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)

    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 1
    err_2 = [1]
    probe_2 = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    probe_N_2 = probe
    objectFunc_2 = objectFunc

    belta = 1e-9
    s = np.ones(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    x = np.zeros(shape=(len(positions), ysize, xsize))
    s_last = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux_last =  np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    f = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)

    for pos in lis:
        objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
        g = objectIlluminated * probe
        s[pos, :, :] = g
        x[pos, :, :] = abs(s[pos, :, :])
        s_last[pos, :, :] =s[pos, :, :]
    while k < n:
        # if k>50:
        #     belta = 0.1
        temp1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp1_1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp2 = np.zeros(objectSize, dtype=np.complex64)
        temp2_1 = np.zeros(objectSize, dtype=np.complex64)
        is_adp = 0
        # 探针更新
        for pos in lis:
            # miu1 = s[pos, :, :] - Propagate(
            #     objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix, wavelength,
            #     z)
            #
            miu2 = probe * abs(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) ** 2  - \
                   np.conj(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) * (s[pos, :, :] - aux[pos, :, :])
            # miu2 = np.conj(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) * (
            #                    s[pos, :, :] - aux[pos, :, :])
            temp1 += miu2
            temp1_1 += abs(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) ** 2

        probe = probe - 0.9 * temp1 / np.max(temp1_1)
        #probe = temp1 / (temp1_1 + 1e-5)
        probe = np.where(abs(probe) > 1e+8, 1e+8, abs(probe)) * Diff.sign(probe)

        # 物体更新
        for pos in lis:
            #
            miu = Propagate(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix,
                            wavelength, z)
            # miu1 = s[pos, :, :] - miu
            err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(miu)))))
            #err_1.append(np.sum((abs(abs(diffSet[pos]) - abs(miu) ** 2)) ** 2))
            miu2 = abs(probe)**2 * objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] -\
                   np.conj(probe)*(s[pos, :, :] - aux[pos, :, :])
            #miu2 =  np.conj(probe) * (s[pos, :, :] - aux[pos, :, :])
            temp2[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += miu2
            temp2_1[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += abs(probe) ** 2
            ##误差计算
            if k == 1:

                #diffSet_num += np.sum(abs(diffSet[pos])**2)
                diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])) )

        objectFunc = objectFunc - 0.9 * temp2 / np.max(temp2_1)
        #objectFunc = temp2 /(temp2_1 + 1e-5)
        objectFunc = np.where(abs(objectFunc) > 1e+8, 1e+8, abs(objectFunc)) * Diff.sign(objectFunc)
        ##位置矫正
        if is_center_correct == 1:
            x_shift, y_shift = center_mass_caculate(probe)
            probe = center_mass_correct(probe, x_shift, y_shift)
            objectFunc = center_mass_correct(objectFunc, x_shift, y_shift)


        for pos in lis:

            miu = Propagate(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe + aux[pos,:,:]  , type, pix,
                            wavelength, z)
            f[pos,:,:] = np.sqrt(diffSet[pos]) * miu / abs(miu)
            miu2 = (f[pos,:,:] + belta * miu) / (1 + belta)
            s[pos, :, :] = Propagate(miu2, type, pix, wavelength, -z)
            aux[pos, :, :] = aux[pos, :, :] + objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe - s[pos, :, :]
            if np.sqrt(np.sum(abs(s[pos, :, :] - s_last[pos, :, :])**2)) > 1e-2 / k or\
               np.sqrt(np.sum(abs(aux[pos, :, :] - aux_last[pos, :, :]) ** 2)) > 1e-2 / k or \
               np.sqrt(np.sum(abs(aux[pos, :, :]) ** 2)) > 1e+8 :

                is_adp = 1
        if is_adp ==1 :
            belta = belta * 1.1
        # if k > 100:
        #     belta = 1
        print(belta)
        s_last = s.copy()
        aux_last =aux.copy()

        #
        # probe = np.where(abs(probe) > 10 ** 8, 10 * 8, abs(probe)) * Diff.sign(probe)
        # objectFunc = np.where(abs(objectFunc) > 10 * 8, 10 * 8, abs(objectFunc)) * Diff.sign(objectFunc)

        k += 1
        print('Iteration %d starts' % k)

        # reset inner loop index
        diffSetIndex = 0
        # if k > 2:
        #  err_3.append(relative_err(object, objectFunc, 250))
        err_2.append(np.sum(err_1) / diffSet_num)

        del err_1[:]
    # out,  objectFunc = dftregistration(object, objectFunc, r=50)
    # objectFunc = abs(imshift(abs(objectFunc), out[2], out[3])) * np.exp(1j * imshift(np.angle(objectFunc), out[2], out[3]))
    # psnr_am, psnr_ph, ssim_am, ssim_ph, RMSE = complex_PNSR(object, objectFunc)
    # print('LADMM振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph, '\n相位RMSE=', RMSE)
    # plt.figure()
    plt.semilogy(err_2 ,linewidth=1.5, color='blue', label='LADMM', alpha=0.7, marker='o',markevery=30)
    plt.legend(loc=3, prop={'size': 9})
    plt.xlabel('iteration number')
    plt.ylabel('relative error')
    plt.savefig('l_admm_loss.png', dpi=300)
    print(err_2[-1],err_3[-1])
    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3

def T_ADMM(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object, lis, is_center_correct ):
    ysize, xsize = probe.shape
    # probe = probe_r.copy()
    objectFunc = np.ones(objectSize,dtype=np.complex64)
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 1
    err_2 = [1]
    probe_2 = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    probe_N_2 = probe
    objectFunc_2 = objectFunc

    belta = 1e-9
    s = np.ones(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    x = np.zeros(shape=(len(positions), ysize, xsize))
    s_last = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux_last =  np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)

    f = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    probe_mask = len(lis)*matrix_diffuse.create_mask_matrix(ysize, xsize, r=0.6*ysize)
    for pos in lis:
        objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
        g = objectIlluminated * probe
        s[pos, :, :] = g
        x[pos, :, :] = abs(s[pos, :, :])
        s_last[pos, :, :] =s[pos, :, :]
    is_tr = 0
    while k < n:
        # if k>50:
        #     belta = 0.1
        temp1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp1_1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp2 = np.zeros(objectSize, dtype=np.complex64)
        temp2_1 = np.zeros(objectSize, dtype=np.complex64)
        is_adp = 0

        # 探针更新
        for pos in lis:

            miu2 = np.conj(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) * (
                               s[pos, :, :] - aux[pos, :, :])
            temp1 += miu2
            temp1_1 += abs(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) ** 2

        #probe = probe - 0.9 * temp1 / np.max(temp1_1)
        probe = temp1 / (temp1_1 + 1e-9)
        probe = np.where(abs(probe) > 1e+8, 1e+8, abs(probe)) * Diff.sign(probe)

        # 物体更新
        for pos in lis:
            #
            miu = Propagate(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix,
                            wavelength, z)
            # miu1 = s[pos, :, :] - miu
            err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(miu)))))
            #err_1.append(np.sum((abs(abs(diffSet[pos]) - abs(miu) ** 2)) ** 2))

            miu2 =  np.conj(probe) * (s[pos, :, :] - aux[pos, :, :])
            temp2[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += miu2
            temp2_1[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += abs(probe) ** 2
            ##误差计算
            if k == 1:

                diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])) )
               # diffSet_num += np.sum(abs(diffSet[pos]) ** 2)


        objectFunc = (temp2) /(temp2_1 + 1e-2)
        objectFunc = np.where(abs(objectFunc) > 1e+8, 1e+8, abs(objectFunc)) * Diff.sign(objectFunc)

        if is_center_correct == 1:
            # 探针位置纠正
            x_shift, y_shift = center_mass_caculate(probe)
            probe = center_mass_correct(probe, x_shift, y_shift)
            objectFunc = center_mass_correct(objectFunc, x_shift, y_shift)


        for pos in lis:

            miu = Propagate(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe + aux[pos,:,:] , type, pix,
                            wavelength, z)
            f[pos,:,:] = np.sqrt(diffSet[pos]) * miu / abs(miu)
            # miu2 = (f[pos,:,:] + belta * miu) / (1 + belta)
            # s[pos, :, :] = Propagate(miu2, type, pix, wavelength, -z)
            s[pos, :, :] =( Propagate(f[pos,:,:], type, pix, wavelength, -z)+belta*(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe + aux[pos,:,:] ))/(1+belta)
            aux[pos, :, :] = aux[pos, :, :] + objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe - s[pos, :, :]
            # if np.sqrt(np.sum(abs(s[pos, :, :] - s_last[pos, :, :])**2)) > 1e-2 / (k+1) or\
            #    np.sqrt(np.sum(abs(aux[pos, :, :] - aux_last[pos, :, :]) ** 2)) > 1e-2 / (k+1) or \
            #    np.sqrt(np.sum(abs(aux[pos, :, :]) ** 2)) > 1e+8 :

               # is_adp = 1
        # if k > 50:
        #     print(np.sqrt(np.sum(abs(aux[pos, :, :] - aux_last[pos, :, :]))) / len(positions))
        if k <= 100: # 0.6 s设置为100
            belta = belta * 1.1
        # if np.sqrt(np.sum(abs(s[pos, :, :] - s_last[pos, :, :]))) / len(positions) / k < 1e-4:
        #     is_tr = 1
        # if is_tr == 1:
        #     belta = 0.1
        if k > 200: # 0.6 s设置为400
            belta = 1
        print(belta)

        # s_last = s.copy()
        # aux_last =aux.copy()
        # if k > 2:
        #  err_3.append(relative_err(object, objectFunc, 250))
        k += 1
        print('Iteration %d starts' % k)


        err_2.append(np.sum(err_1) / diffSet_num)

        del err_1[:]
    # out1, objectFunc = dftregistration(object, objectFunc, r=50)

    # psnr_am, psnr_ph, ssim_am, ssim_ph, RMSE = complex_PNSR(object, objectFunc)
    # print('T_ADMM振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph, '\n相位RMSE=', RMSE)
    plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='red', label='TADMM', alpha=0.7,marker='*',markevery=30)
    plt.legend(loc=3, prop={'size': 9})
    plt.xlabel('iteration number')
    plt.ylabel('relative error')
    plt.savefig('t_admm_loss.png', dpi=300)
    print(err_2[-1],err_3[-1])
    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3

def MY_ADMM2(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type, z, wavelength, pix, object, lis):
    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex64)

    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    diffSetIndex = 0
    diffSet_num = 0
    err_3 = [1]
    err_1 = []
    k = 1
    err_2 = [1]



    belta = 1e-9
    s = np.ones(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    x = np.zeros(shape=(len(positions), ysize, xsize))
    s_last = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    aux_last =  np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    objectFunc_last =  np.ones(objectSize, dtype=np.complex64)
    probe_last = np.ones(shape=(ysize, xsize), dtype=np.complex64)
    s1 = 1e-6
    s2 = 1e-6
    r1 = 1e-6
    r2 = 1e-3
    f = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)
    f_last = np.zeros(shape=(len(positions), ysize, xsize), dtype=np.complex64)

    for pos in lis:
        objectIlluminated = objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
        g = objectIlluminated * probe
        s[pos, :, :] = g
        x[pos, :, :] = abs(s[pos, :, :])
        s_last[pos, :, :] =s[pos, :, :]
    while k < n:
        # if k>50:
        #     belta = 0.1
        temp1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp1_1 = np.zeros(shape=(ysize, xsize), dtype=np.complex64)
        temp2 = np.zeros(objectSize, dtype=np.complex64)
        temp2_1 = np.zeros(objectSize, dtype=np.complex64)
        is_adp = 0
        temp = 0
        # 探针更新
        for pos in lis:
            # miu1 = s[pos, :, :] - Propagate(
            #     objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix, wavelength,
            #     z)
            #
            miu2 = probe * abs(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) ** 2  - \
                   np.conj(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) * (s[pos, :, :] - aux[pos, :, :])
            # miu2 = np.conj(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) * (
            #                    s[pos, :, :] - aux[pos, :, :])
            temp1 += miu2
            temp1_1 += abs(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) ** 2

        probe = probe - 0.9 * temp1 / np.max(temp1_1)
        #probe = temp1 / (temp1_1 + 1e-5)
        probe = np.where(abs(probe) > 10 ** 8, 10 * 8, abs(probe)) * Diff.sign(probe)
        # for pos in lis:
        #
        #     miu = Propagate(
        #         objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe + aux[pos, :, :], type,
        #         pix,
        #         wavelength, z)
        #     f[pos, :, :] = np.sqrt(diffSet[pos]) * miu / abs(miu)
        #     miu2 = (f[pos, :, :] + belta * miu) / (1 + belta)
        #     s[pos, :, :] = Propagate(miu2, type, pix, wavelength, -z)
        #     aux[pos, :, :] = aux[pos, :, :] + objectFunc[
        #         positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe - s[pos, :, :]
        #     if np.sqrt(np.sum(abs(s[pos, :, :] - s_last[pos, :, :]) ** 2)) > 1e-2 / k or \
        #             np.sqrt(np.sum(abs(aux[pos, :, :] - aux_last[pos, :, :]) ** 2)) > 1e-2 / k or \
        #             np.sqrt(np.sum(abs(aux[pos, :, :]) ** 2)) > 1e+8:
        #         is_adp = 1

        # 物体更新
        for pos in lis:
            #
            miu = Propagate(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe, type, pix,
                            wavelength, z)
            # miu1 = s[pos, :, :] - miu
            err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(miu)))))
            miu2 = abs(probe)**2 * objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] -\
                   np.conj(probe)*(s[pos, :, :] - aux[pos, :, :])
            #miu2 =  np.conj(probe) * (s[pos, :, :] - aux[pos, :, :])
            temp2[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += miu2
            temp2_1[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += abs(probe) ** 2
            ##误差计算
            if k == 1:
                # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
                diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])) )

        objectFunc = objectFunc - 0.9 * temp2 / np.max(temp2_1)
        #objectFunc = temp2 /(temp2_1 + 1e-5)
        objectFunc = np.where(abs(objectFunc) > 10 * 8, 10 * 8, abs(objectFunc)) * Diff.sign(objectFunc)



        for pos in lis:

            miu = Propagate(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe + aux[pos,:,:]  , type, pix,
                            wavelength, z)
            f[pos,:,:] = np.sqrt(diffSet[pos]) * miu / abs(miu)
            miu2 = (f[pos,:,:] + belta * miu) / (1 + belta)
            s[pos, :, :] = Propagate(miu2, type, pix, wavelength, -z)
            aux[pos, :, :] = aux[pos, :, :] + objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] * probe - s[pos, :, :]
            if np.sqrt(np.sum(abs(s[pos, :, :] - s_last[pos, :, :])**2)) > 1e-2 / k or\
               np.sqrt(np.sum(abs(aux[pos, :, :] - aux_last[pos, :, :]) ** 2)) > 1e-2 / k or \
               np.sqrt(np.sum(abs(aux[pos, :, :]) ** 2)) > 1e+8 :

                is_adp = 1
            temp += np.sqrt(np.sum(abs(aux[pos, :, :] - aux_last[pos, :, :])**2)) / xsize/ysize
        if is_adp ==1 :
            belta = belta * 1.1
        if k > 100:
            belta = 0.1
        print(belta)
        # if k > 10:
        #      # probe = probe + 0.5*(probe - probe_last)
        #      # objectFunc = objectFunc + 0.5*(objectFunc -objectFunc_last)
        #      s = s + 0.8*(s -s_last)
        #      aux = aux + 0.8*(aux - aux_last)
        #      f_last = f +0.8 *(f - f_last)
        # s_last = s.copy()
        # aux_last =aux.copy()
        # probe_last = copy.copy(probe)
        # objectFunc_last = objectFunc.copy()
        # f_last = f.copy()



        k += 1
        print('Iteration %d starts' % k)

        # reset inner loop index
        diffSetIndex = 0
        err_2.append(np.sum(err_1) / diffSet_num)

        del err_1[:]

    psnr_am, psnr_ph, ssim_am, ssim_ph = complex_PNSR(object, objectFunc)
    print('MY_ADMM2振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph)
    #plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='black', label='MY_ADMM2', alpha=0.7)
    plt.legend()
    plt.plot(err_2, color='black')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')
    print(err_2[-1])
    # End of iterations
    print('End of iterations')
    return objectFunc, probe, err_3[-1]

if __name__ == '__main__':
    print('main prog')
