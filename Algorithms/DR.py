import numpy as np
from util.Propagate import  Propagate
import matplotlib.pyplot as plt
from util.relative_err import relative_err
import regularization
from util.evalution import complex_PNSR
from util.subpixel_registration import  dftregistration
from FFDNet_1.models import FFDNet

from util import Diff

def DR(n, diffSet, probe, objectSize, positions, illu_indy, illu_indx, type,z,wavelength,pix,object,lis):

    ysize, xsize = probe.shape
    objectFunc = np.ones(objectSize, dtype=np.complex32)
    objectIlluminated = np.ones(shape=(ysize, xsize), dtype=np.complex32)
    objectIlluminated_gr = np.ones(shape=(ysize, xsize), dtype=np.complex32)
    diffSetIndex = 0
    diffSet_num = 0
    err_3=[1]
    err_1 = []
    k=0
    err_2= [1]
    mask = np.zeros(objectSize)

    miu = np.zeros((ysize, xsize), dtype=np.complex32)
    probe_2 = np.zeros(shape=(ysize, xsize), dtype=np.complex32)
    phai = np.ones(shape=(len(positions),ysize, xsize),dtype=np.complex32)
    aux = np.zeros(shape=(len(positions),ysize, xsize),dtype=np.complex32)
    for pos in lis:
        phai[pos,:,:] = probe * objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]
    while k<n :


        temp1 = np.zeros(shape=(ysize, xsize),dtype=np.complex32)
        temp1_1 = np.zeros(shape=(ysize, xsize),dtype=np.complex32)
        temp2 = np.zeros(objectSize,dtype=np.complex32)
        temp2_1 = np.zeros(objectSize,dtype=np.complex32)
        # 探针更新
        for pos in lis:
            temp1 += phai[pos, :, :] * np.conj(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx])

            temp1_1 += abs(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]) ** 2
        probe =   temp1 / (temp1_1 + 1e-4)
        #物体更新
        for pos in lis:

            temp2[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += phai[pos,:,:]*np.conj(probe)
            temp2_1[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] += abs(probe)**2
            # temp1 += phai[pos,:,:]*np.conj(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx])
            # temp1_1 += abs(objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx])**2
            # obj_counter[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] = obj_counter[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] +1
        # probe = temp1 / (temp1_1 +1e-10)
        objectFunc = temp2 / (temp2_1 + 1e-4)

        for pos in lis:
            aux[pos,:,:] = 2 * probe * objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx] - phai[pos, :, :]

            G =  Propagate(aux[pos,:,:], type, pix, wavelength, z)
            Gprime = np.sqrt(diffSet[pos]) * np.exp(1j * np.angle(G))

            # inverse Fourier transform
            aux[pos,:,:] = Propagate(Gprime, type, pix, wavelength, -z)

            phai[pos,:,:] = phai[pos,:,:] + aux[pos,:,:]- probe * objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx]

        for pos in lis:
            G = Propagate(probe * objectFunc[positions[pos][0] + illu_indy, positions[pos][1] + illu_indx], type, pix, wavelength, z)
            #err_1.append(np.sum((abs(abs(diffSet[pos]) - abs(G)**2)) ** 2))
            err_1.append(np.sum((abs(np.sqrt(abs(diffSet[pos])) - abs(G)))))
            if k == 0:
             # diffSet_num += np.sum(abs(diffSet[diffSetIndex])**4)
             #diffSet_num += np.sum(abs(diffSet[pos]) ** 2)
             diffSet_num += np.sum(np.sqrt(abs(diffSet[pos])))
            diffSetIndex += 1

        # if k > 1:
        #     probe = probe + (k / (k + 3)) * (probe - probe_N)
        #     objectFunc = objectFunc + (k / (k + 3)) * (objectFunc - objectFunc_N)
        objectFunc_N = objectFunc
        probe_N = probe

        probe_2 = 0
        k+=1
        #err_3.append(relative_err(object, objectFunc, 250))
        print('Iteration %d starts'% k)
        diffSetIndex = 0

        err_2.append(np.sum(err_1) / diffSet_num)
        del err_1[:]
    # psnr_am, psnr_ph, ssim_am, ssim_ph = complex_PNSR(object, objectFunc)
    # print('DM振幅pnsr=', psnr_am, ' 相位pnsr=', psnr_ph, '\n振幅ssim=', ssim_am, ' 相位ssim=', ssim_ph)
    plt.figure()
    plt.semilogy(err_2, linewidth=1.5, color='orange', label='DM', alpha=0.7)
    plt.legend()
    plt.plot(err_2, color='orange')
    plt.xlabel('iteration number')
    plt.ylabel('relative residual norm(res)')
    # End of iterations
    print('End of iterations')
    print(err_2[-1])
    return objectFunc, probe, err_2[-1]

