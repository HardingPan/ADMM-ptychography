import numpy as np
from  util.calculate_shapness import calculate_image_sharpness
from util.Propagate import Propagate
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

    hash_table = {}
    e = 0.01


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
                g = objectIlluminated * probe1

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
                        probe1) / (np.max(abs(probe)) ** 2)  # probe* annars blir det att man delar med massa nollor
                    # objectFunc[pos[0] + illu_indy, pos[1] + illu_indx] = objectIlluminated + alpha * (gprime - g) * np.conj(
                    #       probe)  # probe* annars blir det att man delar med massa nollor
                    # update probe function
                    beta = 0.2  # higher value == faster change
                    probe1 = probe1 + beta * (gprime - g) * np.conj(objectIlluminated) / (
                                np.max(abs(objectIlluminated)) ** 2)



            k += 1
            print('%f distance serch Iteration %d starts' % (z, k))

        sharp = calculate_image_sharpness(abs(objectFunc[objectSize[0]//2 - 75:objectSize[0]//2 + 75, objectSize[1]//2 - 75:objectSize[1]//2 + 75]))
        k = 0
        hash_table[z] = sharp


    # 按值对哈希表进行排序
    sorted_items = sorted(hash_table.items(), key=lambda x: x[1])

    # 输出最大值和次大值对应的键
    if len(sorted_items) >= 1:
        max_key = sorted_items[-1][0]
        print("Max key:", max_key)

    if len(sorted_items) >= 2:
        second_max_key = sorted_items[-2][0]
        print("Second max key:", second_max_key)
    print(hash_table)