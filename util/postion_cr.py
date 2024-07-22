import numpy as np
from util.Propagate import  Propagate
from util.subpixel_registration import dftregistration
import cv2
from  numpy import  fft


def  postion_corection(k, diffset, probe_r, z2,type,lamda,pix,r):

    positions = np.zeros((len(diffset), 2), dtype=np.float)
    [rows ,cols] = probe_r.shape
    for i  in range(0, len(diffset)-1):
      #temp = sigle_wf(k, diffset[i], probe_r, z2, type,lamda, pix)
      #next = sigle_wf(k, diffset[i+1], probe_r, z2, type, lamda, pix)
      temp = Propagate(diffset[i], 'Fresnel', pix, lamda, -z2 )
      temp = temp*np.conj(probe_r)
      next = Propagate(diffset[i+1], 'Fresnel', pix, lamda, -z2 )
      next = next*np.conj(probe_r)
      [out,grey] = dftregistration(abs(temp)[rows // 2 - r :rows // 2 + r,cols // 2 - r :cols // 2 + r],
                                   abs(next)[rows // 2 - r :rows // 2 + r,cols // 2 - r :cols // 2 + r],r = 10)
      positions[i+1][0] = positions[i][0] + out[2]
      positions[i+1][1] = positions[i][1] + out[3]
    cv2.imshow("1",abs(temp)[rows // 2 - r :rows // 2 + r,cols // 2 - r :cols // 2 + r]/np.max(abs(temp)[rows // 2 - r :rows // 2 + r,cols // 2 - r :cols // 2 + r]))
    cv2.imshow("2",abs(next)[rows // 2 - r :rows // 2 + r,cols // 2 - r :cols // 2 + r]/np.max(abs(next)[rows // 2 - r :rows // 2 + r,cols // 2 - r :cols // 2 + r]))
    positions += 250
    print(positions)
    # cv2.waitKey(0)
    return positions


def sigle_wf(k, diffset, probe_r, z2, type,wavelength, pix):
    ysize, xsize = probe_r.shape
    object = np.ones(shape = (ysize, xsize),dtype = np.complex128)

    for i in range(k):

        g = object * probe_r
        miu = Propagate(g, type, pix, wavelength, z2)
        miu_2 = miu - miu * np.sqrt(diffset) / (abs(miu))
        miu_3 = 0.5 * Propagate(miu_2, type, pix, wavelength, -z2) * np.conj(probe_r)
        object = object - 2 * miu_3 / (abs(probe_r) ** 2 + 2)
        probe_r = probe_r -2 * 0.5 * Propagate(miu_2, type, pix, wavelength, -z2) * np.conj(object)/(abs(object)**2+2)

    return object






if __name__ =='__main__':
    f = cv2.imread('E:/pythonProject/pie-penalize/image_data/Set12/lena256.jpg')
    f_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    f2 = cv2.imread('E:/pythonProject/pie-penalize/image_data/Set12/lena256.jpg')
    f_gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    cv2.normalize(f_gray, f_gray, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.normalize(f_gray2,  f_gray2, 0, np.pi, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    x = f_gray
    # probe = Setpinhole(256, 256, 100)
    # probe = Propagate(probe, 'Fresnel', dx=5.5e-3, wavelength=632.8e-6, z=5.0)
    x1 = Propagate(x, 'Fourier', dx=5.5e-3 , wavelength=632.8e-6 , z=100000)
    # x2 = Propagate(abs(x1), 'Fresnel', dx=5.5e-3 , wavelength=632.8e-6 ,z=-10)
    #x1 = fft.fft2(x)
    x2 = fft.ifft2(fft.ifftshift(x1))
    cv2.imshow("1", abs(x2) / np.max(abs(x2)))
    cv2.waitKey(0)