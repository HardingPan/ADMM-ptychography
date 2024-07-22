import numpy as np
import cv2
from numpy import fft
from math import pi




#  imFTout = FTpad(imFT,outsize)
#  Pads or crops the Fourier transform to the desired ouput size. Taking
#  care that the zero frequency is put in the correct place for the output
#  for subsequent FT or IFT. Can be used for Fourier transform based
#  interpolation, i.e. dirichlet kernel interpolation.
#
#   Inputs
#  imFT      - Input complex array with DC in [1,1]
#  outsize   - Output size of array [ny nx]
#
#    Outputs
#  imout   - Output complex image with DC in [1,1]

def FTpad(image, outsize):
    image = np.array(image)
    if image.ndim > 2 :
        print("Maximum number of array dimensions is 2")

    Nout = outsize
    Nin  = np.shape(image)
    imFT = fft.fftshift(image)
    center = np.floor(np.array(np.shape(imFT)) / 2)
    imFTout =  np.zeros(outsize,dtype=complex)
    centerout = np.floor(np.array(np.shape(imFTout))/2)
    cenout_cen = centerout - center
    imFTout[int(np.max([cenout_cen[0] , 0])): int(np.min([cenout_cen[0] + Nin[0], Nout[0]])), int(np.max([cenout_cen[1] , 0])): int(np.min([cenout_cen[1] + Nin[1], Nout[1]]))]=imFT[
     int(np.max([-cenout_cen[0],0])): int(np.min([-cenout_cen[0] + Nout[0], Nin[0]])), int(np.max([-cenout_cen[1] , 0])): int(np.min([-cenout_cen[1] + Nout[1], Nin[1]]))]
    imFTout = fft.ifftshift(imFTout)* Nout[0] * Nout[1] / (Nin[0] * Nin[1])
    return imFTout



def dftups(in_array, nor=None, noc=None, usfac=1, roff=0, coff=0):
        nr, nc = in_array.shape
        if nor is None:
            nor = nr
        if noc is None:
            noc = nc
        # Compute kernels and obtain DFT by matrix products
        #这里运用了python的性质，还可以写成点乘形式(fft.ifftshift(np.arange(nc)).reshape(-1, 1) - np.floor(nc / 2)) @ ([np.arange(noc) - coff)])
        kernc = np.exp((-1j * 2 * pi / (nc * usfac)) * (fft.ifftshift(np.arange(nc)).reshape(-1, 1) - np.floor(nc / 2)) * (
                        np.arange(noc) - coff))

        kernr = np.exp((-1j * 2 * pi / (nr * usfac)) * (np.arange(nor).reshape(-1, 1) - roff) * (
                        fft.ifftshift(np.arange(nr)) - np.floor(nr / 2)))

        out_array = kernr @ in_array @ kernc
        return out_array



def dftregistration(image1, image2, r=1): #image1 is reference image

    image1 = fft.fft2(image1)
    image2 = fft.fft2(image2)
    [nr, nc] = np.shape(image2)
    x = np.array(np.arange(-np.fix(nr / 2), np.ceil(nr / 2), 1))
    y = np.array(np.arange(-np.fix(nc / 2), np.ceil(nc / 2), 1))
    Nr = fft.ifftshift(x)
    Nc = fft.ifftshift(y)

    # 未配准误差计算
    if r==0:
        CCmax = np.sum(image1*np.conj(image2))

        #CCmax =  np.max(fft.ifft2(image1 * np.conj(image2)))*nc*nr
        row_shift = 0
        col_shift = 0

    #像素级配准计算误差
    elif r==1:
        CC = fft.ifft2(image1*np.conj(image2))
        CCabs = abs(CC)
        row_shift,col_shift = np.where(CCabs==np.max(CCabs))
        CCmax = CC[row_shift,col_shift]*nr*nc
        row_shift = Nr[row_shift]
        col_shift = Nc[col_shift]


    #亚像素配准
    elif r > 1:
        CC = fft.ifft2(FTpad(image1*np.conj(image2),[2*nr,2*nc]))
        CCabs = abs(CC)
        [row_shift,col_shift] = np.where(CCabs ==np.max(CCabs))
        CCmax = CC[row_shift, col_shift] * nr * nc
        Nr2 = fft.ifftshift( np.array(np.arange(-np.fix(nr), np.ceil(nr), 1)))
        Nc2 = fft.ifftshift( np.array(np.arange(-np.fix(nr), np.ceil(nr), 1)))
        row_shift = Nr2[row_shift] / 2
        col_shift = Nc2[col_shift] / 2
        if r>2 :
         '''DFT computation in 1.5r around'''
         # Initial shift estimate in upsampled grid
         row_shift = np.round(row_shift * r ) / r
         col_shift = np.round(col_shift * r) / r
         dftshift = np.fix(np.ceil(r * 1.5) / 2)
         CC = np.conj(dftups(image2*np.conj(image1), nor=np.ceil( r*1.5), noc=np.ceil(r*1.5 ), usfac=r,roff= dftshift-row_shift*r,coff= dftshift-col_shift*r))
         # Locate maximum and map back to original pixel grid
         CCabs = abs(CC)
         [rloc , cloc] = np.where(CCabs == np.max(CCabs))
         CCmax =  CC[rloc,cloc]
         rloc = rloc -dftshift -1
         cloc = cloc -dftshift -1
         row_shift = row_shift + rloc/r
         col_shift = col_shift + cloc/r
    rg00 = np.sum(abs(image1)**2)
    rg11 = np.sum(abs(image2) ** 2)
    error = 1.0 - abs(CCmax**2/(rg00*rg11))
    error = np.sqrt(abs(error))
    diffphase = np.angle(CCmax)
    if r>0:
        [Nc, Nr] = np.meshgrid(Nc, Nr)
        Greg = image2*np.exp(1j*2*pi*(-row_shift *Nr/nr - col_shift*Nc/nc))
        Greg = Greg*np.exp(1j * diffphase)
        Greg =fft.ifft2( Greg )
    else:
        Greg = image2 * np.exp(1j * diffphase)
        Greg = fft.ifft2(Greg)
    out=[error, diffphase,row_shift, col_shift ]
    return out,Greg





if __name__ == '__main__':


 f = cv2.imread('E:/pythonProject/pie-penalize/image_data/Set12/lena256.jpg')
 s = cv2.imread('E:/pythonProject/pie-penalize/image_data/Set12/cell256.jpg')
 image1=np.ones(f.shape)
 f_gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
 s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
 dist=np.ones(f_gray.shape)
 cv2.normalize(f_gray, dist, 0, 1,cv2.NORM_MINMAX,dtype=cv2.CV_64F)

 deltar = -3.84515
 deltac = 8.73837
 phase = 2
 [nr,nc]=np.shape(dist)
 x = np.array(np.arange(-nr / 2, nr / 2, 1))
 y = np.array(np.arange(-nc / 2, nc / 2, 1))
 NR = fft.ifftshift(x)
 NC = fft.ifftshift(y)
 [NC,NR] =np.meshgrid(NC,NR)

 g =fft.ifft2((fft.fft2(dist))*np.exp(1j*2*pi*(deltar*NR/nr+deltac*NC/nc)))*np.exp(-1j*phase)
 cv2.imshow("lena", abs(g))
 out,Greg= dftregistration(f_gray, g, r=100)
 print(out[2])
 cv2.imshow("register image", abs(Greg))
 cv2.waitKey(0)


