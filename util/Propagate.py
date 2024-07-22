
import numpy as np
from numpy import fft
from math import pi
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def Propagate(input, propagator, dx=5.5e-3 , wavelength=632.8e-6 , z=5.0):
    # wavefront propagate
    # input -- the wavefront to propagate
    # propagator -- one of 'fourier','fresnel'
    # dx -- the pixel spacing of input wavefront
    # z -- the distance to propagate
    # output -- the distance to propagete

    ysize, xsize = np.shape(input)
    x = np.array(np.arange(-xsize/2, xsize/2, 1))
    y = np.array(np.arange(-ysize/2, ysize/2, 1))
    fx= x/(dx*xsize)
    fy= y/(dx*ysize)
    fx,fy = np.meshgrid(fx, fy)

    if propagator=='Fourier':
        if  z > 0:
            output=fft.fftshift(fft.fft2(input))
        elif z==0:
            output=input
        else:
            output=fft.ifft2(fft.ifftshift(input))

    # Calculate approx phase distribution for each plane wave component
    elif propagator=='Fresnel':
        w=fx**2+fy**2
        F=fft.fftshift(fft.fft2(input))
        output=fft.ifft2(fft.ifftshift(F*np.exp(-1j*pi*z*wavelength*w)))
    else:
         output=input
         print('invalid propagate way,please change')

    return output


if __name__ == '__main__':


 object_am = mpimg.imread('lena.jpg')
 object=Propagate(object_am,'Fourier',5.5e-3,632.8e-6,3)
 a=Propagate(object,'Fourier',5.5e-3,632.8e-6,-3)
 plt.figure()
 plt.imshow(abs(a))
 plt.show()









