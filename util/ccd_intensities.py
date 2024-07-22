import numpy as np
import matplotlib.pyplot as plt
from util.Propagate import Propagate
import util.Noise
import matplotlib.image as mpimg
import copy



def ccd_intensities(object,probe,lamda, origin,Nx,Ny,delta,pix,z,type):
    #obeject -- the all of object
    #probe -- the size of probe(its size is the same with diffrication image)
    #lamda -- wavelength
    #origin -- sampling start point
    #Nx -- sampling numbers of x drection
    #Ny -- sampling numbers of y drection
    #delta -- sampling interval
    #pix -- ccd's pixel size
    #z -- propagate
    #type -- propagator
    positions = np.zeros((Nx * Ny, 2), dtype=np.int32)
    dy=delta
    dx=delta
    # row postions in 1st colum
    positions[:, 1] = np.tile(np.arange(Nx) * dx, Ny)
    positions[:, 0] = np.repeat(np.arange(Ny) * dy, Nx)
    positions += origin
    positions2 = positions.copy()
    #add random offset
    random_offet = np.random.randint(-7,7,np.shape(positions))
    positions += random_offet
    #print(positions)
    diffset = [] #ccd intensities
    sum = 0

    # the indices for the area that is illuminated by the probe (should have the probes shape)

    illuindy, illuindx = np.indices((probe.shape)) #Sampling range

    ims = []
    for pos in positions:
    #propogare to far field with a Fourier transform ,then calcute the absolute square




          # diffset.append(abs(Propagate(object[pos[0]+illuindy,pos[1]+illuindx]*probe, type, pix, lamda, z))**2)
          # img ,sum =util.Noise .gasuss_noise(abs(Propagate(object[pos[0] + illuindy, pos[1] + illuindx] * probe, type, pix, lamda, z))**2,sigma=0.05,sum=sum) #gauss noise
          # diffset.append(abs(util.Noise .poisson_noise(abs(Propagate(object[pos[0] + illuindy, pos[1] + illuindx] * probe, type, pix, lamda, z))**2,500000)))
          img ,sum= util.Noise.poisson_noise2(abs(Propagate(object[pos[0] + illuindy, pos[1] + illuindx] * probe, type, pix, lamda, z)) ** 2,alpha  =1,  sum=sum)
          diffset.append(abs(img))
    print('Diffraction pattern created')

    plt.figure()
    plt.title('one of Diffraction images')
    plt.imshow(np.log10(abs(diffset[3])+0.1))


    return diffset,positions, illuindx, illuindy








if __name__ == '__main__':
    print('main prog')
