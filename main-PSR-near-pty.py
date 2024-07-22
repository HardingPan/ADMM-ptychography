# -*- coding: utf-8 -*-
"""
Created on 2023 03 22 try to get a Pie with Regularization simulation to work

Working!

Loads an image. Saves image as object phase. Creates a Gussia field probe.
Defines scanning postions on the object.
Creates an exit wave Y= object * probe in each scanning positions. Propagates
the exit wave with a fft. Reconstructs the objec and probe by
running the PIE wtih Red Regularization algorithm.

@author: Xianming Wu

"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as  np
import time
from util.Setpinhole import Setpinhole
from util.ccd_intensities import ccd_intensities
from util.Propagate import Propagate
from  math import pi
from util.Overlap_rate import overlap_rate
import cv2
from util import Sampling
from Algorithms import ADMM_Py
import random
from  util import lowPassFilter
from numpy import  fft
from  util.Normlize import  Normlize
from Algorithms import ADMM_Py
#parameter settings
lamda = 632.8e-6     #wavelength
z1 = 5         #distance from pinhole to object
z2 = 12            #distance from object to ccd
pix = 5.5e-3         #pix size
origin = 24    #the position of start 15
Nx = 16   #Nx -- sampling numbers of x drection 5
Ny =  16      #Ny -- sampling numbers of y drection 5
delta = 16   #delta -- sampling interval 20
# circle probe setting
m = 64  #the rows of probe 150
n = 64  #the cols of probe 150
r =  26 #radius of the pinhole in probe 32


# sanmpling rate
image_size = 256
sigma = 2

overlap_rate(r,delta)
#----------------------------------
# create pinhole
#---------------------------------
pinhole=Setpinhole(m,n,r)


#----------------------------------
# create sim probe and object.
# load image and set to object phase or amplitude values
#----------------------------------

probe_am=pinhole
obj_am = mpimg.imread('./image_data/Set12/lena256.png')

#obj_am = imcrop.imcrop(obj_am , pixsum=256 )
obj_ph = mpimg.imread('./image_data/Set12/pepper256.png')

#judge image chanels,if it is not single chanels ,translating it
try:
        probe_am = np.array(np.sum(probe_am, axis=2))
except:
        print('probe amplitude not 3 chanels')
try:
        obj_am = np.array(np.sum(obj_am, axis=2))
except:
        print('object amplitude not 3 chanels')
try:
        obj_ph = np.array(np.sum(obj_ph, axis=2))
except:
        print('object phase not 3 chanels')

#pad image ,adjust object's size to target value
target_obj = 340
target_probe = 64
(n1, n2) = probe_am.shape
(m1, m2) = obj_am.shape
(q1, q2) = obj_ph.shape
probe_am = np.pad(probe_am, (((target_probe-n1)//2,(target_probe-n1)//2),((target_probe-n2)//2,(target_probe-n2)//2)), 'constant', constant_values=(0,0))
obj_am = np.pad(obj_am, (((target_obj-m1)//2,(target_obj-m1)//2),((target_obj-m2)//2,(target_obj-m2)//2)), 'constant', constant_values=(0,0))
obj_ph = np.pad(obj_ph, (((target_obj-q1)//2,(target_obj-q1)//2),((target_obj-q2)//2,(target_obj-q2)//2)), 'constant', constant_values=(0,0))

#normlize amplitude to 1

#probe_am=probe_am / probe_am.max()
obj_am = Normlize(obj_am,0,1)
# obj_ph = 2*pi*(obj_ph / np.max(obj_ph)) - pi
obj_ph = Normlize(obj_ph,0,pi)
probe_am = Normlize(probe_am,0,1)
print('probe size', probe_am.shape,'\nobject size', obj_am.shape)
# np.save('E:/pythonProject/pie_ penalize/image_data/or_am.npy',obj_am)
# np.save('E:/pythonProject/pie_ penalize/image_data/or_ph.npy',obj_ph)

#make object and probe complex

probe =probe_am*np.exp(1*1j)
object = obj_am*np.exp(obj_ph*1j)

#the probe cross pinhole
probe = Propagate(probe,'Fresnel', pix, lamda, z1)


fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'wspace': 0})
plt.suptitle('initial object and probe')
ax1.imshow((np.abs(object)), cmap='gray');ax1.set_title('amplitude');  ax2.imshow(np.angle(object), cmap='gray'); ax2.set_title('Phase')
ax3.imshow(abs(probe), cmap='gray'); ax3.set_title('Amplitude'); ax4.imshow(np.angle(probe),cmap='gray'); ax4.set_title('Phase')


#get diffractions image,postions,illuindy,illuindx


diffset,positions,illuindx,illuindy=ccd_intensities(object,probe,lamda,origin,Nx,Ny,delta,pix,z2,'Fresnel')

diffset_sp = []
rows  =  diffset[0].shape[0]
probe_r = np.zeros(shape=(rows // sigma, rows // sigma))
for i in range(len(diffset)):


        diffset[i]= Sampling.C(diffset[i],sigma = sigma)
        # diffset[i] = diffset[i] / 4
        # diffset[i] = imcrop.imcrop( diffset[i],pixsum=32)
        probe_r += np.sqrt(diffset[i])
        diffset_sp.append(diffset[i])

#difine object shape(needed for reconstruction)
plt.figure()
plt.title('one of Diffraction images')
plt.imshow(np.log10(abs(diffset_sp[3])+0.1))

object_re_shape=object.shape
probe_r = lowPassFilter.lowPassFiltering(probe_r,19)
probe_r = fft.ifft2(probe_r)/len(diffset)
probe_r = abs(fft.ifftshift(probe_r))
probe_r = probe_r / np.max(probe_r)
probe_r = Sampling.CT(probe_r,sigma=sigma )
# numbers of iteration
k = 1000

probe_r = Setpinhole(m,n,r)

#probe_r =  imcrop.imcrop(probe_r, pixsum=150)

probe_sp = Setpinhole(int(m / sigma),int(n / sigma),int( r / sigma ))
#probe_sp =  imcrop.imcrop(probe_sp, pixsum=150)
#run algrithm

time_start=time.perf_counter()

illuindy_sp, illuindx_sp = np.indices((probe_sp.shape))

# object_sp_am=Sampling.C(abs(object),sigma)
# object_sp_ph = Sampling.C(np.angle(object),sigma)
# object_sp = object_sp_am * np.exp(1j*object_sp_ph)
# object_re_shape_sp = object_sp.shape

# random_offet = np.random.randint(-1, 1, np.shape(positions))
# positions += random_offet
a = (int)(positions.size / 2)
lis =list(range(a))
random.shuffle(lis)

#object_re1, probe_re1, err_1=WF.wf(k, diffset, probe_sp, object_re_shape_sp, positions_sp, illuindy_sp, illuindx_sp,'Fresnel',z2,lamda,pix,object)
#object_re2, probe_re2, err_2=WF.rwf_mean(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object
# object_re3, probe_re3, err_3=various_PIE(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,'rPIE',object,lis)
#object_re4, probe_re4, err_4=various_PIE(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,'rPIE',object)
#object_re5, probe_re5, err_5=various_PIE(k, diffset_sp, probe_sp, object_re_shape_sp, positions_sp, illuindy_sp, illuindx_sp,'Fresnel',z2,lamda,pix,'ePIE',object_sp)
#object_re6, probe_re6, err_6=WF.wf_mean(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object)
#object_re7, probe_re7, err_7=WF.wf_epie(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object)
# object_re6, probe_re6, err_6=WF.wf_global(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object)
# object_re8, probe_re8, err_8=WF.wf_Nesterov(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object)
# object_re9, probe_re9, err_9=ADMM_Py.ADMM_Py_TV(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object)
object_re1, probe_re1, err_1= ADMM_Py.ADMM_Py_Aps(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx, 'Fresnel', z2, lamda, pix, object, lis, sigma, lamda=1e-2, colour='r')
# object_re2, probe_re2, err_2= ADMM_Py.ADMM_Py_ps(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,object,lis,sigma)
object_re2, probe_re2, err_2= ADMM_Py.ADMM_Py_Aps(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx, 'Fresnel', z2, lamda, pix, object, lis, sigma, lamda=5e-3, colour='b')
object_re3, probe_re3, err_3= ADMM_Py.ADMM_Py_Aps(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx, 'Fresnel', z2, lamda, pix, object, lis, sigma, lamda=1e-3, colour='g')
object_re4, probe_re4, err_4= ADMM_Py.ADMM_Py_Aps(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx, 'Fresnel', z2, lamda, pix, object, lis, sigma, lamda=5e-4, colour='c')
object_re5, probe_re5, err_5= ADMM_Py.ADMM_Py_Aps(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx, 'Fresnel', z2, lamda, pix, object, lis, sigma, lamda=1e-4, colour='y')

##科学计数法显示
plt.figure()
plt.semilogy(err_1, linewidth=1.5, color='r', label=  r'$\rho = 1 \times 10^{-2}$', alpha=0.7)
plt.semilogy(err_2, linewidth=1.5, color='b', label=  r'$\rho = 5 \times 10^{-3}$', alpha=0.7)
plt.semilogy(err_3, linewidth=1.5, color='g', label=  r'$\rho = 1 \times 10^{-3}$', alpha=0.7)
plt.semilogy(err_4, linewidth=1.5, color='c',label=  r'$\rho = 5 \times 10^{-4}$', alpha=0.7)
plt.semilogy(err_5, linewidth=1.5, color='y',label=  r'$\rho = 1 \times 10^{-4}$', alpha=0.7)
plt.legend(loc = 'upper right', prop={'size': 9})
plt.xlabel('iteration number')
plt.ylabel('relative error')

plt.savefig('C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/ISNesterov.png', dpi= 1000, bbox_inches ='tight')
# np.save('E:/pythonProject/pie_ penalize/image_data/re_am.npy',abs(object_re1))
# np.save('E:/pythonProject/pie_ penalize/image_data/re_ph.npy',np.angle(object_re1))

time_end =time.perf_counter()
time_consume=time_end-time_start

print('iteration consume time ',time_consume)

cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_obr.png",Normlize(np.abs(object_re1[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_phr.png",Normlize(np.angle(object_re1[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_pror.png",Normlize(np.abs(probe_re1),0,255))
# #
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_obb.png",Normlize(np.abs(object_re2[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_phb.png",Normlize(np.angle(object_re2[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_prob.png",Normlize(np.abs(probe_re2),0,255))
#
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_obg.png",Normlize(np.abs(object_re3[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_phg.png",Normlize(np.angle(object_re3[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_prog.png",Normlize(np.abs(probe_re3),0,255))
#
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_obc.png",Normlize(np.abs(object_re4[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_phc.png",Normlize(np.angle(object_re4[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_proc.png",Normlize(np.abs(probe_re4),0,255))
#
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_oby.png",Normlize(np.abs(object_re5[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_phy.png",Normlize(np.angle(object_re5[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/lamda/aps_proy.png",Normlize(np.abs(probe_re5),0,255))
#----------------------------------
# image the result
#----------------------------------
#

fig, ((x1,x2), (x3,x4)) = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction')
x1.imshow( np.abs(object_re1) /np.max(np.abs(object_re1)), cmap='gray'); x1.set_title('object amplitude')
x2.imshow(np.angle(object_re1), cmap='gray'); x2.set_title('object phase ')
x3.imshow(np.abs(probe_re1/np.max(np.abs(object_re1))), cmap='gray'); x3.set_title('probe amplitude')
x4.imshow(np.angle(probe_re1), cmap='gray'); x4.set_title('probe phase')
fig, ((x1,x2), (x3,x4)) = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction2')
x1.imshow(np.abs(object_re2) / np.max(np.abs(object_re2)), cmap='gray'); x1.set_title('object amplitude')
x2.imshow(np.angle(object_re2) , cmap='gray'); x2.set_title('object phase ')
x3.imshow(np.abs(probe_re2) / np.max(np.abs(probe_re2)), cmap='gray'); x3.set_title('probe amplitude')
x4.imshow(np.angle(probe_re2), cmap='gray'); x4.set_title('probe phase')
# plt.savefig("C:/Users/86364/Desktop/实验数据/迭代五十次重建图像对比/rpie_50.png")

plt.show()
cv2.waitKey(0)