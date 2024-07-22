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
import numpy as  np
import time
from util.Setpinhole import Setpinhole,Setpinhole2
from util.ccd_intensities import ccd_intensities
from util.Propagate import Propagate
from numpy import  fft
from  math import pi
from util.Overlap_rate import overlap_rate
import cv2
import imcrop
from util.Normlize import  Normlize
import random
from  util import lowPassFilter
from Algorithms import WF

#parameter settings
lamda = 632.8e-6     #wavelength
z1 = 5    #distance from pinhole to object
z2 = 40           #distance from object to ccd
pix = 5.5e-3         #pix size
origin = 50#the position of start 15
Nx =  10 #Nx -- sampling numbers of x drection 5    /10  /7
Ny = 10 #Ny -- sampling numbers of y drection 5   /10   /7
delta = 26 #delta -- sampling interval 20  /24      /35

# circle probe setting
m = 64 #the rows of probe  150
n = 64 #the cols of probe  150
r = 26  #radius of the pinhole in probe 32  60

image_size = 256
cropsize = 64
overlap_rate(r,delta)
#----------------------------------
# create pinhole
#--------------------------------
pinhole=Setpinhole(m,n,r)


#----------------------------------
# create sim probe and object.
# load image and set to object phase or amplitude values
#----------------------------------

#probe_am = mpimg.imread('./image_data/feiyuantanzhen.png')
# probe_am = probe_am[51:201,51:201]
probe_am=pinhole
obj_am = cv2.imread('./image_data/Set12/cameraman256.png',cv2.IMREAD_GRAYSCALE)

#obj_am = imcrop.imcrop(obj_am , pixsum=256 )
obj_ph = cv2.imread('./image_data/Set12/cell256.jpg',cv2.IMREAD_GRAYSCALE)
#judge image chanels,if it is not single chanels ,translating it
try:
        probe_am = np.array(np.sum(probe_am, axis=2))
except:
        print('probe amplitude not 3  chanels')
try:
        obj_am = np.array(np.sum(obj_am, axis=2))
except:
        print('object amplitude not 3 chanels')
try:
        obj_ph = np.array(np.sum(obj_ph, axis=2))
except:
        print('object phase not 3 chanels')

#pad image ,adjust object's size to target value
target_obj = 400
target_probe = 64
(n1, n2) = probe_am.shape
(m1, m2) = obj_am.shape
(q1, q2) = obj_ph.shape
probe_am = np.pad(probe_am, (((target_probe-n1)//2,(target_probe-n1)//2),((target_probe-n2)//2,(target_probe-n2)//2)), 'constant', constant_values=(0,0))
obj_am = np.pad(obj_am, (((target_obj-m1)//2,(target_obj-m1)//2),((target_obj-m2)//2,(target_obj-m2)//2)), 'constant', constant_values=(0,0))
obj_ph = np.pad(obj_ph, (((target_obj-q1)//2,(target_obj-q1)//2),((target_obj-q2)//2,(target_obj-q2)//2)), 'constant', constant_values=(0,0))

#normlize amplitude to 1

#probe_am=probe_am / probe_am.max()
# obj_am = obj_am / np.max(obj_am)
obj_am = Normlize(obj_am,0,1)
# obj_ph = 2*pi*(obj_ph / np.max(obj_ph)) - pi
obj_ph = Normlize(obj_ph,-pi, pi)
probe_am = Normlize(probe_am,0, 1)

print('probe size', probe_am.shape,'\nobject size', obj_am.shape)


#make object and probe complex

# probe_am = cv2.imread('./image_data/yuantanzhen.png', cv2.IMREAD_GRAYSCALE)
# probe_am  = probe_am / np.max(probe_am)
probe = probe_am * np.exp(1*1j)
object = obj_am*np.exp(obj_ph*1j)

#the probe cross pinhole
probe = Propagate(probe,'Fresnel', pix, lamda, z1)

# cv2.imwrite("C:/Users/86364/Desktop/Alps/image/probe_ob.png",Normlize(np.abs(probe),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/Alps/image/probe_ph.png",Normlize(np.angle(probe),0,255))


#show object and probe
fig, ax = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe')
ax = ax.flatten()
ax0=ax[0].imshow(obj_am, cmap='gray'); ax[0].set_title('object amplitude')
ax1=ax[1].imshow(obj_ph, cmap='gray'); ax[1].set_title('object phase ')
ax[2].imshow(np.abs(probe)/np.max(np.abs(probe)), cmap='gray'); ax[2].set_title('probe amplitude')
ax[3].imshow(np.angle(probe), cmap='gray'); ax[3].set_title('probe phase')
fig.colorbar(ax0,ax = ax[0]);fig.colorbar(ax1,ax = ax[1])



#get diffractions image,postions,illuindy,illuindx

diffset,positions,illuindx,illuindy=ccd_intensities(object,probe,lamda,origin,Nx,Ny,delta,pix,z2,'Fourier')
probe_r =np.zeros(diffset[2].shape)

for i in range(len(diffset)):

        diffset[i] = imcrop.imcrop(diffset[i], pixsum=cropsize)
        probe_r += np.sqrt(diffset[i])
#print(positions)
#difine object shape(needed for reconstruction)
probe_r = lowPassFilter.lowPassFiltering(probe_r, 17)
probe_r = fft.ifft2(probe_r)/len(diffset)
probe_r = abs(fft.ifftshift(probe_r))
probe_r = probe_r / np.max(probe_r)
# probe_b = probe_r
object_re_shape = object.shape
probe_c = abs(Propagate(probe,'Fourier', pix, lamda, z1))**2
# numbers of iteration
k = 500
probe_b = Setpinhole(m,n,r)


#probe_r = imcrop.imcrop(probe_r,pixsum= cropsize)

illuindy, illuindx = np.indices(probe_r.shape)
#run walgrithm

a = (int)(positions.size / 2)
lis =list(range(a))
random.shuffle(lis)
lis2 = list(range(a))
b =Setpinhole2(m,n)

time_start0=time.perf_counter()


# object = object / norm(object, 'fro')
# object_re1, probe_re1, err_1=WF.rwf(k, diffset, probe_r , object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object,lis)
#object_re5, probe_re5, err_5 = postion_corection(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,'ePIE',object,lis)
#object_re2, probe_re2, err_2=WF.rwf_mean(k, diffset, probe_r, object_re_shape, po sitions, illuindy, illuindx,'Fresnel',z2,lamda,pix,object)
#object_re3, probe_re3, err_3 = pos_crrect(k, diffset, p robe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,'ePIE',object,lis,r,is_center_correct = 1)
# object_re3, probe_re3, err_3=various_PIE_engine.various_PIE(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,'ePIE',object,lis,is_center_correct = 1)
# time_start1 =time.perf_counter()
# object_re4, probe_re4, err_4=various_PIE_engine.various_PIE(k, diffset,  probe_r.copy(), object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,'rPIE',object,lis,is_center_correct = 1)
# object_re4, probe_re4, err_4= WASP.Wasp(k, diffset,  probe_r.copy(), object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object,lis,is_center_correct = 1)
# time_start2 =time.perf_counter()
# # #object_re4, probe_re4, err_4=WF.wf_nosie(k, dif  fset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object,lis)
object_re8, probe_re8, err_8 = WF.T_ADMM(k, diffset, probe_r.copy(), object_re_shape, positions, illuindy, illuindx, 'Fourier', z2, lamda, pix, object, lis, is_center_correct = 0)
object_re6, probe_re6, err_6 = WF.ADMM(k, diffset, probe_r.copy(), object_re_shape, positions, illuindy, illuindx, 'Fourier', z2, lamda, pix, object, lis, is_center_correct = 1)
# time_start3 =time.perf_counter()
object_re7, probe_re7, err_7 = WF.L_ADMM(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx, 'Fourier', z2, lamda, pix, object, lis, is_center_correct = 1)
# time_start4 =time.perf_counter()
# time_start5 =time.perf_counter()
#object_re6, probe_re6, err_6=WF.wf_red_net(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object)
# object_re7, probe_re7, err_7=WF.wf_global(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object,lis)
#object_re6, probe_re6, err_6 = WF.wf(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object,lis)
# object_re8, probe_re8, err_8=WF.wf_Nesterov(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object)
# object_re6, probe_re6, err_6= ADMM_Py.ADMM_net_denoise(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx, 'Fourier', z2, lamda, pix, object, lis, is_center_correct = 1)
# object_re10, probe_re10, err_10=ADMM_Py.ADMM_TV_denoise(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object,lis, is_center_correct = 1)
#object_re12, probe_re12, err_12=ADMM_Py.ADMM_Py_TV(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object)
# plt.savefig('C:/Users/86364/Desktop/OL/wucha/16/R_16_noise.png', dpi= 1000, bbox_inches ='tight') #save pq
#np.save('E:/pythonProject/pie-penalize/image_data/re_am.npy+',abs(object_re5))
# np.save('E:/pythonProject/pie_ penalize/image_data/re_ph.npy',np.angle(object_re1))6
##额外绘制相对误差
plt.figure()
# plt.semilogy(err_3, linewidth=1.5, color='green', label='ePIE', alpha=0.7, marker='h', markevery=30)
# plt.semilogy(err_4, linewidth=1.5, color='purple', label='WASP', alpha=0.7,marker='d', markevery=30)
plt.semilogy(err_6, linewidth=1.5, color='grey', label='ADMM', alpha=0.7, marker='^', markevery=30)
plt.semilogy(err_7, linewidth=1.5, color='blue', label='LADMM', alpha=0.7, marker='o', markevery=30)
plt.semilogy(err_8, linewidth=1.5, color='red', label='TADMM', alpha=0.7, marker='*', markevery=30)
# plt.semilogy(err_10, linewidth=1.5, color='red', label='TADMM_tv', alpha=0.7, marker='o', markevery=30)
plt.legend(loc=3, prop={'size': 9})
plt.xlabel('iteration number')
plt.ylabel('relative error')
# plt.savefig('C:/Users/86364/Desktop/OL/wucha/16/Re_16_noise.png', dpi= 1000, bbox_inches ='tight')


# time_end =time.perf_counter()
# time_consume1 = time_start1 - time_start0
# time_consume2 = time_start2 - time_start1
# time_consume3 = time_start3 - time_start2
# time_consume4 = time_start4 - time_start3
# time_consume5 = time_start5 - time_start4
# print('iteration consume time ',time_consume1,time_consume2,time_consume3,time_consume4,time_consume5)
# #print('last frame  erro' , err_5)
# #phase correction
# out, object_re3 = dftregistration(object,object_re3, r= 50)
# out, object_re4 = dftregistration(object,object_re4, r= 50)
# out, object_re6 = dftregistration(object,object_re6, r= 50)
# out, object_re7 = dftregistration(object,object_re7, r= 50)
# out, object_re8 = dftregistration(object,object_re8, r= 50)
# out, object_re10 = dftregistration(object,object_re10, r= 50)

#object_re11 = phase_correct(object,object_re11)
#----------------------------------
# image
#----------------------------------
# cv2.imwrite("C:/Users/86364/Desktop/OL/wucha/16/image/noise/epie_ob.png",Normlize(np.abs(object_re3[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/OL/wucha/16/image/noise/epie_ph.png",Normlize(np.angle(object_re3[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/OL/wucha/16/image/noise/epie_pro.png",Normlize(np.abs(probe_re3),0,255))
# # #
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/rpie_ob4.png",Normlize(np.abs(object_re4[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/rpie_ph4.png",Normlize(np.angle(object_re4[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/rpie_pro4.png",Normlize(np.abs(probe_re4),0,255))
# # #
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/admm_ob4.png",Normlize(np.abs(object_re6[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/admm_ph4.png",Normlize(np.angle(object_re6[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/admm_pro4.png",Normlize(np.abs(probe_re6),0,255))

# cv2.imwrite("C:/Users/86364/Desktop/OL/wucha/16/image/noise/ladmm_ob.png",Normlize(np.abs(object_re7[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/OL/wucha/16/image/noise/ladmm_ph.png",Normlize(np.angle(object_re7[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/OL/wucha/16/image/noise/ladmm_pro.png",Normlize(np.abs(probe_re7),0,255))
#
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/tadmm_ob4.png",Normlize(np.abs(object_re8[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/tadmm_ph4.png",Normlize(np.angle(object_re8[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part//fangzhen/2/4/tadmm_pro4.png",Normlize(np.abs(probe_re8),0,255))
# # #
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/tadmm_ob_net4.png",Normlize(np.abs(object_re10[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/tadmm_ph_net4.png",Normlize(np.angle(object_re10[target_obj//2 - image_size//2:target_obj//2 + image_size//2,target_obj//2 - image_size//2:target_obj//2 + image_size//2]),0,255))
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/fangzhen/2/4/tadmm_pro_net4.png",Normlize(np.abs(probe_re10),0,255))
# #----------------------------------
# image the result
#----------------------------------
# #
fig, ax = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction ADMM')
ax = ax.flatten()
ax0=ax[0].imshow(np.abs(object_re6[50:350,50:350]) , cmap='gray'); ax[0].set_title('object amplitude')
ax1=ax[1].imshow(np.angle(object_re6[50:350,50:350]), cmap='gray'); ax[1].set_title('object phase ')
ax[2].imshow(np.abs(probe_re6), cmap='jet'); ax[2].set_title('probe amplitude')
ax[3].imshow(np.angle(probe_re6), cmap='jet'); ax[3].set_title('probe phase')
fig.colorbar(ax0,ax = ax[0]);fig.colorbar(ax1,ax = ax[1])
# # #
# fig, ax = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction LADMM')
# ax = ax.flatten()
# ax0=ax[0].imshow(np.abs(object_re7)/np.max(np.abs(object_re7)) , cmap='gray'); ax[0].set_title('object amplitude')
# ax1=ax[1].imshow(np.angle(object_re7), cmap='gray'); ax[1].set_title('object phase ')
# ax[2].imshow(np.abs(probe_re7)/np.max(np.abs(probe_re7)), cmap='gray'); ax[2].set_title('probe amplitude')
# ax[3].imshow(np.angle(probe_re7), cmap='gray'); ax[3].set_title('probe phase')
# fig.colorbar(ax0,ax = ax[0]);fig.colorbar(ax1,ax = ax[1])
# # plt.savefig("C:/Users/86364/Desktop/实验数据/真实实验结果/wf重建/近场重建仿真/rpie.png")
# # # # # # #
# fig, ax = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction RPIE')
# ax = ax.flatten()
# ax0=ax[0].imshow(np.abs(object_re4[50:350,50:350]) , cmap='gray'); ax[0].set_title('object amplitude')
# ax1=ax[1].imshow(np.angle(object_re4[50:350,50:350]), cmap='gray'); ax[1].set_title('object phase ')
# ax[2].imshow(np.abs(probe_re4)/np.max(np.abs(probe_re4)), cmap='gray'); ax[2].set_title('probe amplitude')
# ax[3].imshow(np.angle(probe_re4), cmap='gray'); ax[3].set_title('probe phase')
# fig.colorbar(ax0,ax = ax[0]);fig.colorbar(ax1,ax = ax[1])
# # # plt.savefig("C:/Users/86364/Desktop/实验数据/真实实验结果/wf重建/近场重建仿真/wf.png")
# # # # # #
# fig, ax = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction TADMM-TV')
# ax = ax.flatten()
# ax0=ax[0].imshow(np.abs(object_re10[50:350,50:350]) , cmap='gray'); ax[0].set_title('object amplitude')
# ax1=ax[1].imshow(np.angle(object_re10[50:350,50:350]), cmap='gray'); ax[1].set_title('object phase ')
# ax[2].imshow(np.abs(probe_re10), cmap='gray'); ax[2].set_title('probe amplitude')
# ax[3].imshow(np.angle(probe_re10), cmap='gray'); ax[3].set_title('probe phase')
# fig.colorbar(ax0,ax = ax[0]);fig.colorbar(ax1,ax = ax[1])
# #
fig, ax = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction TADMM')
ax = ax.flatten()
ax0=ax[0].imshow(np.abs(object_re8[50:350,50:350]) , cmap='gray'); ax[0].set_title('object amplitude')
ax1=ax[1].imshow(np.angle(object_re8[50:350,50:350]), cmap='gray'); ax[1].set_title('object phase ')
ax[2].imshow(np.abs(probe_re8), cmap='jet'); ax[2].set_title('probe amplitude')
ax[3].imshow(np.angle(probe_re8), cmap='jet'); ax[3].set_title('probe phase')
fig.colorbar(ax0,ax = ax[0]);fig.colorbar(ax1,ax = ax[1])

fig, ax = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction LADMM')
ax = ax.flatten()
ax0=ax[0].imshow(np.abs(object_re7[50:350,50:350]) , cmap='gray'); ax[0].set_title('object amplitude')
ax1=ax[1].imshow(np.angle(object_re7[50:350,50:350]), cmap='gray'); ax[1].set_title('object phase ')
ax[2].imshow(np.abs(probe_re7), cmap='jet'); ax[2].set_title('probe amplitude')
ax[3].imshow(np.angle(probe_re7), cmap='jet'); ax[3].set_title('probe phase')
fig.colorbar(ax0,ax = ax[0]);fig.colorbar(ax1,ax = ax[1])
# plt.savefig("C:/Users/86364/Desktop/实验数据/迭代五十次重建图像对比/rpie_50.png")
plt.show()
cv2.waitKey(0)