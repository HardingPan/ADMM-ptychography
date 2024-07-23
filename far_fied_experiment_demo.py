import numpy as np
from numpy import  fft
import cv2
from util.Setpinhole import Setpinhole
import matplotlib.pyplot as plt
from Algorithms import WF, ADMM_Py
import copy
from util.center_find import center_find2
import random

from  util import lowPassFilter
import  time
from  util.random_sampling import random_sampling
from util.Normlize import Normlize
from Algorithms import  ADMM_Py
from Algorithms.various_PIE_engine import *
sigma = 1

def imcrop(image, pixsum=100):
    [row, col] = np.shape(image)
    x_center = col // 2
    y_center = row // 2
    abstract = image[(y_center - pixsum // 2) : (y_center + pixsum // 2),
                     (x_center - pixsum // 2): (x_center + pixsum // 2)]
    return abstract

#########参数设置#########
lamda = 632.8e-6     #wavelength 637
z2 = 31   #propagate_distance 10.77
pix = 2*4.65e-3  / sigma  #pix size4.5

m = 200
n =  200
r = 90
####### 数据加载 ###########
# p = loadmat("positions.mat")
# p = p['coords']
p = np.load("../data/positions_test.npy")
p = p - (p[0] - 0.3)
delta_r = 1/(lamda *50/(m * pix))

print(delta_r)
p = p * delta_r
positions1 = np.floor(np.array(p))
#positions = positions[0:1]
positions1 =[[int(x) for x in positions_x] for  positions_x in positions1]
positions = copy.deepcopy(positions1)

print(positions)
# diffset = getphotos('C:/Users/86364/Desktop/yanshetuxiang/yanshetuxiang')
diffset = np.load("../data/diffset_test.npy")
backnoise  = cv2.imread("../data/backnoise.tiff",cv2.IMREAD_ANYDEPTH)

for i in range(len(diffset)):

    diffset[i] = (diffset[i] - backnoise)
    diffset[i] = np.where(diffset[i] < 0, 0, diffset[i])

diffset = center_find2(diffset, r=1)  #将衍射强度图的低频部分移到中心
print(diffset[0].shape)
probe_r =np.zeros(shape=(m,n))
diffsets = np.zeros(shape=(len(diffset),m,m))

for i in range(len(diffset)):

    diffsets[i] = imcrop(diffset[i] , m)
    # diffsets[i] = diffsets[i] / np.max(diffsets[i])
    probe_r += np.sqrt(diffsets[i])
print(len(diffset))
probe_temp = Setpinhole(m,n,r)

#
probe_r = lowPassFilter.lowPassFiltering(probe_r,17)   ##17
probe_r = fft.ifft2(probe_r)/len(diffsets)
probe_r = abs(fft.ifftshift(probe_r))
probe_r = probe_r / np.max(probe_r)
illuindy, illuindx = np.indices((probe_r.shape))

k = 500
## 防止重建图像过大
size_rows = max(positions[len(positions) -1][0], positions[len(positions) -1][1]) + diffsets[0].shape[0] + 30
size_cols = max(positions[len(positions) -1][0], positions[len(positions) - 1 ][1]) + diffsets[0].shape[1] + 30
object = np.ones(shape=(size_rows,size_cols))
print(object.shape)
object_re_shape = object.shape
# probe_r = Setpinhole(m,n,1.5*delta_r/2)
#######像素超分辨###########
probe_sp = Setpinhole(int(m *sigma),int(n *sigma),int(r*sigma))

#probe_sp = imcrop.imcrop(probe_sp, pixsum=150)


object_sp = np.ones((1024*sigma,1024*sigma))
object_re_shape_sp = object_sp.shape

positions_sp = [[int(x)*sigma for x in positions_x] for positions_x in positions]

illuindy_sp, illuindx_sp = np.indices((probe_sp.shape))
# positions , diffsets = random_sampling(positions, diffsets, num_samples = 64)
##构造随机读取顺序
a = (int)(len(positions))
lis =list(range(a))
random.shuffle(lis)
#

# probe_r = center_find2(probe_r, r=0)

# probe_r = imcrop.imcrop(probe_r,m)B
# probe_r = probe_r / np.max(probe_r)
time_start=time.perf_counter()
# dis_crrect(k,10.5,11,0.1,diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,'ePIE',object,lis)
object_re1, probe_re1, err=various_PIE(k, diffsets, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,'rPIE',object,lis)
# object_re5, probe_re5, err=various_PIE(k, diffsets, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,'rPIE',object,lis)
# object_re4, probe_re4, err_=WF.ADMM(k, diffsets, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object,lis)
# object_re4, probe_re4, err_=WF.L_ADMM(k, diffsets, probe_r, object_re_shape, positions, illuindy, illuindx,'Fourier',z2,lamda,pix,object,lis)
# object_re5, probe_re5, err= WF.T_ADMM(k, diffsets, probe_r, object_re_shape, positions, illuindy, illuindx, 'Fourier', z2, lamda, pix, object, lis, is_center_correct = 1)
# object_re6,probe_re6, err= ADMM_Py.ADMM_net_denoise(k, diffsets, probe_r, object_re_shape, positions, illuindy, illuindx, 'Fourier', z2, lamda, pix, object, lis)

time_end =time.perf_counter()
time_consume=time_end-time_start
print("time consume ", time_consume)
plt.savefig("six.png",dpi= 1000,bbox_inches = 'tight')  #save


# out, object_re5 = dftregistration(object_re6, object_re5, r= 50)

#image = Propagate(diffset[0], 'Fresnel', pix, lamda, -31)

object1 = abs(object_re1) / np.max(abs(object_re1))
object1_ph = np.angle(object_re1)
object1_ph = cv2.rotate(object1_ph , cv2.ROTATE_90_CLOCKWISE)
object1 = cv2.rotate(object1, cv2.ROTATE_90_CLOCKWISE)


# #
# object5 = abs(object_re5) / np.max(abs(object_re5))
# object5_ph = np.angle(object_re5)
# object5_ph = cv2.rotate(object5_ph , cv2.ROTATE_90_CLOCKWISE)
# object5 = cv2.rotate(object5, cv2.ROTATE_90_CLOCKWISE)

# ax0=ax[0].imshow(abs(object5) , cmap='gray'); ax[0].set_title('object amplitude')
# ax1=ax[1].imshow(object5_ph, cmap='gray'); ax[1].set_title('object phase ')
# ax[2].imshow(np.abs(probe_re5), cmap='jet'); ax[2].set_title('probe amplitude')
# ax[3].imshow(np.angle(probe_re5), cmap='jet'); ax[3].set_title('probe phase')
# fig.colorbar(ax0,ax = ax[0]);fig.colorbar(ax1,ax = ax[1])

fig, ax = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction')
ax = ax.flatten()
ax0=ax[0].imshow(abs(object1) , cmap='gray'); ax[0].set_title('object amplitude')
ax1=ax[1].imshow(object1_ph, cmap='gray'); ax[1].set_title('object phase ')
ax[2].imshow(np.abs(probe_re1), cmap='jet'); ax[2].set_title('probe amplitude')
ax[3].imshow(np.angle(probe_re1), cmap='jet'); ax[3].set_title('probe phase')
fig.colorbar(ax0,ax = ax[0]);fig.colorbar(ax1,ax = ax[1])


plt.show()