import numpy as np
from scipy.io import loadmat
import cv2
from util.dataload import getphotos
from util.Setpinhole import Setpinhole
from Algorithms.various_PIE_engine import various_PIE
import matplotlib.pyplot as plt
from util.Propagate import Propagate
import random
from Algorithms import ADMM_Py
sigma = 1

#########参数设置#########
lamda = 637e-6     #wavelength 637
z2 = 10.77        #propagate_distance 10.77
pix = 5.5e-3  / sigma  #pix size5.5

m = 512
n = 512
r = 90
####### 数据加载 ###########
p = loadmat("new_loc.mat")

positions = np.floor(np.array(p['new_loc']))

positions =[[int(x) for x in positions_x] for  positions_x in positions]
print(positions)


diffset = getphotos('E:\pythonProject\pie-penalize\Diffraction_Patterns')
a = diffset[10]
for i in range(len(diffset)):

    diffset[i] = diffset[i] / np.max(diffset[i])


probe_r = Setpinhole(m,n,r)

illuindy, illuindx = np.indices((probe_r.shape))
k= 20
object = np.ones((1024,1024))
object_re_shape = object.shape


#######像素超分辨###########
probe_sp = Setpinhole(int(m *sigma),int(n *sigma),int(r*sigma))
#probe_sp =  imcrop.imcrop(probe_sp, pixsum=150)


object_sp = np.ones((1024*sigma,1024*sigma))
object_re_shape_sp = object_sp.shape

positions_sp = [[int(x)*sigma for x in positions_x] for  positions_x in positions]

illuindy_sp, illuindx_sp = np.indices((probe_sp.shape))

a = (int)(len(positions))
lis =list(range(a))
random.shuffle(lis)

#dis_crrect(k,10.5,11,0.1,diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,'ePIE',object,lis)
object_re2, probe_re2, err=various_PIE(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,'ePIE',object,lis)
# object_re, probe_re, err_=WF.MY_ADMM(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,object,lis)
#object_re, probe_re, err_=ADMM_Py.ADMM_Py_ps(k, diffset, probe_sp, object_re_shape_sp , positions_sp, illuindy_sp, illuindx_sp,'Fresnel',z2,lamda,pix,object,sigma)
# object_re2,probe_re2, err=ADMM_Py.ADMM_net_denoise(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,object, lis)
plt.savefig("six.png",dpi= 1000,bbox_inches = 'tight') #save p
#postion_corection(1, diffset, probe_r, z2, 'Fresnel', lamda, pix,r-20)
image = Propagate(diffset[1], 'Fresnel', pix, lamda, -z2 )


# object_re = object_re[400*sigma:700*sigma,400 *sigma :700*sigma]
object_re2 = object_re2[400*sigma:700*sigma,400 *sigma :700*sigma]
# object1 = abs(object_re)
object2 = abs(object_re2 )
# object1 = cv2.flip(object1,flipCode = 1)
object2 = cv2.flip(object2,flipCode = 1)
# object1 = object1 / np.max(object1)*255
# object2 = object2 / np.max(object2)*255
# cv2.imwrite("object_WF.bmp",object1 )
# cv2.imwrite("object_epie.bmp",object2)
# fig, ((x1,x2), (x3,x4)) = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction')
# x1.imshow(np.abs(object1) , cmap='gray'); x1.set_title('object amplitude')
# x2.imshow(np.angle(object1), cmap='gray'); x2.set_title('object phase ')
# x3.imshow(np.abs(probe_re), cmap='gray'); x3.set_title('probe amplitude')
# x4.imshow(np.angle(probe_re), cmap='gray'); x4.set_title('probe phase')
fig, ((x1,x2), (x3,x4)) = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction')
x1.imshow(np.abs(image) , cmap='gray'); x1.set_title('object amplitude')
x2.imshow(np.angle(object2), cmap='gray'); x2.set_title('object phase ')
x3.imshow(np.abs(probe_re2), cmap='gray'); x3.set_title('probe amplitude')
x4.imshow(np.angle(probe_re2), cmap='gray'); x4.set_title('probe phase')
plt.show()