import numpy as np
import cv2
from util.Setpinhole import Setpinhole
from Algorithms.various_PIE_engine import various_PIE
import matplotlib.pyplot as plt
from util.Propagate import Propagate
import copy
import random
import imcrop
from util.Sampling import CT

sigma =1

#########参数设置#########
lamda = 632.8e-6     #wavelength 632.8
z2 = 32 #propagate_distance 28.38 （7 12.15）   （6  22.71）
pix = 4.65e-3  / sigma  #pix size5.5

m = 512
n = 512
r = 125
####### 数据加载 ###########
# p = loadmat("E:\pythonProject\pie-penalize\Diffraction_Patterns2\positions.mat")
# p = p['coords']
# p = p - (p[0] - 0.1)

p = np.load("pos_test.npy")
p = p - (p[0] - 0.3)
# p = p /(( lamda * z2 )/(512 * pix))
print(p)
p = p*215
print(p)
positions1 = np.floor(np.array(p))
#positions = positions[0:1]
positions2 =[[int(x) for x in positions_x] for  positions_x in positions1]
positions = copy.deepcopy(positions2)



print(positions)
# diffset = getphotos('E:\pythonProject\pie-penalize\Diffraction_Patterns2\yanshetuxixang')
diffset = np.load("diff_test.npy")
probe_r =np.zeros(shape=(m,n))
backnoise  = cv2.imread("C:/Users/86364/Desktop/yanshetuxiang/backnoise_test.tiff",cv2.IMREAD_ANYDEPTH)
for i in range(len(diffset)):

    diffset[i] = imcrop.imcrop(diffset[i] - backnoise, 512)
    diffset[i] = np.where(diffset[i] < 0, 0, diffset[i])
    # diffset[i] =  np.round((diffset[i] / np.max(diffset[i])) * 255).astype(np.uint8)
    diffset[i] =  diffset[i] / np.max(diffset[i])
    # probe_r += np.sqrt(diffset[i])

probe_temp = Setpinhole(m,n,r)
# diffset = center_find(abs(probe_temp),diffset,r = 1)
#
# probe_r = lowPassFilter.lowPassFiltering(probe_r,19)
# probe_r = fft.ifft2(probe_r)/len(diffset)
# probe_r = abs(fft.ifftshift(probe_r))
illuindy, illuindx = np.indices((probe_r.shape))
# probe_r = diffset[40]


k = 0
object = np.ones((1200,1200))
object_re_shape = object.shape
probe_r = Setpinhole(n, m, r=120)

#######像素超分辨###########
probe_sp = Setpinhole(int(m *sigma),int(n *sigma),int(r*sigma))
probe_sp = CT(diffset[0],sigma=sigma)
#probe_sp =  imcrop.imcrop(probe_sp, pixsum=150)


object_sp = np.ones((800*sigma,800*sigma))
object_re_shape_sp = object_sp.shape

positions_sp = [[int(x*sigma) for x in positions_x] for  positions_x in positions1]

illuindy_sp, illuindx_sp = np.indices((probe_sp.shape))
##构造随机读取顺序
a = (int)(len(positions))
lis =list(range(a))
random.shuffle(lis)
# dis_crrect(k,15,25,1,diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,'ePIE',object,lis)
object_re1, probe_re1, err= various_PIE(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,'ePIE',object,lis)
# object_re1, probe_re1, err= WF.MY_ADMM(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,object,lis)
# object_re, probe_re, err_=ADMM_Py.ADMM_Py_ps(k, diffset, probe_sp, object_re_shape_sp , positions_sp, illuindy_sp, illuindx_sp,'Fresnel',z2,lamda,pix,object,sigma)
# object_re1,probe_re1, err=ADMM_Py.ADMM_net_denoise(k, diffset, probe_r, object_re_shape, positions, illuindy, illuindx,'Fresnel',z2,lamda,pix,object, lis)
# object_re1, probe_re1, err_1= ADMM_Py.ADMM_Py_Aps(k, diffset, probe_sp,object_re_shape_sp, positions_sp, illuindy_sp, illuindx_sp,'Fresnel',z2,lamda,pix,object,lis,sigma=sigma,lamda=1e-3,colour='g')
# plt.savefig("six.png",dpi= 1000,bbox_inches = 'tight') #save p
# postion_corection(1, diffset, probe_r, 28.38, 'Fresnel', lamda, pix,40)

object_re1 = Propagate(diffset[10], 'Fresnel', pix, lamda, -34.5)

# object_re1 = object_re1[200*sigma:600*sigma,200 *sigma :600*sigma]
# object_re2 = object_re2[200*sigma:600*sigma,200 *sigma :600*sigma]
# object_re1 = object_re1[400*sigma:700*sigma,400 *sigma :700*sigma]
# object1 = abs(object_re1)
# # object2 = abs(object_re2 )
# object1 = cv2.flip(object1,flipCode = 0)
# object1 = cv2.rotate(object1, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
object1 = abs(object_re1) / np.max(abs(object_re1))
object1_ph = np.angle(object_re1)
object1_ph = cv2.rotate(object1_ph , cv2.ROTATE_90_CLOCKWISE)
object1 = cv2.rotate(object1, cv2.ROTATE_90_CLOCKWISE)

#
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/usaf/ePIE_am.png",Normlize(object1,0,255))
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/usaf/ePIE_ph.png",Normlize(object1_ph,0,255))
# cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/5part/usaf/ePIE_pr.png",Normlize(np.abs(probe_re1),0,255))


# object2 = cv2.flip(object2,flipCode = 0)
# object2 = cv2.rotate(object2, cv2.ROTATE_90_COUNTERCLOCKWISE)
#object1 = object1 / np.max(object1)*255
# object2 = object2 / np.max(object2)*255
#cv2.imwrite("object_WF.bmp",object1 )
# cv2.imwrite("object_epie.bmp",object2)
fig, ((x1,x2), (x3,x4)) = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction')
x1.imshow(diffset[50]); x1.set_title('object amplitude')
x2.imshow(object1_ph, cmap='gray'); x2.set_title('object phase ')
x3.imshow(np.abs(probe_re1), cmap='gray'); x3.set_title('probe amplitude')
x4.imshow(np.angle(probe_re1), cmap='gray'); x4.set_title('probe phase')
# fig, ((x1,x2), (x3,x4)) = plt.subplots(2, 2, gridspec_kw={ 'wspace': 0 , 'hspace':0.5}); plt.suptitle('object and probe reconstruction')
# x1.imshow(np.abs(object2) , cmap='gray'); x1.set_title('object amplitude')
# x2.imshow(np.angle(object2), cmap='gray'); x2.set_title('object phase ')
# x3.imshow(np.abs(probe_re2), cmap='gray'); x3.set_title('probe amplitude')
# x4.imshow(np.angle(probe_re2), cmap='gray'); x4.set_title('probe phase')


plt.show()