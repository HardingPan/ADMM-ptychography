import numpy as np
import cv2
import time
import math
from util.evalution import complex_PNSR
from numba import jit
import bm3d
from FFDNet_1.test_ffdnet_ipol import ffdnet
from FFDNet_1.models import FFDNet


class denosise(object):

     def meandeblur(self,filter_size):

        # image = self
        # k = (filter_size - 1) // 2
        # [rows, cols] = np.shape(image)
        # output = np.zeros((rows, cols) , dtype=self.dtype)
        # for i in range(k, rows - k):
        #   for j in range(k, cols - k):
        #        neighbors = image[i-k:i+k+1, j-k:j+k+1].ravel()
        #        mean_value = np.mean(neighbors)
        #        output[i, j] = mean_value
        am = abs(self)
        ph = np.angle(self)
        am_r = cv2.blur(am, [filter_size, filter_size])
        ph_r = cv2.blur(ph, [filter_size, filter_size])
        output = am_r * np.exp(1j*ph_r)
        return output

     def mediandeblur(self, filter_size):
         # image = self
         # k = (filter_size - 1) // 2
         # [rows, cols] = np.shape(image)
         # output = np.zeros((rows, cols) , dtype=self.dtype)
         # for i in range(k, rows - k):
         #   for j in range(k, cols - k):
         #        neighbors = image[i-k:i+k+1, j-k:j+k+1].ravel()
         #        mean_value = np.median(neighbors)
         #        output[i, j] = mean_value

         am = abs(self)
         ph = np.angle(self)
         am_r = cv2.medianBlur(am, filter_size)
         ph_r = cv2.medianBlur(ph,  filter_size)

         output = am_r * np.exp(1j * ph_r)
         return output
         return output


     def gaussiandeblur(self, filter_size):
         am = abs(self)
         ph = np.angle(self)
         am_r = cv2.GaussianBlur(am, filter_size)
         ph_r = cv2.GaussianBlur(ph, filter_size)
         output = am_r * np.exp(1j * ph_r)
         return output

     @jit
     def TV(self, iter, dt, epsilon, lamb):

         m_imgData = self
         [NX,NY] =m_imgData.shape
         ep2 = epsilon * epsilon
         I_t = m_imgData.astype(np.float64)
         I_tmp = m_imgData.astype(np.float64)

         for t in range(iter):
             for i in range(NY):  # 一次迭代
                 for j in range(NX):
                     iUp = i - 1
                     iDown = i + 1
                     jLeft = j - 1
                     jRight = j + 1  # 边界处理
                     if i == 0:
                         iUp = i
                     if NY - 1 == i:
                         iDown = i
                     if j == 0:
                         jLeft = j
                     if NX - 1 == j:
                         jRight = j
                     tmp_x = (I_t[i][jRight] - I_t[i][jLeft]) / 2.0
                     tmp_y = (I_t[iDown][j] - I_t[iUp][j]) / 2.0
                     tmp_xx = I_t[i][jRight] + I_t[i][jLeft] - 2 * I_t[i][j]
                     tmp_yy = I_t[iDown][j] + I_t[iUp][j] - 2 * I_t[i][j]
                     tmp_xy = (I_t[iDown][jRight] + I_t[iUp][jLeft] - I_t[iUp][jRight] - I_t[iDown][jLeft]) / 4.0
                     tmp_num = tmp_yy * (tmp_x * tmp_x + ep2) + tmp_xx * (
                                 tmp_y * tmp_y + ep2) - 2 * tmp_x * tmp_y * tmp_xy
                     tmp_den = math.pow(tmp_x * tmp_x + tmp_y * tmp_y + ep2, 1.5)
                     I_tmp[i][j] += dt * (tmp_num / tmp_den + ( lamb) * (m_imgData[i][j] - I_t[i][j]))


             I_t = I_tmp  # 迭代结束


         return I_t  # 返回去噪图

     def bm3d1(self,sigma):

         out = bm3d.bm3d(self,sigma_psd=sigma)

         return out




     def ffdnet_(self, model, sigma,model_fn):

         out = ffdnet(self, model, sigma,model_fn)

         return out

     def hinet_(self, model):

         out = hinet(self, model)

         return out

     def tvdeblurr(self,sigma):
         am = abs(self)
         ph = np.angle(self)
         am_r = denosise.TV(am, 5 ,0.1, 0.5,0.2) #2
         ph_r = denosise.TV(ph, 5,0.1, 0.5,0.2)
         output = am_r * np.exp(1j * ph_r)
         return output

     def bm3ddeblur(self, sigma):
         am = abs(self)
         ph = np.angle(self)
         am_r = denosise.bm3d1(am, sigma)
         ph_r = denosise.bm3d1(ph, sigma)
         output = am_r * np.exp(1j * ph_r)
         return output

     def FFDNetdeblur(self, model, sigma, model_fn):
         am = abs(self)
         #am = np.real(self)
         ph = np.angle(self)
         # ph = np.imag(self)
         am_r = denosise.ffdnet_(am, model, sigma, model_fn)
         ph_r = denosise.ffdnet_(ph, model, sigma / 5, model_fn)

         output = am_r * np.exp(1j * ph_r)
         #output = am_r+1j*ph_r
         return output

     def HINetdeblur(self, model):
         am = abs(self)
         ph = np.angle(self)
         am_r = denosise.hinet_(am, model)
         ph_r = denosise.hinet_(ph, model)
         output = am_r * np.exp(1j * ph_r)
         return output
if __name__ =='__main__':

 re_am = np.load('E:/pythonroject/pie-penalize/image_data/re_am.npy')
 re_ph = np.load('E:/pythonProject/pie-penalize/image_data/re_am.npy')
 # re_am = cv2.cvtColor( re_am,cv2.COLOR_BGR2GRAY)
 # re_ph = cv2.cvtColor( re_ph,cv2.COLOR_BGR2GRAY)
 re_image = re_am * np.exp(1j * re_ph)
 #######模型创建##########

 # print('Loading model ...\n')
 # net = FFDNet(num_input_channels=1)

 # create model
 opt = parse_options(is_train=False)
 model=create_model(opt)
 h = denosise
 # re_am = np.array(re_am)
 time_start1=time.time()
 #tv2 =h. FFDNetdeblur(re_image,net,0.05)  # 0.6

 tv2 =h.HINetdeblur(re_image,model)  # 0.6
 #re_am1 = h.medianblur(re_image,3)q
 time_end1=time.time()
 time1 = time_end1 -time_start1
 psnr_am, psnr_ph,_,_ = complex_PNSR(re_image,tv2)
 print('振幅pnsr=',psnr_am,'\n相位pnsr=',psnr_ph)
 print(time1)
 cv2.imshow("zhenfu",np.abs(tv2))
 cv2.imshow("xiangwei" , np.angle(tv2))
 cv2.imshow("yuantu",re_am)
 cv2.waitKey(0)

