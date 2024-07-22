import numpy as np
import cv2
from util.subpixel_registration import  dftregistration
from util.quancomp import quancomp
from  util.phase_unwrap import  phase_unwrap
from util.Normlize import Normlize
from scipy.linalg import norm
from util.imshift import imshift
'''***************** 
 The error calculation formula comes from "An improved ptychographical phase retrieval algorithm
for diffractive imaging" ,Andrew M. Maiden, John M. Rodenburg

    ***************** '''
def relative_err( image1, image2, pixsum=130): #image1 is reference image,image2 is ismatch




  out, grey = dftregistration(image1, image2, r= 50)  #grey is the image after registration
  # grey = abs(imshift(abs(image2),out[2],out[3]))*np.exp(1j*imshift(np.angle(image2),out[2],out[3]))
  [row, col] = np.shape(image1)
  x_center = col // 2
  y_center =  row // 2
  abstract1  = image1[( x_center- pixsum // 2) : ( x_center+pixsum // 2), ( y_center- pixsum // 2) : ( y_center+pixsum // 2)]
  abstract2  = grey[( x_center- pixsum // 2) : ( x_center+pixsum // 2), ( y_center- pixsum // 2) : ( y_center+pixsum // 2)]
  # abstract1_angle = Normlize(np.angle(abstract1),0,1)
  # abstract2_angle = Normlize(np.angle(abstract2),0,1)

  ab_ph1 = np.angle(abstract1)
  ab_ph2 = np.angle(abstract2)
  abstract1_ab = Normlize(abs(abstract1), 0, 1)
  abstract2_ab = Normlize(abs(abstract2), 0, 1)
  ab_ph2 = ab_ph2 - np.mean(ab_ph2) + np.mean(ab_ph1)
  # ab_ph1 = Normlize(ab_ph1, 0, 1)
  # ab_ph2 = Normlize(ab_ph2, 0, 1)
  x_gr =  abstract1_ab * np.exp(1j * ab_ph1)
  x_es =  abstract2_ab * np.exp(1j * ab_ph2)
  # gama1 = np.sum(abstract2*np.conj(abstract1)) / np.sum(np.abs(abstract1)**2)
  # gama2 = np.sum(abstract1 * np.conj(abstract2)) / np.sum(np.abs(abstract2) ** 2)
  #cv2.imshow("register image", abs(abstract2))
  # E = np.sum(np.abs(abstract1 -gama2* abstract2)**2) / np.sum(np.abs(abstract1)**2)
  # E = -10 * np.log10(np.sum(abs(gama2* abstract2 -abstract1)**2)/np.sum(abs(gama2*abstract2)**2))
  E = np.sum(np.abs(abs(x_gr) - abs(x_es))**2) / np.sum(np.abs(x_gr)**2)



  return   E

def phase_correct(image1,image2,pixsum=250):


    out1, grey = dftregistration(image1, image2, r= 1)  # grey is the image after registration

    [row, col] = np.shape(image1)
    x_center = col // 2
    y_center = row // 2
    abstract1 = image1[(x_center - pixsum // 2): (x_center + pixsum // 2),
                (y_center - pixsum // 2): (y_center + pixsum // 2)]
    abstract2 = grey[(x_center - pixsum // 2): (x_center + pixsum // 2),
                (y_center - pixsum // 2): (y_center + pixsum // 2)]

    gama = np.sum(abstract1 * np.conj(abstract2)) / np.sum(np.abs(abstract2) ** 2)

    return gama*abstract2

if __name__ == '__main__':
 or_am = np.load('E:/pythonProject/pie—penalize/image_data/or_am.npy')
 or_ph = np.load('E:/pythonProject/pie—penalize/image_data/or_ph.npy')

 re_am = np.load('E:/pythonProject/pie—penalize/image_data/re_am.npy')
 re_ph = np.load('E:/pythonProject/pie—penalize/image_data/re_ph.npy')
 or_image = or_am * np.exp(1j *or_ph)
 re_image = re_am * np.exp(1j * re_ph)
 # or_image = np.array(or_image)
 # re_image = np.array( re_image)




 E = relative_err(or_image,   re_image, pixsum=100)
 print(E)
 cv2.waitKey(0)

