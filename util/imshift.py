import numpy as np
from numpy import fft
from math import pi
import  cv2

def imshift(img, s1, s2):
    image2 = fft.fft2(img)
    [nr, nc] = np.shape(image2)
    x = np.array(np.arange(-np.fix(nr / 2), np.ceil(nr / 2), 1))
    y = np.array(np.arange(-np.fix(nc / 2), np.ceil(nc / 2), 1))
    Nr = fft.ifftshift(x)
    Nc = fft.ifftshift(y)
    [Nc, Nr] = np.meshgrid(Nc, Nr)
    img = image2 * np.exp(1j * 2 * pi * (-s1 * Nr / nr - s2 * Nc / nc))
    img =  fft.ifft2(img)



    # [rows ,cols] = img.shape
    # x = np.array(np.arange(-rows / 2, rows / 2, 1))
    # y = np.array(np.arange(-cols / 2, cols / 2, 1))
    # [nr, nc] = np.meshgrid(x, y)
    # img = fft.ifft2(fft.fftshift(fft.fft2(img)) * np.exp(-1j * 2 * pi * (s1 * nr / rows + s2 * nc / cols)))

    return img




if __name__ == '__main__':


 f = cv2.imread('E:/pythonProject/pie-penalize/image_data/Set12/lena256.jpg')
 f_gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
 dist=np.ones(f_gray.shape)
 cv2.normalize(f_gray,  dist, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
 print(dist)
 shift_image = imshift(dist,240,240)
 print(abs(shift_image))
 cv2.imshow("lena" , abs(f_gray))
 cv2.imshow("shift_lena",abs(shift_image))
 cv2.waitKey(0)