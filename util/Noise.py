import numpy as np
import cv2
from util.Normlize import  Normlize
from matplotlib import pyplot as plt

# SNR caculate
def SNR(image ,noise_image ):

     image = image / np.max(image)
     noise_image = noise_image / np.max(noise_image)
     image2 = image - noise_image
     image1 = image
     image11 = image1 ** 2
     image22 = image2 ** 2
     p = np.sum(image11 )
     d = np.sum(image22)

     snr = -10 * np.log10(d / p)

     return snr


def gasuss_noise(image, mean=0, sigma=0.002, sum = 0 ):
    """"
    add gasuss noise
    :param image: input image
    :param mu: mean
    :param sigma: std
    :return: image with noise
    """
    max = np.max(image)
    image = image / max
    [rows, cols] = image.shape
    noise = np.random.randn(rows, cols)*sigma

    gauss_noise = image*(1+noise)
    gauss_noise = gauss_noise * max
    gauss_noise = np.clip(gauss_noise,0,np.max(gauss_noise))

    print(SNR(image,gauss_noise))
    sum = sum + SNR(image, gauss_noise)
    print(sum)


    return gauss_noise , sum



def poisson_noise(img, photon_number):
   """
        An image corresponding to the number of photons is generated from the input image
        the noise of image is poisson distribution
   """

   # max = np.max(img)
   # min = np.min(img)
   # img1 = (img - min) / (max -min)
   #
   # N = np.size(img)
   #
   # P = img1 / np.sum(img1)
   # P = P.reshape((N, 1))
   # postion = np.array(np.arange(1, N+1))#pixel Location distribution
   #
   # photon_number = np.random.choice(postion, photon_number, p=P.ravel()) #Location distribution of photon_number ,the size equals photon_number
   # out,h = np.histogram(photon_number.ravel(),N)
   # out = (out - np.min(min)) / (np.max(out) -np.min(out))
   # out = out * ( max - min) + min
   #
   # out3 = out.reshape((img.shape))
   max = np.max(img)
   total_energy = np.sum(img)
   energy_per_pixel = img.astype(np.float64) / total_energy * photon_number
   #energy_per_pixel =  total_energy / photon_number
   noise = np.random.poisson(energy_per_pixel)
   #noise = noise / photon_number
   out3  = noise / np.max(noise) * max
   output = np.clip(out3, 0, np.max(out3))
   print(SNR(img, output))

   return  output


#
def poisson_noise2(image, alpha, sum):


   [rows, cols] = image.shape

   image_noise = image + np.random.normal(loc=0, scale = alpha*np.sqrt(image), size=(rows, cols))



   print(SNR(image,image_noise))
   sum = sum + SNR(image,image_noise)
   print(sum)

   return image_noise,sum











if __name__ == '__main__':

    # ----------------------读取图片-----------------------------
    img = obj_ph = cv2.imread('E:/pythonProject/pie-penalize/image_data/Set12/lena256.png',0)
    # --------------------添加噪声---------------------------
    # out2,sum = poisson_noise2( img ,4, 0)
    # out2 = out2.astype(np.uint8)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #out2 = poisson_noise(f_gray, 100000)
    # ----------------------显示结果-----------------------------
    cv2.imshow('origion_pic', img )
    cv2.imshow('gauss_noise', img/np.max( img ))
    cv2.imwrite("C:/Users/86364/Desktop/biyelunwen/image/4part/lena_.png", Normlize(img, 0, 255))


    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文显示为黑体
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.title('无噪声图像直方图分布')
    plt.xlabel('像素分布')
    plt.ylabel('像素数')
    plt.plot(hist)
    plt.xlim([0, 256])

    # 保存直方图到文件
    histogram_path = 'C:/Users/86364/Desktop/biyelunwen/image/4part/grayscale_histogram_wuzaosheng.png'
    plt.savefig(histogram_path)
    cv2.waitKey(0)



