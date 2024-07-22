import numpy as np
from util.subpixel_registration import  dftregistration
from skimage.metrics import structural_similarity as ssim
from  util.phase_unwrap import  phase_unwrap
def PSNR(re_image,test_image):

    #归一化，将像素值转换为0-1之间的值
    re_image = re_image / np.max(re_image)
    test_image = test_image /np.max(test_image)
    #计算均方误差
    mse = np.mean((re_image - test_image) ** 2)
    #计算峰值信噪比
    pnsr = 20*np.log10( 1 / np.sqrt(mse))

    #pnsr = 20*np.log10(np.sum(test_image**2)/np.sum((re_image-test_image)**2))
    return pnsr
def RMSE(image_gr, image_re):
    #计算图像的RMSE
    rows,cols = image_gr.shape
    RMSE = np.sqrt(np.sum((image_gr-image_re)**2 /(rows * cols) ))
    return RMSE

def complex_PNSR(image1, image2, pixsum = 250):


    #使用dftregistration函数进行图像的配准
    out, image2 = dftregistration(image1, image2, r=40)
    #获取图像的中心位置
    [row, col] = np.shape(image1)
    x_center = col // 2
    y_center = row // 2
    #截取图像的中心区域
    abstract1 = image1[(x_center - pixsum // 2): (x_center + pixsum // 2),
                (y_center - pixsum // 2): (y_center + pixsum // 2)]
    abstract2 = image2[(x_center - pixsum // 2): (x_center + pixsum // 2),
                (y_center - pixsum // 2): (y_center + pixsum // 2)]


    ##解决相位模糊问题，注意这可能并不是一个良好的解决方式，特别是相位部分，用psnr和ssim感觉评估并不是很准确
    #计算振幅和相位的gamma系数
    gama = np.sum(abstract1 * np.conj(abstract2)) / np.sum(np.abs(abstract2) ** 2)
    #使用gamma系数对第二个图像进行调整
    abstract2 = gama*abstract2
    #计算振幅和相位的PSNR
    abstract1_am = abs(abstract1)
    abstract1_ph = np.angle(abstract1)
    abstract2_am =  abs(abstract2)
    abstract2_ph = np.angle(abstract2)

    psnr_am = PSNR(abstract1_am, abstract2_am)
    psnr_ph = PSNR(abstract1_ph, abstract2_ph)
    #计算振幅和相位的SSIM
    ssim_am,_ = ssim(abstract1_am, abstract2_am, data_range=np.max(abstract1_am) - np.min(abstract1_am) , full = True)
    ssim_ph,_ = ssim(abstract1_ph, abstract2_ph, data_range=np.max(abstract2_am) - np.min(abstract2_am) ,full = True)
    #计算RMSE
    rmse = RMSE(abstract1_ph, abstract2_ph )
    #print('振幅pnsr=',psnr_am,'\n相位pnsr=',psnr_ph)
    return psnr_am, psnr_ph , ssim_am, ssim_ph, rmse












