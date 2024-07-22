import cv2
import numpy as np

def calculate_image_sharpness(image):
    # 读取图像

    # 转换为灰度图

    # 使用Sobel算子计算x和y方向的边缘强度
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    # 计算锐度
    abs_sobel64f = np.absolute(sobelx) + np.absolute(sobely)
    sharpness = np.mean(abs_sobel64f)
    return sharpness