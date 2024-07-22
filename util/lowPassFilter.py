import numpy as np

def lowPassFiltering(img,size):#传递参数为傅里叶变换后的频谱图和滤波尺寸
    h, w = img.shape[0:2]#获取图像属性
    h1,w1 = int(h/2), int(w/2)#找到傅里叶频谱图的中心点
    img2 = np.zeros((h, w), dtype=float)#定义空白黑色图像，和傅里叶变换传递的图尺寸一致
    img2[h1-int(size/2):h1 +int(size/2), w1-int(size/2):w1+int(size/2)] = 1#中心点加减滤波尺寸的一半，刚好形成一个定义尺寸的滤波大小，然后设置为1，保留低频部分
    img3=img2*img #将定义的低通滤波与传入的傅里叶频谱图一一对应相乘，得到低通滤波
    return img3