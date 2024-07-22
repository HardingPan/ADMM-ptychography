import numpy as np
import matplotlib.pyplot as plt
import cv2


def center_find(probe, diffset, r = 1):  ## r=1时输入的是diffset输入的列表，r==0时输入的是单幅图像


    thresh,probe_r = cv2.threshold(probe, 0.01, 1, cv2.THRESH_BINARY) #阈值分割
    probe_r =  cv2.medianBlur(np.uint8(probe_r),7)
    radius = 50 # 圆形半径
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)) # 执行开运算
    probe_r = cv2.morphologyEx(probe_r, cv2.MORPH_OPEN, kernel)
    ##计算图像质心
    M = cv2.moments(probe_r)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])

    ## 计算图像中心
    image_center_x = probe_r.shape[1] // 2
    image_center_y = probe_r.shape[0] // 2
    #计算平移距离
    shift_x = image_center_x - centroid_x
    shift_y = image_center_y - centroid_y
    ##平移图像
    rows, cols = probe_r.shape
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    if r == 1:
     for i in range(len(diffset)):
       diffset[i] = cv2.warpAffine(diffset[i], translation_matrix, (cols, rows))
    else:
       diffset = cv2.warpAffine(diffset, translation_matrix, (cols, rows))

    return diffset



def center_find2(diffset, r = 1):  ## r=1时输入的是diffset输入的列表，r==0时输入的是单幅图像,对衍射图像聚焦中心移动到图像中心



    # ##计算图像质心
    # M = cv2.moments(diffset[10])
    # centroid_x = int(M['m10'] / M['m00'])
    # centroid_y = int(M['m01'] / M['m00'])
    temp = 0
    if r==1:
        # for i  in range(len(diffset)):
          temp = diffset[0]
    else:
        temp = diffset
    max = np.max( temp )
    centroid_y,centroid_x = np.where(temp == max)

    ## 计算图像中心
    image_center_x = temp.shape[1] // 2
    image_center_y = temp.shape[0] // 2
    #计算平移距离
    shift_x = image_center_x - centroid_x[0]
    shift_y = image_center_y - centroid_y[0]

    ##平移图像
    rows, cols = temp.shape
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    if r == 1:
     for i in range(len(diffset)):
       diffset[i] = cv2.warpAffine(diffset[i], translation_matrix, (cols, rows))
    else:
       diffset = cv2.warpAffine(diffset, translation_matrix, (cols, rows))

    return diffset

def center_mass_caculate(image_array):

        """计算图像的质心。"""
        height, width = image_array.shape
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        total_mass = abs(image_array).sum()

        x_center = (abs(image_array).sum(axis=0) * x_coords).sum() / total_mass
        y_center = (abs(image_array).sum(axis=1) * y_coords).sum() / total_mass

        """计算移动距离"""
        x_shift = width / 2 - x_center
        y_shift = height / 2 - y_center
        return x_shift, y_shift

def center_mass_correct(image_array, x_shift, y_shift ):
    """将图像中的质心移动到指定位置。"""
    shifted_image = np.roll(image_array, int(round(y_shift)), axis=0)
    shifted_image = np.roll(shifted_image, int(round(x_shift)), axis=1)
    return shifted_image


if __name__ =='__main__':
    probe_r = cv2.imread('C:/Users/86364\Desktop/biyelunwen/image/4part/ADMM_pr.png',cv2.IMREAD_GRAYSCALE)
    # diffset = getphotos('C:/Users/86364/Desktop/yanshetuxixang/yanshetuxixang')
    # for i in range(len(diffset)):
    #     diffset[i] = imcrop.imcrop(diffset[i], 512)
    #     diffset[i] = diffset[i] / np.max(diffset[i])
    # probe_r = imcrop.imcrop(probe_r, 512)
    # probe_r = probe_r / np.max(probe_r)
    # grey = center_find(probe_r,probe_r,r = 0) #如果没有光斑probe_r可以直接用diffset[0]代替
    probe_r = center_mass_correct(probe_r)
    plt.figure()
    plt.imshow(abs(probe_r))
    plt.show()