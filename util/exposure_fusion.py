import numpy as np
from skimage import io

def load_image(image_path):
    """ 加载16位图像 """
    return io.imread(image_path)

def merge_hdr_images(images, exposure_times, base_exposure_time):
    """
    融合HDR图像。
    :param images: 图像列表（NumPy数组列表）。
    :param exposure_times: 相应图像的曝光时间列表。
    :param base_exposure_time: 基准曝光时间。
    :return: 合并后的HDR图像。
    """
    # 创建空的浮点数HDR图像
    hdr_image = np.zeros(images[0].shape, dtype=np.float32)

    for image, exp_time in zip(images, exposure_times):
        # 调整每个图像的像素值以反映其曝光时间
        hdr_image += (image.astype(np.float32) * base_exposure_time / exp_time)

    return hdr_image

