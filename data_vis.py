import os
import numpy as np
import matplotlib.pyplot as plt

# 设置文件夹路径
folder_path = '../data/data_test'
output_folder = '../data/data_to_show'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件名中是否包含字母 'd' 并且以 '.npy' 结尾
    if 'd' in filename and filename.endswith('.npy'):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 加载.npy文件
        array = np.load(file_path)
        max_value = np.max(array)
        data_to_show = array[0] * 256.0 / max_value

        # 将data_to_show保存为图片
        plt.figure(figsize=(10, 10))  # 可以设置图像大小
        plt.imshow(data_to_show, cmap='gray')  # 使用灰度色图
        plt.axis('off')  # 不显示坐标轴
        output_file_path = os.path.join(output_folder, filename.replace('.npy', '.png'))
        plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭图像以释放内存

print("Images have been saved.")