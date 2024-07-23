import numpy as np
from numpy import fft
import cv2
import matplotlib.pyplot as plt
import sys
import os
import copy
import random

sys.path.append("..")

from config import config
from util import lowPassFilter
from util.center_find import center_find2
from util.Setpinhole import Setpinhole
from Algorithms.various_PIE_engine import *
from WF import T_ADMM

def denoise(diffset, backnoise):
    for i in range(len(diffset)):
        diffset[i] = (diffset[i] - backnoise)
        diffset[i] = np.where(diffset[i] < 0, 0, diffset[i])
    return diffset

def imcrop(image, pixsum=100):
    [row, col] = np.shape(image)
    x_center = col // 2
    y_center = row // 2
    abstract = image[(y_center - pixsum // 2) : (y_center + pixsum // 2),
                     (x_center - pixsum // 2): (x_center + pixsum // 2)]
    return abstract

def plot_and_save_results(target, target_ph, probe_re, save_arg):
    save_path = save_arg.save_path
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig, ax = plt.subplots(2, 2, gridspec_kw={'wspace': 0, 'hspace': 0.5})
    plt.suptitle('Object and Probe Reconstruction')
    ax = ax.flatten()

    # 绘制重构对象的幅度和相位
    ax0 = ax[0].imshow(abs(target), cmap='gray')
    ax[0].set_title('Object Amplitude')
    fig.colorbar(ax0, ax=ax[0])

    ax1 = ax[1].imshow(target_ph, cmap='gray')
    ax[1].set_title('Object Phase')
    fig.colorbar(ax1, ax=ax[1])

    # 绘制探测波的幅度和相位
    ax2 = ax[2].imshow(np.abs(probe_re), cmap='jet')
    ax[2].set_title('Probe Amplitude')
    fig.colorbar(ax2, ax=ax[2])

    ax3 = ax[3].imshow(np.angle(probe_re), cmap='jet')
    ax[3].set_title('Probe Phase')
    fig.colorbar(ax3, ax=ax[3])

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局
    plt.savefig(save_path)
    plt.close(fig)

# PIE方法基类
class ComputeMethod:
    def compute(self, cfg, diffsets, probe_r, target_size, positions, 
                illuindy, illuindx, image_target, lis):
        raise NotImplementedError("Subclasses should implement this!")

# 'various_PIE'
class VariousPIEMethod(ComputeMethod):
    def compute(self, cfg, diffsets, probe_r, target_size, positions, 
                illuindy, illuindx, image_target, lis):
        target_re, probe_re, err = various_PIE(cfg['k'], diffsets, probe_r, target_size, positions, 
                                   illuindy, illuindx, 'Fourier', cfg['z2'], cfg['lamda'], 
                                   cfg['pix'], 'rPIE', image_target, lis)
        return target_re, probe_re, err

# 'T_ADMM'
class TADMMMethod(ComputeMethod):
    def compute(self, cfg, diffsets, probe_r, target_size, positions, 
                illuindy, illuindx, image_target, lis):
        target_re, probe_re, err = T_ADMM(cfg['k'], diffsets, probe_r, target_size, positions, 
                                   illuindy, illuindx, 'Fourier', cfg['z2'], cfg['lamda'], 
                                   cfg['pix'], image_target, lis, is_center_correct=1)
        return target_re, probe_re, err

class Admm:
    def __init__(self, args) -> None:
        self.cfg = config()
        p = np.load(args.positions_path)
        p = (p - (p[0] - 0.3)) * self.cfg['delta_r']
        positions_ = np.floor(np.array(p))
        positions_ = [[int(x) for x in positions_x] for positions_x in positions_]
        self.positions = copy.deepcopy(positions_)
        # 数据初始化
        self.backnoise = cv2.imread(args.backnoise_path, cv2.IMREAD_ANYDEPTH)
        self.diffset = self.diffset_init(np.load(args.diffset_path))
        probe_temp = Setpinhole(self.cfg['m'], self.cfg['m'], self.cfg['r'])
        self.probe_r, self.diffsets = self.probe_r_init(self.diffset)
        self.illuindy, self.illuindx = np.indices((self.probe_r.shape))
        self.image_target, self.target_size = self.limit_image_target_size(self.positions, self.diffsets)

    def diffset_init(self, diffset):
        diffset = denoise(diffset, self.backnoise)
        diffset = center_find2(diffset, r=1)  # 将衍射强度图的低频部分移到中心
        return diffset

    def probe_r_init(self, diffset):
        probe_r = np.zeros(shape=(self.cfg['m'], self.cfg['n']))
        diffsets = np.zeros(shape=(len(diffset), self.cfg['m'], self.cfg['m']))
        for i in range(len(diffset)):
            diffsets[i] = imcrop(diffset[i], self.cfg['m'])
            probe_r += np.sqrt(diffsets[i])

        probe_r = lowPassFilter.lowPassFiltering(probe_r, 17)  ##17
        probe_r = fft.ifft2(probe_r) / len(diffsets)
        probe_r = abs(fft.ifftshift(probe_r))
        probe_r = probe_r / np.max(probe_r)

        return probe_r, diffsets

    def limit_image_target_size(self, positions, diffsets):
        size_rows = max(positions[len(positions) - 1][0],
                        positions[len(positions) - 1][1]) + diffsets[0].shape[0] + 30
        size_cols = max(positions[len(positions) - 1][0],
                        positions[len(positions) - 1][1]) + diffsets[0].shape[1] + 30
        image_target = np.ones(shape=(size_rows, size_cols))
        return image_target, (size_rows, size_cols)

    def compute(self, method_name):
        """
        计算幅度和相位图像。
    
        输入: 计算的方法名    
        - various_PIE
        - T_ADMM

        输出:    
        - target: 重构的对象幅度数据
        - target_ph: 重构的对象相位数据
        - probe_re: 重构的波数据
        """
        # 构造随机读取顺序
        a = int(len(self.positions))
        lis = list(range(a))
        random.shuffle(lis)

        # 选择计算方法
        if method_name == 'various_PIE':
            self.compute_method = VariousPIEMethod()
        elif method_name == 'T_ADMM':
            self.compute_method = TADMMMethod()
        else:
            raise ValueError(f"Unknown method: {method_name}")

        target_re, probe_re, err = self.compute_method.compute(self.cfg, self.diffsets, self.probe_r,
                                                               self.target_size, self.positions,
                                                               self.illuindy, self.illuindx,
                                                               self.image_target, lis)
        
        target = abs(target_re) / np.max(abs(target_re))
        target_ph = np.angle(target_re)
        target_ph = cv2.rotate(target_ph , cv2.ROTATE_90_CLOCKWISE)
        target = cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE)

        return target, target_ph, probe_re