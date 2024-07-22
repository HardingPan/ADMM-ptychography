import numpy as np
from numpy import fft
import cv2
import sys
sys.path.append("..")

from config import config
from util import lowPassFilter
from util.center_find import center_find2
from util.Setpinhole import Setpinhole
from Algorithms.various_PIE_engine import *
from WF import T_ADMM

import copy
import random

def denoise(diffset, backnoise):
    for i in range(len(diffset)):
        diffset[i] = (diffset[i] - backnoise)
        diffset[i] = np.where(diffset[i] < 0, 0, diffset[i])
    return diffset

def imcrop(image,pixsum=100):
    [row, col] = np.shape(image)
    x_center = col // 2
    y_center = row // 2
    abstract = image[(y_center - pixsum // 2) : (y_center + pixsum // 2),
                (x_center - pixsum // 2): (x_center + pixsum // 2)]
    return abstract

class Admm():
    def __init__(self, args) -> None:
        self.cfg = config()
        p = np.load(args.positions_path)
        self.p = (p - (p[0] - 0.3)) * self.cfg['delta_r']
        positions_ = np.floor(np.array(p))
        positions_ =[[int(x) for x in positions_x] for  positions_x in positions_]
        self.positions = copy.deepcopy(positions_)
        # 数据初始化
        self.backnoise  = cv2.imread(args.backnoise_path, cv2.IMREAD_ANYDEPTH)
        self.diffset = self.diffset_init(np.load(args.diffset_path))
        probe_temp = Setpinhole(self.cfg['m'], self.cfg['m'], self.cfg['r'])
        self.probe_r, self.diffsets = self.probe_r_init(self.diffset)
        self.illuindy, self.illuindx = np.indices((self.probe_r.shape))
        self.image_target, self.target_size = self.limit_image_target_size(self.positions, self.diffsets)

    def diffset_init(self, diffset):
        diffset = denoise(diffset, self.backnoise)
        diffset = center_find2(diffset, r=1)  #将衍射强度图的低频部分移到中心
        return diffset
    
    def probe_r_init(self, diffset):
        probe_r =np.zeros(shape=(self.cfg['m'], self.cfg['n']))
        diffsets = np.zeros(shape=(len(diffset), \
                                   self.cfg['m'], self.cfg['m']))
        for i in range(len(diffset)):
            diffsets[i] = imcrop(diffset[i], self.cfg['m'])
            probe_r += np.sqrt(diffsets[i])

        probe_r = lowPassFilter.lowPassFiltering(probe_r,17)   ##17
        probe_r = fft.ifft2(probe_r)/len(diffsets)
        probe_r = abs(fft.ifftshift(probe_r))
        probe_r = probe_r / np.max(probe_r)

        return probe_r, diffsets
    
    def limit_image_target_size(self, positions, diffsets):
        size_rows = max(positions[len(positions) -1][0], \
                        positions[len(positions) -1][1]) + diffsets[0].shape[0] + 30
        size_cols = max(positions[len(positions) -1][0], \
                        positions[len(positions) -1][1]) + diffsets[0].shape[1] + 30
        image_target = np.ones(shape=(size_rows,size_cols))
        return image_target, (size_rows,size_cols)
    
    def compute(self):
        # 构造随机读取顺序
        a = (int)(len(self.positions))
        lis =list(range(a))
        random.shuffle(lis)

        object_re1, probe_re1, err = various_PIE(self.cfg['k'], self.diffsets, self.probe_r,
                                                 self.target_size, self.positions, 
                                                 self.illuindy, self.illuindx,
                                                 'Fourier', self.cfg['z2'], self.cfg['lamda'], self.cfg['pix'],
                                                 'rPIE', self.image_target, lis)

        object_re5, probe_re5, err = T_ADMM(self.cfg['k'], self.diffsets, self.probe_r, 
                                            self.target_size, self.positions,
                                            self.illuindy, self.illuindx,
                                            'Fourier', self.cfg['z2'], self.cfg['lamda'], self.cfg['pix'],
                                            self.image_target, lis, is_center_correct=1)
